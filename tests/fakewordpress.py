"""An in-memory WordPress REST API (``/wp-json/wp/v2``) for tests, mounted with respx.

It behaves like WordPress where publishing depends on it: Application Password (Basic) auth,
drafts keep the slug they're given while a published post gets a unique one, terms are
matched by name and ``term_exists`` is reported, responses use the ``context=edit`` shape.
Failures can be injected per action, including the dangerous one: the change is saved but
the answer is lost (``"lost"``).
"""

import base64
import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx

USERNAME = "editor"
PASSWORD = "abcd EFGH 1234 ijkl MNOP 5678"  # a fake Application Password
BASE = "https://blog.example.com"


@dataclass
class FakeWordPress:
    base: str = BASE
    username: str = USERNAME
    password: str = PASSWORD
    posts: dict[int, dict[str, Any]] = field(default_factory=dict)
    categories: dict[int, str] = field(default_factory=lambda: {1: "Uncategorized", 5: "AI agents", 6: "Web & Analytics"})  # fmt: skip
    tags: dict[int, str] = field(default_factory=lambda: {11: "ai agents", 12: "customer support"})  # fmt: skip
    capabilities: dict[str, bool] = field(default_factory=lambda: {"edit_posts": True, "publish_posts": True, "manage_categories": True})  # fmt: skip
    # Per action ("index", "me", "get_posts", "get_post", "create_post", "update_post",
    # "get_terms", "create_term"): failures to inject, in order. A failure is an HTTP status
    # (401, 403, 400, 429, 500, ...), "timeout" (nothing saved), "lost" (saved, then the
    # answer times out), "connect", "malformed" (saved, answer isn't JSON) or "redirect".
    failures: dict[str, list[Any]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    next_id: int = 100

    # ── setup ────────────────────────────────────────────────────────────────

    def mount(self, router: respx.MockRouter) -> None:
        host = httpx.URL(self.base).host
        routes = [
            ("GET", r"/wp-json/?$", self._index),
            ("GET", r"/wp-json/wp/v2/users/me$", self._me),
            ("GET", r"/wp-json/wp/v2/posts$", self._list_posts),
            ("POST", r"/wp-json/wp/v2/posts$", self._create_post),
            ("GET", r"/wp-json/wp/v2/posts/(?P<post_id>\d+)$", self._get_post),
            ("POST", r"/wp-json/wp/v2/posts/(?P<post_id>\d+)$", self._update_post),
            ("GET", r"/wp-json/wp/v2/(?P<taxonomy>categories|tags)$", self._list_terms),
            ("POST", r"/wp-json/wp/v2/(?P<taxonomy>categories|tags)$", self._create_term),
            (
                "GET",
                r"/wp-json/wp/v2/(?P<taxonomy>categories|tags)/(?P<term_id>\d+)$",
                self._get_term,
            ),
        ]
        for method, path, handler in routes:
            router.route(method=method, host=host, path__regex=path).mock(side_effect=self._wrap(handler))  # fmt: skip

    def fail(self, action: str, *failures: Any) -> None:
        self.failures.setdefault(action, []).extend(failures)

    def add_post(self, *, slug: str, status: str = "publish", content: str = "<p>Someone else's post</p>", title: str = "Existing post") -> int:  # fmt: skip
        post_id = self._new_id()
        self.posts[post_id] = self._make(post_id, {"slug": slug, "status": status, "content": content, "title": title})  # fmt: skip
        return post_id

    @property
    def mutations(self) -> list[tuple[str, str]]:
        return [c for c in self.calls if c[0] == "POST"]

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _wrap(self, handler: Any) -> Any:
        def side_effect(request: httpx.Request, **kwargs: str) -> httpx.Response:
            self.calls.append((request.method, request.url.path.removeprefix("/wp-json/")))
            self.requests.append(request)
            if request.url.path.rstrip("/") != "/wp-json" and not self._authorized(request):
                return _error(401, "incorrect_password", "The provided password is an invalid application password.")  # fmt: skip
            response: httpx.Response = handler(request, **kwargs)
            return response

        return side_effect

    def _authorized(self, request: httpx.Request) -> bool:
        expected = "Basic " + base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return request.headers.get("authorization") == expected

    def _injected(self, action: str, request: httpx.Request) -> Any:
        pending = self.failures.get(action)
        return pending.pop(0) if pending else None

    def _failure_response(self, failure: Any, request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out (fake)", request=request)
        if failure == "connect":
            raise httpx.ConnectError("connection refused (fake)", request=request)
        if failure == "redirect":
            return httpx.Response(301, headers={"location": "https://elsewhere.example.net/wp-json/"})  # fmt: skip
        if failure == 429:
            return _error(429, "rest_too_many_requests", "Too many requests", headers={"retry-after": "1"})  # fmt: skip
        return _error(int(failure), {401: "incorrect_password", 403: "rest_cannot_create", 400: "rest_invalid_param", 500: "internal_server_error", 503: "service_unavailable"}.get(int(failure), "error"), f"fake {failure}")  # fmt: skip

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _make(self, post_id: int, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": post_id,
            "status": fields.get("status", "draft"),
            "slug": fields.get("slug", ""),
            "title": fields.get("title", ""),
            "content": fields.get("content", ""),
            "excerpt": fields.get("excerpt", ""),
            "categories": fields.get("categories") or [1],
            "tags": fields.get("tags") or [],
            "author": fields.get("author", 1),
        }

    def _json(self, post: dict[str, Any]) -> dict[str, Any]:
        link = f"{self.base}/{post['slug']}/" if post["status"] == "publish" else f"{self.base}/?p={post['id']}"  # fmt: skip
        return {
            "id": post["id"], "status": post["status"], "slug": post["slug"], "link": link,
            "title": {"raw": post["title"], "rendered": post["title"]},
            "content": {"raw": post["content"], "rendered": post["content"]},
            "excerpt": {"raw": post["excerpt"], "rendered": post["excerpt"]},
            "categories": post["categories"], "tags": post["tags"], "author": post["author"],
        }  # fmt: skip

    def _unique_slug(self, slug: str, post_id: int) -> str:
        taken = {p["slug"] for p in self.posts.values() if p["id"] != post_id and p["status"] != "trash"}  # fmt: skip
        candidate, n = slug, 2
        while candidate in taken:
            candidate, n = f"{slug}-{n}", n + 1
        return candidate

    def _apply(self, post_id: int | None, body: dict[str, Any]) -> dict[str, Any] | httpx.Response:
        status = body.get("status", "draft")
        if status not in ("draft", "pending", "publish", "private", "future"):
            return _error(400, "rest_invalid_param", "Invalid parameter(s): status")
        for key, taxonomy in (("categories", self.categories), ("tags", self.tags)):
            unknown = [i for i in body.get(key, []) if i not in taxonomy]
            if unknown:
                return _error(400, "rest_invalid_param", f"Invalid parameter(s): {key}")
        if status == "publish" and not self.capabilities.get("publish_posts"):
            return _error(403, "rest_cannot_publish", "Sorry, you are not allowed to publish posts as this user.")  # fmt: skip
        if post_id is None:
            post_id = self._new_id()
            post = self._make(post_id, {})
            self.posts[post_id] = post
        post = self.posts[post_id]
        for key in ("title", "content", "excerpt", "categories", "tags", "author", "slug"):
            if key in body:
                post[key] = body[key]
        post["status"] = status
        if status == "publish":  # WordPress makes a published slug unique; drafts keep theirs
            post["slug"] = self._unique_slug(post["slug"], post_id)
        return post

    def _mutate(self, action: str, request: httpx.Request, post_id: int | None) -> httpx.Response:
        failure = self._injected(action, request)
        if failure is not None and failure not in ("lost", "malformed"):
            return self._failure_response(failure, request)
        result = self._apply(post_id, json.loads(request.content))
        if isinstance(result, httpx.Response):
            return result
        if failure == "lost":  # saved, but the answer never arrives
            raise httpx.ReadTimeout("timed out after the post was saved (fake)", request=request)
        if failure == "malformed":
            return httpx.Response(200, text="<html>proxy error</html>")
        return httpx.Response(201 if post_id is None else 200, json=self._json(result))

    # ── routes ───────────────────────────────────────────────────────────────

    def _index(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("index", request)
        if failure is not None:
            return self._failure_response(failure, request)
        return httpx.Response(200, json={"name": "Example Blog", "url": self.base, "namespaces": ["oembed/1.0", "wp/v2"]})  # fmt: skip

    def _me(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("me", request)
        if failure is not None:
            return self._failure_response(failure, request)
        return httpx.Response(200, json={"id": 1, "name": "Editor", "slug": "editor", "capabilities": self.capabilities})  # fmt: skip

    def _list_posts(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("get_posts", request)
        if failure is not None:
            return self._failure_response(failure, request)
        params = request.url.params
        statuses = set((params.get("status") or "publish").split(","))
        found = [p for p in self.posts.values() if p["status"] in statuses]
        if params.get("slug"):
            found = [p for p in found if p["slug"] == params["slug"]]
        if params.get("search"):
            term = params["search"].lower()
            found = [p for p in found if term in p["content"].lower() or term in p["title"].lower()]
        return httpx.Response(200, json=[self._json(p) for p in found])

    def _get_post(self, request: httpx.Request, post_id: str) -> httpx.Response:
        failure = self._injected("get_post", request)
        if failure is not None:
            return self._failure_response(failure, request)
        post = self.posts.get(int(post_id))
        if post is None:
            return _error(404, "rest_post_invalid_id", "Invalid post ID.")
        return httpx.Response(200, json=self._json(post))

    def _create_post(self, request: httpx.Request) -> httpx.Response:
        if not self.capabilities.get("edit_posts"):
            return _error(403, "rest_cannot_create", "Sorry, you are not allowed to create posts as this user.")  # fmt: skip
        return self._mutate("create_post", request, None)

    def _update_post(self, request: httpx.Request, post_id: str) -> httpx.Response:
        if int(post_id) not in self.posts:
            return _error(404, "rest_post_invalid_id", "Invalid post ID.")
        return self._mutate("update_post", request, int(post_id))

    def _terms(self, taxonomy: str) -> dict[int, str]:
        return self.categories if taxonomy == "categories" else self.tags

    def _list_terms(self, request: httpx.Request, taxonomy: str) -> httpx.Response:
        failure = self._injected("get_terms", request)
        if failure is not None:
            return self._failure_response(failure, request)
        term = (request.url.params.get("search") or "").lower()
        rows = [{"id": i, "name": html.escape(n, quote=False), "slug": _slug(n)} for i, n in self._terms(taxonomy).items() if term in n.lower()]  # fmt: skip
        return httpx.Response(200, json=rows)

    def _get_term(self, request: httpx.Request, taxonomy: str, term_id: str) -> httpx.Response:
        name = self._terms(taxonomy).get(int(term_id))
        if name is None:
            return _error(404, "rest_term_invalid", "Term does not exist.")
        return httpx.Response(200, json={"id": int(term_id), "name": html.escape(name, quote=False), "slug": _slug(name)})  # fmt: skip

    def _create_term(self, request: httpx.Request, taxonomy: str) -> httpx.Response:
        failure = self._injected("create_term", request)
        if failure is not None and failure != "lost":
            return self._failure_response(failure, request)
        name = json.loads(request.content)["name"]
        terms = self._terms(taxonomy)
        existing = next((i for i, n in terms.items() if n.lower() == name.lower()), None)
        if existing is not None:
            return _error(400, "term_exists", "A term with the name provided already exists.", data={"status": 400, "term_id": existing})  # fmt: skip
        term_id = self._new_id()
        terms[term_id] = name
        if failure == "lost":
            raise httpx.ReadTimeout("timed out after the term was saved (fake)", request=request)
        return httpx.Response(201, json={"id": term_id, "name": html.escape(name, quote=False), "slug": _slug(name)})  # fmt: skip


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _error(status: int, code: str, message: str, *, data: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> httpx.Response:  # fmt: skip
    return httpx.Response(status, json={"code": code, "message": message, "data": data or {"status": status}}, headers=headers)  # fmt: skip
