"""A fake GitHub API, a fake Vercel and a fake website for the publishing tests: an in-memory
repository (branches, files, pull requests, deployments) behind respx, so the adapter runs
end to end with no network. Tests inject failures per action and decide how deployments
go (success, failure, never reported) and whether previews are protected.

The fake site serves ``/blog/<slug>`` from the base branch's content directory as the real
site would (title, first heading, canonical, published metadata), and each preview
deployment serves its branch's file at ``https://preview-<n>.vercel.app/blog/<slug>``.

Binary files (cover images) are kept in ``blobs``, branch by branch, beside the text
``trees``: they are committed, branched and merged exactly like a post, and ``image()``
reads one back.
"""

import base64
import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx

API = "https://api.github.com"
REPO = "siddharthpathania/engageo-website"
OWNER = REPO.split("/")[0]
TOKEN = "github_pat_fake_token_0123456789abcdefghijklmnop"  # a fake test value, not a credential
SITE = "https://www.engageoagency.com"
CONTENT_DIR = "src/content/blog"
BASE = "main"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass
class FakePullRequest:
    number: int
    head: str
    base: str
    title: str
    body: str
    state: str = "open"
    merged_at: str | None = None
    merge_commit_sha: str | None = None


@dataclass
class FakeDeployment:
    id: int
    sha: str
    environment: str  # Preview | Production
    branch: str
    state: str  # pending | in_progress | success | failure
    url: str | None = None


@dataclass
class FakeGitHub:
    repo: str = REPO
    token: str = TOKEN
    push: bool = True
    # branch → {path: content}; the base branch starts with the site's existing posts.
    trees: dict[str, dict[str, str]] = field(default_factory=dict)
    # branch → {path: bytes}: the files that aren't text (cover images).
    blobs: dict[str, dict[str, bytes]] = field(default_factory=dict)
    branch_heads: dict[str, str] = field(default_factory=dict)
    # The base branch's head when a branch was created or reset: what tells a branch that
    # is merely ahead from one left behind by a squash merge ("diverged").
    branch_bases: dict[str, str] = field(default_factory=dict)
    pulls: dict[int, FakePullRequest] = field(default_factory=dict)
    deployments: list[FakeDeployment] = field(default_factory=list)
    # How a deployment goes once created: "success" (default), "failure", "pending"
    # (never finishes), "none" (Vercel never reports one), "later" (pending on the first
    # poll, success on the next).
    preview_outcome: str = "success"
    production_outcome: str = "success"
    preview_protected: bool = False
    bypass_secret: str | None = None  # accepted in the x-vercel-protection-bypass header
    # Per action ("get_repo", "get_ref", "create_ref", "get_contents", "put_contents",
    # "list_pulls", "create_pull", "get_pull", "merge", "deployments", "site", "preview"):
    # failures to inject, in order: an HTTP status, "timeout" (nothing happened), "lost"
    # (it happened, then the answer timed out), "connect".
    failures: dict[str, list[Any]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    next_pull: int = 60
    next_deployment: int = 900
    commits: int = 0
    polls: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trees.setdefault(BASE, {})
        self.blobs.setdefault(BASE, {})
        self.branch_heads.setdefault(BASE, "b" * 40)

    # ── setup ────────────────────────────────────────────────────────────────

    def add_post(self, slug: str, text: str, *, branch: str = BASE) -> None:
        self.trees.setdefault(branch, {})[f"{CONTENT_DIR}/{slug}.mdx"] = text
        self.branch_heads.setdefault(branch, self._sha(branch))

    def fail(self, action: str, *failures: Any) -> None:
        self.failures.setdefault(action, []).extend(failures)

    @property
    def mutations(self) -> list[tuple[str, str]]:
        return [c for c in self.calls if c[0] in ("POST", "PUT", "PATCH", "DELETE")]

    def open_pulls(self) -> list[FakePullRequest]:
        return [p for p in self.pulls.values() if p.state == "open"]

    def file(self, slug: str, *, branch: str = BASE) -> str | None:
        return self.trees.get(branch, {}).get(f"{CONTENT_DIR}/{slug}.mdx")

    def image(self, path: str, *, branch: str = BASE) -> bytes | None:
        return self.blobs.get(branch, {}).get(path)

    def images(self, *, branch: str = BASE) -> list[str]:
        return sorted(self.blobs.get(branch, {}))

    def mount(self, router: respx.MockRouter) -> None:
        r = f"/repos/{re.escape(self.repo)}"
        routes = [
            ("GET", rf"^{r}$", self._get_repo),
            ("GET", rf"^{r}/branches/(?P<branch>[^/]+)$", self._get_branch),
            ("GET", rf"^{r}/git/ref/heads/(?P<branch>.+)$", self._get_ref),
            ("POST", rf"^{r}/git/refs$", self._create_ref),
            ("PATCH", rf"^{r}/git/refs/heads/(?P<branch>.+)$", self._update_ref),
            ("GET", rf"^{r}/compare/(?P<spec>.+)$", self._compare),
            ("GET", rf"^{r}/contents/(?P<path>.+)$", self._get_contents),
            ("PUT", rf"^{r}/contents/(?P<path>.+)$", self._put_contents),
            ("GET", rf"^{r}/pulls$", self._list_pulls),
            ("POST", rf"^{r}/pulls$", self._create_pull),
            ("GET", rf"^{r}/pulls/(?P<number>\d+)$", self._get_pull),
            ("PUT", rf"^{r}/pulls/(?P<number>\d+)/merge$", self._merge),
            ("GET", rf"^{r}/deployments$", self._list_deployments),
            ("GET", rf"^{r}/deployments/(?P<deployment>\d+)/statuses$", self._deployment_statuses),
        ]
        host = httpx.URL(API).host
        for method, path, handler in routes:
            router.route(method=method, host=host, path__regex=path).mock(side_effect=self._wrap(handler))  # fmt: skip
        router.route(host=host).mock(return_value=_error(404, "Not Found"))
        router.route(host=httpx.URL(SITE).host).mock(side_effect=self._site)
        router.route(host__regex=r".*\.vercel\.app$").mock(side_effect=self._preview)

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _wrap(self, handler: Any) -> Any:
        def side_effect(request: httpx.Request, **kwargs: str) -> httpx.Response:
            self.calls.append((request.method, request.url.path))
            self.requests.append(request)
            if request.headers.get("authorization") != f"Bearer {self.token}":
                return _error(401, "Bad credentials")
            response: httpx.Response = handler(request, **kwargs)
            return response

        return side_effect

    def _injected(self, action: str) -> Any:
        pending = self.failures.get(action)
        return pending.pop(0) if pending else None

    def _failure(self, failure: Any, request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out (fake)", request=request)
        if failure == "connect":
            raise httpx.ConnectError("connection refused (fake)", request=request)
        if failure == 429:
            return _error(429, "API rate limit exceeded", headers={"retry-after": "1"})
        if failure == "secondary":
            return _error(403, "You have exceeded a secondary rate limit", headers={"retry-after": "2"})  # fmt: skip
        return _error(int(failure), {401: "Bad credentials", 403: "Resource not accessible by personal access token", 404: "Not Found", 409: "is at a different sha", 422: "Validation Failed", 500: "Server Error", 502: "Bad Gateway"}.get(int(failure), "error"))  # fmt: skip

    def _sha(self, seed: str) -> str:
        self.commits += 1
        return (f"{self.commits:08x}" + re.sub(r"[^0-9a-f]", "0", seed.lower()))[:40].ljust(40, "0")

    # ── repository ───────────────────────────────────────────────────────────

    def _get_repo(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("get_repo")
        if failure is not None:
            return self._failure(failure, request)
        return httpx.Response(200, json={"full_name": self.repo, "private": True, "default_branch": BASE, "permissions": {"admin": False, "maintain": False, "push": self.push, "triage": self.push, "pull": True}})  # fmt: skip

    def _get_branch(self, request: httpx.Request, branch: str) -> httpx.Response:
        if branch not in self.trees:
            return _error(404, "Branch not found")
        return httpx.Response(200, json={"name": branch, "commit": {"sha": self.branch_heads[branch]}})  # fmt: skip

    def _get_ref(self, request: httpx.Request, branch: str) -> httpx.Response:
        failure = self._injected("get_ref")
        if failure is not None:
            return self._failure(failure, request)
        if branch not in self.trees:
            return _error(404, "Not Found")
        return httpx.Response(200, json={"ref": f"refs/heads/{branch}", "object": {"type": "commit", "sha": self.branch_heads[branch]}})  # fmt: skip

    def _create_ref(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("create_ref")
        if failure is not None and failure != "lost":
            return self._failure(failure, request)
        body = json.loads(request.content)
        branch = str(body["ref"]).removeprefix("refs/heads/")
        if branch in self.trees:
            return _error(422, "Reference already exists")
        source = next((b for b, sha in self.branch_heads.items() if sha == body["sha"]), BASE)
        self.trees[branch] = dict(self.trees[source])
        self.blobs[branch] = dict(self.blobs.get(source, {}))
        self.branch_heads[branch] = self.branch_heads[source]
        self.branch_bases[branch] = self.branch_heads[BASE]
        if failure == "lost":
            raise httpx.ReadTimeout("timed out after the branch was created (fake)", request=request)  # fmt: skip
        return httpx.Response(201, json={"ref": body["ref"], "object": {"sha": self.branch_heads[branch]}})  # fmt: skip

    def _get_contents(self, request: httpx.Request, path: str) -> httpx.Response:
        failure = self._injected("get_contents")
        if failure is not None:
            return self._failure(failure, request)
        ref = request.url.params.get("ref") or BASE
        branch = ref if ref in self.trees else next((b for b, sha in self.branch_heads.items() if sha == ref), None)  # fmt: skip
        if branch is None:
            return _error(404, "No commit found for the ref")
        tree = self.trees[branch]
        blobs = self.blobs.get(branch, {})
        if path in tree:
            return httpx.Response(200, json=self._file_json(path, tree[path].encode()))
        if path in blobs:
            return httpx.Response(200, json=self._file_json(path, blobs[path]))
        entries = sorted({p for p in (*tree, *blobs) if p.startswith(path + "/")})
        if entries:
            return httpx.Response(200, json=[{"name": p.rsplit("/", 1)[-1], "path": p, "type": "file", "sha": _content_sha(tree[p].encode() if p in tree else blobs[p])} for p in entries])  # fmt: skip
        return _error(404, "Not Found")

    def _put_contents(self, request: httpx.Request, path: str) -> httpx.Response:
        failure = self._injected("put_contents")
        if failure is not None and failure != "lost":
            return self._failure(failure, request)
        body = json.loads(request.content)
        branch = body.get("branch") or BASE
        if branch not in self.trees:
            return _error(404, "Branch not found")
        # Images are stored as bytes, everything else as the text the site's build reads.
        store: dict[str, Any] = self.blobs.setdefault(branch, {}) if path.lower().endswith(IMAGE_SUFFIXES) else self.trees[branch]  # fmt: skip
        current = store.get(path)
        existing = None if current is None else _content_sha(current if isinstance(current, bytes) else current.encode())  # fmt: skip
        if current is not None and body.get("sha") != existing:
            return _error(409, f"{path} does not match {body.get('sha')}")
        if current is None and body.get("sha"):
            return _error(422, "sha given for a new file")
        data = base64.b64decode(body["content"])
        store[path] = data if path.lower().endswith(IMAGE_SUFFIXES) else data.decode()
        self.branch_heads[branch] = self._sha(branch + path)
        self._new_deployment(branch)
        if failure == "lost":
            raise httpx.ReadTimeout("timed out after the file was committed (fake)", request=request)  # fmt: skip
        return httpx.Response(200 if body.get("sha") else 201, json={"content": self._file_json(path, data), "commit": {"sha": self.branch_heads[branch]}})  # fmt: skip

    def _file_json(self, path: str, data: bytes) -> dict[str, Any]:
        return {"type": "file", "name": path.rsplit("/", 1)[-1], "path": path, "sha": _content_sha(data), "content": base64.b64encode(data).decode()}  # fmt: skip

    # ── pull requests ────────────────────────────────────────────────────────

    def _pull_json(self, pr: FakePullRequest) -> dict[str, Any]:
        return {
            "number": pr.number, "state": pr.state, "title": pr.title, "body": pr.body,
            "html_url": f"https://github.com/{self.repo}/pull/{pr.number}",
            "head": {"ref": pr.head, "sha": self.branch_heads.get(pr.head, "0" * 40)},
            "base": {"ref": pr.base}, "merged": pr.merged_at is not None,
            "merged_at": pr.merged_at, "merge_commit_sha": pr.merge_commit_sha,
        }  # fmt: skip

    def _list_pulls(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("list_pulls")
        if failure is not None:
            return self._failure(failure, request)
        params = request.url.params
        state = params.get("state") or "open"
        found = [p for p in self.pulls.values() if state == "all" or p.state == state]
        if params.get("head"):
            head = params["head"].split(":", 1)[-1]
            found = [p for p in found if p.head == head]
        if params.get("base"):
            found = [p for p in found if p.base == params["base"]]
        found.sort(key=lambda p: p.number, reverse=True)
        return httpx.Response(200, json=[self._pull_json(p) for p in found])

    def _create_pull(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("create_pull")
        if failure is not None and failure != "lost":
            return self._failure(failure, request)
        body = json.loads(request.content)
        head, base = str(body["head"]), str(body["base"])
        if head not in self.trees:
            return _error(422, "Validation Failed", errors=[{"message": f"head branch {head} not found"}])  # fmt: skip
        if any(p.head == head and p.state == "open" for p in self.pulls.values()):
            return _error(422, "Validation Failed", errors=[{"message": f"A pull request already exists for {OWNER}:{head}."}])  # fmt: skip
        if self.trees[head] == self.trees[base] and self.blobs.get(head, {}) == self.blobs.get(base, {}):  # fmt: skip
            return _error(422, "Validation Failed", errors=[{"message": f"No commits between {base} and {head}"}])  # fmt: skip
        self.next_pull += 1
        pr = FakePullRequest(self.next_pull, head, base, str(body["title"]), str(body.get("body") or ""))  # fmt: skip
        self.pulls[pr.number] = pr
        if failure == "lost":
            raise httpx.ReadTimeout("timed out after the pull request was created (fake)", request=request)  # fmt: skip
        return httpx.Response(201, json=self._pull_json(pr))

    def _get_pull(self, request: httpx.Request, number: str) -> httpx.Response:
        failure = self._injected("get_pull")
        if failure is not None:
            return self._failure(failure, request)
        pr = self.pulls.get(int(number))
        if pr is None:
            return _error(404, "Not Found")
        return httpx.Response(200, json=self._pull_json(pr))

    def _merge(self, request: httpx.Request, number: str) -> httpx.Response:
        failure = self._injected("merge")
        if failure is not None and failure != "lost":
            return self._failure(failure, request)
        pr = self.pulls.get(int(number))
        if pr is None:
            return _error(404, "Not Found")
        if pr.state != "open":
            return _error(405, "Pull Request is not mergeable")
        if not self.push:
            return _error(403, "Resource not accessible by personal access token")
        body = json.loads(request.content)
        if body.get("merge_method") != "squash":
            return _error(405, "Merge method not allowed")
        if self._status_of(pr.head) == "diverged":
            return _error(405, "Pull Request has merge conflicts")
        self.trees[pr.base].update(self.trees[pr.head])
        self.blobs.setdefault(pr.base, {}).update(self.blobs.get(pr.head, {}))
        sha = self._sha(f"merge{pr.number}")
        self.branch_heads[pr.base] = sha
        pr.state, pr.merged_at, pr.merge_commit_sha = "closed", "2026-09-16T10:00:00Z", sha
        self._new_deployment(pr.base)
        if failure == "lost":
            raise httpx.ReadTimeout("timed out after the merge (fake)", request=request)
        return httpx.Response(200, json={"sha": sha, "merged": True, "message": "Pull Request successfully merged"})  # fmt: skip

    def _status_of(self, branch: str) -> str:
        """What GitHub's compare endpoint would say about ``base...branch``."""
        if branch == BASE or branch not in self.trees:
            return "identical"
        head, base = self.branch_heads[branch], self.branch_heads[BASE]
        if head == base:
            return "identical"
        cut = self.branch_bases.get(branch, base)
        own = head != cut  # commits of its own
        if cut == base:
            return "ahead" if own else "identical"
        return "diverged" if own else "behind"

    def _compare(self, request: httpx.Request, spec: str) -> httpx.Response:
        failure = self._injected("compare")
        if failure is not None:
            return self._failure(failure, request)
        base, _, head = spec.partition("...")
        if head not in self.trees or base not in self.trees:
            return _error(404, "Not Found")
        return httpx.Response(200, json={"status": self._status_of(head), "ahead_by": 0, "behind_by": 0})  # fmt: skip

    def _update_ref(self, request: httpx.Request, branch: str) -> httpx.Response:
        """Moving a branch, as `git push --force` does: the tree comes with it."""
        failure = self._injected("update_ref")
        if failure is not None:
            return self._failure(failure, request)
        if branch not in self.trees:
            return _error(404, "Not Found")
        if not self.push:
            return _error(403, "Resource not accessible by personal access token")
        body = json.loads(request.content)
        sha = str(body["sha"])
        source = next((b for b, head in self.branch_heads.items() if head == sha), None)
        if source is None:
            return _error(422, "Object does not exist")
        self.trees[branch] = dict(self.trees[source])
        self.blobs[branch] = dict(self.blobs.get(source, {}))
        self.branch_heads[branch] = sha
        self.branch_bases[branch] = self.branch_heads[BASE]
        self.mutations.append(("PATCH", f"git/refs/heads/{branch}"))
        return httpx.Response(200, json={"ref": f"refs/heads/{branch}", "object": {"sha": sha}})

    # ── deployments (what Vercel reports to GitHub) ──────────────────────────

    def _new_deployment(self, branch: str) -> None:
        production = branch == BASE
        outcome = self.production_outcome if production else self.preview_outcome
        if outcome == "none":
            return
        self.next_deployment += 1
        state = "success" if outcome == "success" else ("failure" if outcome == "failure" else "pending")  # fmt: skip
        url = SITE if production else f"https://preview-{self.next_deployment}.vercel.app"
        self.deployments.append(FakeDeployment(self.next_deployment, self.branch_heads[branch], "Production" if production else "Preview", branch, state, url))  # fmt: skip

    def _list_deployments(self, request: httpx.Request) -> httpx.Response:
        failure = self._injected("deployments")
        if failure is not None:
            return self._failure(failure, request)
        sha = request.url.params.get("sha")
        found = [d for d in self.deployments if not sha or d.sha == sha]
        return httpx.Response(200, json=[{"id": d.id, "sha": d.sha, "environment": d.environment, "ref": d.branch} for d in found])  # fmt: skip

    def _deployment_statuses(self, request: httpx.Request, deployment: str) -> httpx.Response:
        d = next((x for x in self.deployments if x.id == int(deployment)), None)
        if d is None:
            return _error(404, "Not Found")
        self.polls[d.id] = self.polls.get(d.id, 0) + 1
        outcome = self.production_outcome if d.environment == "Production" else self.preview_outcome  # fmt: skip
        if outcome == "later" and self.polls[d.id] >= 2:
            d.state = "success"
        status = {"state": d.state, "environment": d.environment, "log_url": f"https://vercel.com/deployments/{d.id}", "description": "Deployment has completed" if d.state == "success" else d.state}  # fmt: skip
        if d.state == "success":
            status["environment_url"] = d.url
            status["target_url"] = d.url
        return httpx.Response(200, json=[status])

    # ── the site and its previews ────────────────────────────────────────────

    def _page(self, tree: dict[str, str], path: str, host: str) -> httpx.Response:
        slug = path.removeprefix("/blog/").strip("/")
        text = tree.get(f"{CONTENT_DIR}/{slug}.mdx")
        if not path.startswith("/blog/") or text is None:
            return httpx.Response(404, text="<html><title>404</title><body>Not found</body></html>")
        match = _FRONT.match(text)
        fields = _yaml(match.group(1)) if match else {}
        body = text[match.end() :] if match else text
        if fields.get("draft") in ("true", True):
            return httpx.Response(404, text="<html><title>404</title></html>")
        headings = "".join(f"<h2>{html.escape(line[3:])}</h2>" for line in body.splitlines() if line.startswith("## "))  # fmt: skip
        page = (
            f"<html><head><title>{html.escape(str(fields.get('title', '')))} | Engageo</title>"
            f'<link rel="canonical" href="{host}/blog/{slug}"/>'
            f'<meta property="article:published_time" content="{html.escape(str(fields.get("publishedAt", "")))}"/></head>'
            f"<body><h1>{html.escape(str(fields.get('title', '')))}</h1><article>{headings}</article></body></html>"
        )
        return httpx.Response(200, text=page)

    def _site(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, f"site:{request.url.path}"))
        self.requests.append(request)
        failure = self._injected("site")
        if failure is not None:
            return self._failure(failure, request)
        if request.url.path == "/sitemap.xml":
            locs = [f"{SITE}/", f"{SITE}/services", f"{SITE}/clinics", f"{SITE}/contact", f"{SITE}/pricing", f"{SITE}/blog"] + [f"{SITE}/blog/{p.rsplit('/', 1)[-1].removesuffix('.mdx')}" for p in self.trees[BASE]]  # fmt: skip
            return httpx.Response(200, text="<urlset>" + "".join(f"<url><loc>{u}</loc></url>" for u in locs) + "</urlset>")  # fmt: skip
        return self._page(self.trees[BASE], request.url.path, SITE)

    def _preview(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, f"preview:{request.url.host}{request.url.path}"))
        self.requests.append(request)
        failure = self._injected("preview")
        if failure is not None:
            return self._failure(failure, request)
        if self.preview_protected and not (self.bypass_secret and request.headers.get("x-vercel-protection-bypass") == self.bypass_secret):  # fmt: skip
            return httpx.Response(401, text="<html><body>Authentication Required</body></html>", headers={"set-cookie": "_vercel_sso_nonce=x"})  # fmt: skip
        deployment = next((d for d in self.deployments if d.url and httpx.URL(d.url).host == request.url.host), None)  # fmt: skip
        if deployment is None:
            return httpx.Response(404, text="DEPLOYMENT_NOT_FOUND")
        return self._page(self.trees.get(deployment.branch, {}), request.url.path, f"https://{request.url.host}")  # fmt: skip


def _yaml(text: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _content_sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(data).hexdigest()  # noqa: S324 - a fake id, not security


def _error(status: int, message: str, *, errors: list[dict[str, Any]] | None = None, headers: dict[str, str] | None = None) -> httpx.Response:  # fmt: skip
    body: dict[str, Any] = {"message": message, "documentation_url": "https://docs.github.com/rest"}
    if errors:
        body["errors"] = errors
    return httpx.Response(status, json=body, headers=headers)


__all__ = ["API", "BASE", "CONTENT_DIR", "IMAGE_SUFFIXES", "REPO", "SITE", "TOKEN", "FakeDeployment", "FakeGitHub", "FakePullRequest"]  # fmt: skip
