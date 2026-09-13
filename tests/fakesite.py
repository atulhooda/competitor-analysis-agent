"""A small fake competitor site (acme.test) served through respx, for offline tests.

Timeline (NOW = 2026-09-13): two recent posts in the feed, one sitemap-only recent post,
one old post, a robots-disallowed page, a tracked pricing page, and a stale case study.
"""

import json
from datetime import UTC, datetime

import respx

from app.config import Settings
from app.domain.competitors import CompetitorConfig

BASE = "https://acme.test"
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
PUBLIC_IP = "93.184.216.34"  # any globally routable address satisfies the SSRF guard


async def public_resolver(host: str) -> list[str]:
    return [PUBLIC_IP]


class FakeClock:
    """Monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_settings(**overrides: object) -> Settings:
    """Settings isolated from the developer's .env file and environment."""
    values: dict[str, object] = {"app_env": "test", "crawler_min_delay_seconds": 1.0}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def acme_competitor(**overrides: object) -> CompetitorConfig:
    values: dict[str, object] = {
        "slug": "acme",
        "name": "Acme",
        "website": f"{BASE}/",
        "tracked_pages": [f"{BASE}/pricing"],
    }
    values.update(overrides)
    return CompetitorConfig.model_validate(values)


_PARAGRAPHS = (
    "Customer support teams are adopting AI agents to resolve routine tickets faster, while "
    "escalating complex conversations to people who can apply judgment and empathy.",
    "The biggest gains come from automating password resets, order status questions and refund "
    "eligibility checks, which together make up a large share of incoming volume.",
    "Teams that succeed start with a narrow scope, measure resolution quality weekly, and expand "
    "the agent's responsibilities only after customers report consistently good outcomes.",
    "Pricing models are shifting too: vendors increasingly charge per resolution instead of per "
    "seat, which aligns cost with value but makes monthly budgets harder to predict.",
    "Finally, strong handoff design matters. Customers should never repeat themselves when a "
    "conversation moves from an automated agent to a human specialist on the team.",
)
_PARAGRAPH = " ".join(_PARAGRAPHS[:2])


def article_html(
    path: str,
    title: str,
    *,
    published: str | None = None,
    meta_published: str | None = None,
    author: str = "Jane Doe",
) -> str:
    jsonld = ""
    if published:
        data = {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "headline": title,
            "datePublished": published,
            "author": {"@type": "Person", "name": author},
        }
        jsonld = f'<script type="application/ld+json">{json.dumps(data)}</script>'
    meta = (
        f'<meta property="article:published_time" content="{meta_published}">'
        if meta_published
        else ""
    )
    body = "".join(f"<p>{paragraph}</p>" for paragraph in _PARAGRAPHS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title} | Acme</title><meta name="description" content="{title} on the Acme blog.">
<meta property="og:type" content="article">{meta}{jsonld}
<link rel="canonical" href="{BASE}{path}"></head>
<body><nav><a href="/">Home</a> <a href="/blog">Blog</a></nav>
<article><h1>{title}</h1>{body}<h2>Key takeaways</h2>
<ul><li>Faster first response</li><li>Lower cost per ticket</li></ul></article>
<footer>© Acme Inc.</footer></body></html>"""


def simple_page(title: str, body: str, *, jsonld_type: str | None = None) -> str:
    jsonld = (
        f'<script type="application/ld+json">{json.dumps({"@type": jsonld_type})}</script>'
        if jsonld_type
        else ""
    )
    paragraphs = "".join(f"<p>{body}</p>" for _ in range(4))
    return f"""<!doctype html><html lang="en"><head><title>{title} | Acme</title>{jsonld}</head>
<body><main><h1>{title}</h1>{paragraphs}</main></body></html>"""


HOME_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Acme — AI customer support platform</title>
<meta name="description" content="Acme resolves support tickets with AI agents.">
<meta property="og:type" content="website">
<link rel="alternate" type="application/rss+xml" title="Acme blog" href="/blog/feed.xml">
<script type="application/ld+json">{{"@context":"https://schema.org","@type":"Organization","name":"Acme"}}</script>
</head><body><nav><a href="/blog">Blog</a> <a href="/pricing">Pricing</a>
<a href="/customers/globex">Customers</a> <a href="/careers">Careers</a></nav>
<main><h1>Resolve support tickets with AI agents</h1>
{"".join(f"<p>{p}</p>" for p in _PARAGRAPHS[:3])}</main></body></html>"""

PRICING_HTML = simple_page(
    "Pricing",
    "Starter costs $29 per agent per month. Growth costs $79 per agent per month with "
    "automation workflows, analytics, and priority support for growing teams.",
)

ROBOTS_TXT = """User-agent: *
Disallow: /private/
Crawl-delay: 2

Sitemap: https://acme.test/sitemap_index.xml
"""

FEED_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Acme blog</title><link>{BASE}/blog</link>
<item><title>AI Support Agents: A Practical Guide</title>
  <link>{BASE}/blog/ai-support-agents?utm_source=rss&amp;utm_medium=feed</link>
  <pubDate>Thu, 10 Sep 2026 08:00:00 GMT</pubDate><category>AI</category><category>Support</category></item>
<item><title>Our New Pricing</title><link>{BASE}/blog/new-pricing</link>
  <pubDate>Sat, 05 Sep 2026 09:30:00 GMT</pubDate></item>
<item><title>An Old Post</title><link>{BASE}/blog/old-post</link>
  <pubDate>Mon, 01 Jun 2026 10:00:00 GMT</pubDate></item>
<item><title>Cross-post</title><link>https://medium.com/@acme/cross-post</link>
  <pubDate>Fri, 11 Sep 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""

SITEMAP_INDEX = f"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>{BASE}/sitemap-posts.xml</loc><lastmod>2026-09-11</lastmod></sitemap>
<sitemap><loc>{BASE}/sitemap-pages.xml</loc><lastmod>2026-09-12</lastmod></sitemap>
<sitemap><loc>{BASE}/sitemap-archive.xml</loc><lastmod>2025-01-01</lastmod></sitemap>
</sitemapindex>"""

SITEMAP_POSTS = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>{BASE}/blog/ai-support-agents</loc><lastmod>2026-09-10</lastmod></url>
<url><loc>{BASE}/blog/new-pricing</loc><lastmod>2026-09-05</lastmod></url>
<url><loc>{BASE}/blog/old-post</loc><lastmod>2026-06-01</lastmod></url>
<url><loc>{BASE}/blog/sitemap-only-post</loc><lastmod>2026-09-11T07:00:00+00:00</lastmod></url>
<url><loc>{BASE}/blog/tag/ai</loc><lastmod>2026-09-11</lastmod></url>
<url><loc>{BASE}/private/internal-post</loc><lastmod>2026-09-12</lastmod></url>
</urlset>"""

SITEMAP_PAGES = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>{BASE}/</loc></url>
<url><loc>{BASE}/pricing</loc><lastmod>2026-09-09</lastmod></url>
<url><loc>{BASE}/customers/globex</loc><lastmod>2026-08-01</lastmod></url>
<url><loc>{BASE}/features/automation</loc></url>
<url><loc>{BASE}/careers</loc><lastmod>2026-09-12</lastmod></url>
<url><loc>{BASE}/legal/privacy</loc><lastmod>2026-09-12</lastmod></url>
</urlset>"""

SITEMAP_ARCHIVE = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>{BASE}/blog/2019/01/ancient-post</loc><lastmod>2019-01-15</lastmod></url>
</urlset>"""


def mount_site(
    router: respx.MockRouter,
    *,
    robots: str = ROBOTS_TXT,
    feed: str | None = None,
    sitemap_posts: str | None = None,
    pages: dict[str, str | int] | None = None,
) -> dict[str, respx.Route]:
    """Register every acme.test URL on ``router``; returns routes keyed by path.

    ``feed``/``sitemap_posts`` replace those documents; ``pages`` maps a path to replacement
    HTML or to an HTTP status code (e.g. 404), to simulate a site changing between scans.
    """
    routes: dict[str, respx.Route] = {}

    def html(path: str, body: str) -> None:
        routes[path] = router.get(BASE + path).respond(200, html=body)

    def xml(path: str, body: str, content_type: str = "application/xml") -> None:
        routes[path] = router.get(BASE + path).respond(
            200, content=body.encode(), headers={"content-type": content_type}
        )

    routes["/robots.txt"] = router.get(BASE + "/robots.txt").respond(200, text=robots)
    html("/", HOME_HTML)
    xml("/blog/feed.xml", feed or FEED_XML, "application/rss+xml; charset=utf-8")
    xml("/sitemap_index.xml", SITEMAP_INDEX)
    xml("/sitemap-posts.xml", sitemap_posts or SITEMAP_POSTS)
    xml("/sitemap-pages.xml", SITEMAP_PAGES)
    xml("/sitemap-archive.xml", SITEMAP_ARCHIVE)
    html(
        "/blog/ai-support-agents",
        article_html(
            "/blog/ai-support-agents",
            "AI Support Agents: A Practical Guide",
            published="2026-09-10T08:00:00Z",
        ),
    )
    html(
        "/blog/new-pricing",
        article_html("/blog/new-pricing", "Our New Pricing", meta_published="2026-09-05T09:30:00Z"),
    )
    html(
        "/blog/old-post",
        article_html("/blog/old-post", "An Old Post", published="2026-06-01T10:00:00Z"),
    )
    html(
        "/blog/sitemap-only-post",
        article_html(
            "/blog/sitemap-only-post", "Launch Week Recap", published="2026-09-11T07:00:00Z"
        ),
    )
    html(
        "/blog/2019/01/ancient-post",
        article_html(
            "/blog/2019/01/ancient-post", "Ancient Post", published="2019-01-15T10:00:00Z"
        ),
    )
    html("/pricing", PRICING_HTML)
    html(
        "/customers/globex",
        simple_page("How Globex cut response times", _PARAGRAPH, jsonld_type="Article"),
    )
    html("/features/automation", simple_page("Automation", _PARAGRAPH))
    html("/blog/tag/ai", simple_page("Posts tagged AI", _PARAGRAPH))
    html("/careers", simple_page("Careers", _PARAGRAPH))
    html("/legal/privacy", simple_page("Privacy", _PARAGRAPH))
    html("/private/internal-post", simple_page("Internal", "This must never be requested."))
    for path, replacement in (pages or {}).items():
        if isinstance(replacement, int):
            routes[path] = router.get(BASE + path).respond(replacement)
        else:
            html(path, replacement)
    return routes


def feed_with(*items: tuple[str, str, str]) -> str:
    """The standard feed plus extra (path, title, RFC 822 pubDate) items at the top."""
    extra = "".join(
        f"<item><title>{title}</title><link>{BASE}{path}</link><pubDate>{date}</pubDate></item>"
        for path, title, date in items
    )
    return FEED_XML.replace(
        "<link>https://acme.test/blog</link>", f"<link>{BASE}/blog</link>{extra}", 1
    )


def sitemap_posts_with(lastmods: dict[str, str], extra_urls: tuple[str, ...] = ()) -> str:
    """The standard posts sitemap with some lastmod values changed and extra URLs added."""
    body = SITEMAP_POSTS
    for path, lastmod in lastmods.items():
        start = body.index(f"<loc>{BASE}{path}</loc>")
        end = body.index("</url>", start)
        body = body[:start] + f"<loc>{BASE}{path}</loc><lastmod>{lastmod}</lastmod>" + body[end:]
    extra = "".join(f"<url><loc>{BASE}{path}</loc></url>" for path in extra_urls)
    return body.replace("</urlset>", f"{extra}</urlset>")
