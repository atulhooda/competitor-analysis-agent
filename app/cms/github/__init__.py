"""GitHub publishing (Phase 8): the site's own repository is the publishing target."""

from app.cms.github.client import GitHubClient
from app.cms.github.mdx import MDXDocument, SiteConfig, compose, validate
from app.cms.github.publisher import GitHubPublishingAdapter, SiteClient

__all__ = [
    "GitHubClient",
    "GitHubPublishingAdapter",
    "MDXDocument",
    "SiteClient",
    "SiteConfig",
    "compose",
    "validate",
]
