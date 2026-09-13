"""Content classification enums."""

from enum import StrEnum


class ContentType(StrEnum):
    HOMEPAGE = "homepage"
    BLOG_POST = "blog_post"
    PRICING = "pricing"
    CASE_STUDY = "case_study"
    PRODUCT = "product"
    LANDING_PAGE = "landing_page"
    RESOURCE = "resource"
    DOCS = "docs"
    CHANGELOG = "changelog"
    PRESS = "press"
    LISTING = "listing"
    CAREERS = "careers"
    LEGAL = "legal"
    OTHER = "other"


# Noise for competitor monitoring unless a competitor config opts back in.
DEFAULT_EXCLUDED_TYPES: frozenset[ContentType] = frozenset(
    {ContentType.LISTING, ContentType.CAREERS, ContentType.LEGAL}
)

# Types that usually carry a publication date on the page. In a dated scan, undated
# candidates of these types are still worth fetching to discover their date.
DATED_CONTENT_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.BLOG_POST,
        ContentType.CASE_STUDY,
        ContentType.PRESS,
        ContentType.CHANGELOG,
        ContentType.RESOURCE,
    }
)


class DiscoverySource(StrEnum):
    HOMEPAGE = "homepage"
    TRACKED = "tracked"
    FEED = "feed"
    SITEMAP = "sitemap"


class DateSource(StrEnum):
    """Where an item's publication date came from, most trustworthy first."""

    STRUCTURED_DATA = "structured_data"
    META = "meta"
    FEED = "feed"
    SITEMAP_NEWS = "sitemap_news"
    PAGE = "page"
