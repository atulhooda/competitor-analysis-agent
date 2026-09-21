"""Pictures for published posts: where a cover comes from, and how big it really is.

- ``app.images.dimensions``: the native pixel size (and type) read from an image's own
  header. Both cover sources need it, and neither reports it.
- ``app.images.pexels``: the Pexels client and the deterministic rules that turn an
  article into a photo search. The generated-illustration source is the Gemini image
  model, which lives with the other model calls in ``app.llm``.

``app.services.covers`` chooses between them (``COVER_IMAGE_SOURCE``) and is the only
caller; nothing here touches the database or knows what a publication is.
"""

from app.images.dimensions import dimensions, sniff_mime
from app.images.pexels import (
    IMAGE_HOST,
    MAX_PHOTO_BYTES,
    MIN_PHOTO_WIDTH,
    QUERY_VERSION,
    PexelsAuthError,
    PexelsClient,
    PexelsError,
    PexelsPhoto,
    PexelsResponseError,
    PexelsTransientError,
    choose,
    parse_photos,
    photo_alt_text,
    search_queries,
)

__all__ = [
    "IMAGE_HOST",
    "MAX_PHOTO_BYTES",
    "MIN_PHOTO_WIDTH",
    "QUERY_VERSION",
    "PexelsAuthError",
    "PexelsClient",
    "PexelsError",
    "PexelsPhoto",
    "PexelsResponseError",
    "PexelsTransientError",
    "choose",
    "dimensions",
    "parse_photos",
    "photo_alt_text",
    "search_queries",
    "sniff_mime",
]
