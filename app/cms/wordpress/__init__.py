"""The WordPress adapter: REST API client and publisher."""

from app.cms.wordpress.client import WordPressClient
from app.cms.wordpress.publisher import WordPressPublisher, wordpress_payload

__all__ = ["WordPressClient", "WordPressPublisher", "wordpress_payload"]
