"""Opt-in scan of a real website. Skipped unless LIVE_SCAN_URL is set.

    LIVE_SCAN_URL=https://www.example.com uv run pytest -m live

Only point this at sites whose Terms of Service allow automated access. The scan
obeys robots.txt and paces its requests like any other.
"""

import os

import pytest

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.domain.competitors import CompetitorConfig
from app.services.monitoring import MonitoringService

pytestmark = pytest.mark.live
LIVE_URL = os.environ.get("LIVE_SCAN_URL")


@pytest.mark.skipif(not LIVE_URL, reason="set LIVE_SCAN_URL to run the live scan")
async def test_live_scan() -> None:
    settings = Settings(_env_file=None, app_env="test")  # type: ignore[call-arg]
    competitor = CompetitorConfig(slug="live", name="Live", website=LIVE_URL)  # type: ignore[arg-type]
    async with PoliteFetcher(settings) as fetcher:
        result = await MonitoringService(fetcher, settings).scan(competitor, limit=3)
    assert result.status != "failed", result.errors
    assert result.items
