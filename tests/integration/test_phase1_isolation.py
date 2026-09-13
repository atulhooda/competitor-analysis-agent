"""Phase 1 must work with GEMINI_API_KEY empty and must never load the Gemini SDK.

Runs in a fresh interpreter because other tests in this process import the SDK.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parents[2]

SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    import respx

    import app.cli, app.main, app.llm  # the llm package itself must not pull in the SDK
    from app.config import Settings
    from app.crawling.fetcher import PoliteFetcher
    from app.services.monitoring import MonitoringService
    from tests.fakesite import FakeClock, NOW, acme_competitor, mount_site, public_resolver

    settings = Settings(_env_file=None, app_env="test", crawler_min_delay_seconds=1.0)
    assert settings.gemini_api_key is None and not settings.llm_configured

    async def main():
        clock = FakeClock()
        async with PoliteFetcher(settings, resolver=public_resolver, clock=clock, sleep=clock.sleep) as f:
            with respx.mock(assert_all_called=False) as router:
                mount_site(router)
                return await MonitoringService(f, settings, now=lambda: NOW).scan(acme_competitor())

    result = asyncio.run(main())
    app.main.create_app(settings)
    assert result.status == "ok", result.errors
    assert len(result.items) >= 5
    leaked = sorted(m for m in sys.modules if m == "google.genai" or m.startswith("google.genai."))
    assert not leaked, f"Gemini SDK imported during Phase 1: {leaked[:5]}"
    print("PHASE1_OK", len(result.items))
    """
)


def test_phase1_runs_without_gemini_and_never_imports_its_sdk() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GEMINI_", "GOOGLE_"))}
    env["GEMINI_API_KEY"] = ""
    completed = subprocess.run(  # noqa: S603 - fixed script, current interpreter
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "PHASE1_OK" in completed.stdout
