# Competitor Analysis Agent

An AI-powered competitor intelligence and content generation agent. It monitors competitors'
websites (and later their social channels), keeps a history of what they publish, finds content
opportunities, and generates and publishes original blog posts.

> **Status: Phase 1 of 10: competitor website monitoring.** Scanning is deterministic and
> rule-based: no database and no LLM calls. See [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for
> the architecture and the phased roadmap.

## What Phase 1 does

For each configured competitor, a scan:

1. reads **robots.txt** (RFC 9309) and obeys it: disallowed URLs are never requested;
2. fetches the **homepage** and any **tracked pages** (e.g. pricing);
3. discovers content from **RSS/Atom feeds** (configured, advertised, or conventional paths)
   and **sitemaps** (configured, declared in robots.txt, or `/sitemap.xml`), including
   sitemap indexes and gzipped sitemaps;
4. filters candidates by scope, URL patterns, content type and date window, and ranks them.
   Publication dates outrank sitemap `lastmod`;
5. fetches the top pages within a limit and extracts title, description, author, dates,
   categories, tags, language, word count, a content hash, and the main text as Markdown;
6. classifies each page deterministically (`blog_post`, `pricing`, `case_study`, `product`,
   `landing_page`, `resource`, `press`, `changelog`, `docs`, …), with a human-readable reason.

Results are returned (CLI or API); they are not stored yet. Persistence and change detection
arrive in Phase 2.

### Compliance

- Honest bot identity (`CRAWLER_USER_AGENT`, with a contact URL). No browser spoofing.
- robots.txt `Disallow` and `Crawl-delay` are obeyed. An unreachable robots.txt (5xx) means
  "crawl nothing".
- One request at a time per host, at least `CRAWLER_MIN_DELAY_SECONDS` apart.
- Retries only for 408/429/5xx/network errors, honoring `Retry-After`.
- **401/403 and bot challenges stop immediately.** Logins, CAPTCHAs, rate limits and other
  access controls are never bypassed.
- The crawler refuses URLs that resolve to private or internal addresses (SSRF guard).
- You are responsible for reviewing each competitor site's Terms of Service before monitoring it.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                                                   # create .venv and install dependencies
cp .env.example .env                                      # local settings (gitignored)
cp config/competitors.example.yaml config/competitors.yaml  # your competitors (gitignored)
uv run python -m app check                                # validate configuration
```

Edit `config/competitors.yaml`. Only `slug`, `name` and `website` are required:

```yaml
competitors:
  - slug: acme
    name: Acme Inc.
    website: https://www.acme.com
    tracked_pages: [https://www.acme.com/pricing]
```

See [`config/competitors.example.yaml`](config/competitors.example.yaml) for every option
(explicit feeds and sitemaps, extra domains, include/exclude patterns, excluded types).

## Usage

### CLI

```bash
uv run python -m app competitors                        # list configured competitors
uv run python -m app scan acme --since 7d               # what did Acme publish in the last 7 days?
uv run python -m app scan acme --since 2026-09-01 --limit 10
uv run python -m app scan acme --json --include-text    # full result, with extracted Markdown
```

`--since` accepts `24h`, `7d`, `2w` or an ISO date. `--limit` caps how many discovered pages
are fetched; the homepage and tracked pages are always fetched. Scans are paced per host, so a
scan takes roughly `requests × delay` seconds.

### API

```bash
uv run uvicorn app.main:create_app --factory --port 8000
```

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Status, version and LLM configuration (never the key) |
| GET | `/api/v1/competitors` | Configured competitors |
| POST | `/api/v1/competitors/{slug}/scan` | Body: `{"since": "7d", "limit": 10, "include_text": false}` |

```bash
curl -X POST localhost:8000/api/v1/competitors/acme/scan \
  -H 'content-type: application/json' -H "X-API-Key: $API_KEY" -d '{"since": "7d"}'
```

`/api/v1/*` requires the `X-API-Key` header whenever `API_KEY` is set. Outside development
(`APP_ENV=production`), requests are refused until `API_KEY` is configured. Interactive docs
are served at `/docs`.

## LLM provider: Google Gemini

**Google Gemini is the project's primary and only LLM provider.** No other LLM provider is
used. All LLM access goes through a provider-agnostic interface in [`app/llm/`](app/llm/):

```python
from app.llm import LLMRequest, get_llm

llm = get_llm()  # configured from the environment
reply = await llm.generate(LLMRequest(prompt="Summarize ...", reasoning_effort="low"))
topics = await llm.generate_structured(LLMRequest(prompt="..."), TopicList)  # a Pydantic model
```

- `app/llm/base.py`: the `LLMProvider` interface, request and response types.
- `app/llm/gemini.py`: `GeminiProvider`, built on the official
  [`google-genai`](https://pypi.org/project/google-genai/) SDK and the **Gemini Interactions
  API** (Google's recommended API for new projects). It is the only module that imports the
  Gemini SDK, and it is loaded lazily.
- Calls are stateless (`store=false`). Retries for 429/5xx are handled by the SDK (honoring
  `Retry-After`). Errors are mapped to provider-neutral types (`LLMRateLimitError`,
  `LLMAuthenticationError`, …).

### Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | From Phase 3 | *(empty)* | Gemini API key. Create one in [Google AI Studio](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` | Model ID. Leave empty for the default. |
| `LLM_TIMEOUT_SECONDS` | No | `120` | Per-request timeout |
| `LLM_MAX_RETRIES` | No | `2` | SDK retries for 429/5xx/timeouts |

Put these in `.env` (never commit it) or in the environment:

```bash
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.8-flash
```

- The default, `gemini-3.8-flash`, was the latest stable model in
  [Google's model list](https://ai.google.dev/gemini-api/docs/models) when this was written.
  Change it with `GEMINI_MODEL`, never in code. Later phases can also pick a model per request.
- The key is passed to the SDK explicitly, so a `GOOGLE_API_KEY` in your shell is not used.
- The key is stored as a secret and is never logged, returned by the API, or printed by
  `python -m app check`.
- Use a **billing-enabled (paid tier) key in production**. At the time of writing, Google's
  terms allow content sent on the unpaid tier to be used to improve Google's products; check
  the current terms.

### Which phases use Gemini

| Phase | Uses Gemini? |
|---|---|
| 1 · Website monitoring (current) | **No.** Crawling, robots.txt, sitemaps, RSS, extraction, URL normalization and classification are deterministic. **`GEMINI_API_KEY` may be empty.** |
| 2 · Persist content | No |
| 3 · Competitor analysis | Yes: topics, formats, audience, positioning, change summaries |
| 4 · Opportunity detection | Yes: relevance judgments and angles (trend math and scores are deterministic) |
| 5 · Blog research and generation | Yes: research (Google Search grounding), outline, drafting |
| 6 · SEO, editing, fact-checking, quality | Yes |
| 7–10 · Publishing, scheduling, social, dashboard | Publishing itself never uses an LLM |

Phase 1 is tested to run with `GEMINI_API_KEY` empty and to never import the Gemini SDK
(`tests/integration/test_phase1_isolation.py`).

## Configuration reference

All settings are environment variables (or `.env`); see [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test` or `production` |
| `API_KEY` | *(empty)* | Required `X-API-Key` for `/api/v1/*` outside development |
| `COMPETITORS_FILE` | `config/competitors.yaml` | Competitor list |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `false` | Logging (JSON for production) |
| `CRAWLER_USER_AGENT` | `CompetitorMonitorBot/0.1 (+repo URL)` | Bot identity; include contact info |
| `CRAWLER_MIN_DELAY_SECONDS` | `3` | Minimum gap per host (robots `Crawl-delay` can raise it) |
| `CRAWLER_DEFAULT_SCAN_LIMIT` | `25` | Discovered pages fetched per scan |
| `CRAWLER_MAX_RETRIES` | `2` | Retries for 408/429/5xx/network errors |
| `CRAWLER_MAX_SITEMAP_FILES` / `_URLS` | `20` / `10000` | Sitemap discovery bounds |

## Development

```bash
uv run pytest                        # all tests: offline, no real Gemini calls
uv run ruff check . && uv run ruff format --check .
uv run mypy app                      # strict type checking
LIVE_SCAN_URL=https://www.example.com uv run pytest -m live   # opt-in real-site scan
```

- Tests never touch the network. HTTP, including the Gemini API, is mocked with
  [respx](https://lundberg.github.io/respx/), and an autouse guard blocks real sockets.
- CI runs lint, format, type checks and tests on every push (`.github/workflows/ci.yml`).

## Project layout

```
app/
  main.py, cli.py, config.py   FastAPI app factory, CLI, settings + competitor config
  core/                        errors, logging, time utilities
  domain/                      shared types (content types, competitor config, scan results)
  crawling/                    fetcher, robots.txt, rate limiting, SSRF guard, sitemaps,
                               feeds, HTML signals, extraction, classification (no LLM)
  services/monitoring.py       the Phase 1 scan orchestration
  llm/                         provider-agnostic LLM interface + Gemini provider (Phase 3+)
  api/                         HTTP routes and schemas
config/                        competitors.example.yaml
tests/                         unit, integration and opt-in live tests
```

## Licensing

Small patterns are adapted from MIT-licensed projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
