# Competitor Analysis Agent

An AI-powered competitor intelligence and content generation agent. It monitors competitors'
websites (and later their social channels), keeps a history of what they publish, finds content
opportunities, and generates and publishes original blog posts.

> **Status: Phase 2 of 10: persisted competitor history.** Scans are deterministic and
> rule-based (no LLM calls) and are now stored in PostgreSQL: the system remembers what each
> competitor has published, detects new and changed content, and answers questions about it.
> See [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the architecture and roadmap.

## What it does today

**Monitoring (Phase 1).** For each competitor, a scan:

1. reads **robots.txt** (RFC 9309) and obeys it: disallowed URLs are never requested or recorded;
2. fetches the **homepage** and any **tracked pages** (e.g. pricing);
3. discovers content from **RSS/Atom feeds** and **sitemaps** (indexes, gzip, news sitemaps);
4. filters candidates by scope, URL patterns, content type and date window;
5. fetches the most relevant pages and extracts title, description, author, dates, categories,
   tags, headings, word count, a content hash, and the main text as Markdown;
6. classifies each page deterministically (`blog_post`, `pricing`, `case_study`, `product`, …).

**History (Phase 2).**

- **Everything discovered is remembered.** Every in-scope URL found in a feed or sitemap is
  recorded with `first_seen_at` / `last_seen_at`, even if it isn't fetched yet.
- **Scans are incremental.** Pages already captured are re-fetched only when a feed or sitemap
  reports a newer date, or via conditional GET (`ETag` / `Last-Modified`), plus a small budget of
  stale pages per scan. Each scan's budget goes to new content first.
- **Versions.** A new immutable version (title, headings, text, word count, raw HTML) is stored
  only when the extracted main text actually changes.
- **Change detection.** Deterministic change events:

  | Event | When |
  |---|---|
  | `new` | A URL is discovered for the first time |
  | `updated` | The main text changes. `minor` means the title is unchanged and ≤ 12 words changed |
  | `pricing_changed` | The prices on a pricing page change |
  | `removed` | The page returns 404/410. Absence from a sitemap is *not* removal |
  | `restored` | A removed page comes back |

- **Baseline.** A competitor's first scan records its back catalogue without reporting it as `new`.
- **Duplicates.** URLs that redirect to, or declare as canonical, another page are marked
  `duplicate` rather than counted as separate content.
- **Runs.** Every scan is recorded with its statistics and an audit trail (skipped URLs, errors).
  Scans of the same competitor never overlap (Postgres advisory lock).

### Dates: what they mean

| Field | Meaning |
|---|---|
| `published_at` | When the competitor published the page. **Only set when reliably known**, with its source (`published_at_source`), most trusted first: `structured_data` (JSON-LD), `meta` (`article:published_time`), `feed` (RSS/Atom pubDate), `sitemap_news`, and `page` (trafilatura's heuristics, accepted for articles only). A less trusted source never overwrites a more trusted one. |
| `first_seen_at` / `last_seen_at` | When *this system* first and last saw the URL. **Never used as a publication date.** |
| `modified_at` | Last modification the page itself declares (JSON-LD, meta tags, feed `updated`) |
| `sitemap_lastmod` | The sitemap's `lastmod`: a hint that the page changed, **never a publication date** |

Queries such as "published since" only match items with a reliable `published_at`. Undated
items are never guessed into a window.

### Compliance

- Honest bot identity (`CRAWLER_USER_AGENT`, with a contact URL). No browser spoofing.
- robots.txt `Disallow` and `Crawl-delay` are obeyed. An unreachable robots.txt (5xx) means
  "crawl nothing".
- One request at a time per host, at least `CRAWLER_MIN_DELAY_SECONDS` apart. Incremental scans
  make far fewer requests than a full re-crawl.
- Retries only for 408/429/5xx/network errors, honoring `Retry-After`.
- **401/403 and bot challenges stop immediately.** Logins, CAPTCHAs, rate limits and other
  access controls are never bypassed.
- The crawler refuses URLs that resolve to private or internal addresses (SSRF guard).
- You are responsible for reviewing each competitor site's Terms of Service before monitoring it.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 16. The included `docker-compose.yml` runs one locally; any PostgreSQL 16
  reachable through `DATABASE_URL` works.

## Setup (fresh checkout)

```bash
uv sync                                                     # install dependencies
cp .env.example .env                                        # local settings (gitignored)
docker compose up -d db                                     # PostgreSQL on 127.0.0.1:5433
uv run python -m app db upgrade                             # create the schema (migrations)
cp config/competitors.example.yaml config/competitors.yaml  # your competitors (gitignored)
uv run python -m app competitors import                     # load them into the database
uv run python -m app check                                  # validate everything
```

The database is the source of truth for competitors; the YAML file is an import format.
Only `slug`, `name` and `website` are required:

```yaml
competitors:
  - slug: acme
    name: Acme Inc.
    website: https://www.acme.com
    tracked_pages: [https://www.acme.com/pricing]
```

See [`config/competitors.example.yaml`](config/competitors.example.yaml) for every option.

## Usage

### CLI

```bash
# Competitors
uv run python -m app competitors                     # list (with item counts and last scan)
uv run python -m app competitors import              # create/update from COMPETITORS_FILE
uv run python -m app competitors deactivate acme     # stop monitoring (history is kept)

# Scanning (persisted, incremental)
uv run python -m app scan acme                       # scan one competitor and record history
uv run python -m app scan --all --since 7d           # every active competitor
uv run python -m app scan acme --dry-run             # Phase 1 behavior: nothing is saved

# History
uv run python -m app content acme --published-since 7d    # what did Acme publish this week?
uv run python -m app content acme --new-only              # first seen after the baseline
uv run python -m app changes acme --since 30d             # new / updated / pricing / removed
uv run python -m app activity acme --weeks 12             # weekly publication and change counts
uv run python -m app runs acme                            # recent scans
uv run python -m app run 42                               # one scan with its audit events

# Database
uv run python -m app db upgrade                      # apply migrations
uv run python -m app db current                      # show the schema revision
```

Most commands accept `--json`. `--since` / `--published-since` accept `24h`, `7d`, `2w` or an
ISO date.

### API

```bash
uv run uvicorn app.main:create_app --factory --port 8000
```

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Status, database schema state, LLM configuration (never secrets) |
| GET / POST | `/api/v1/competitors` | List (`?include_inactive=true`) / create |
| GET / PATCH | `/api/v1/competitors/{slug}` | Read / partial update (name, website, options, `active`) |
| POST | `/api/v1/competitors/{slug}/scans` | Start a scan: `202` plus a run to poll. With `?wait=true` it runs synchronously. Body: `{"since": "7d", "limit": 10}` |
| GET | `/api/v1/competitors/{slug}/activity` | Weekly publication and change counts (`?weeks=12`) |
| GET | `/api/v1/content` | Filters: `competitor`, `content_type`, `status`, `published_since`/`_until`, `first_seen_since`, `new_only`, `q`, `limit`/`offset` |
| GET | `/api/v1/content/{id}` | One item with its current version (`?include_text=true`) |
| GET | `/api/v1/content/{id}/versions` | Version history |
| GET | `/api/v1/changes` | Change events: `competitor`, `change_type`, `since`, `include_minor` |
| GET | `/api/v1/runs`, `/api/v1/runs/{id}` | Scan runs, with their audit events |

`/api/v1/*` requires the `X-API-Key` header whenever `API_KEY` is set. Outside development,
requests are refused until it is. Interactive docs are served at `/docs`.

> Changed in Phase 2: the Phase 1 endpoint `POST /api/v1/competitors/{slug}/scan` is now
> `POST /api/v1/competitors/{slug}/scans`, and scans are persisted.

## Data model

PostgreSQL, managed with Alembic migrations (`migrations/`), in separate layers:

| Layer | Tables | Contents |
|---|---|---|
| Configuration | `competitors` | Who is monitored, and how (options stored as validated JSON) |
| Raw | `raw_documents` | HTML exactly as fetched (gzip), stored only for captured versions |
| Normalized | `content_items`, `content_versions`, `change_events` | One row per URL (lifecycle + reliable dates); immutable snapshots; the change log |
| Operations | `runs`, `run_events` | What ran, when, with what result, and why URLs were skipped |

Nothing in these tables is produced by an LLM. AI-generated analysis (Phase 3) and
recommendations (Phase 4+) get their own tables.

## LLM provider: Google Gemini

**Google Gemini is the project's primary and only LLM provider.** All LLM access goes through a
provider-agnostic interface in [`app/llm/`](app/llm/):

```python
from app.llm import LLMRequest, get_llm

llm = get_llm()  # configured from the environment
reply = await llm.generate(LLMRequest(prompt="Summarize ...", reasoning_effort="low"))
topics = await llm.generate_structured(LLMRequest(prompt="..."), TopicList)  # a Pydantic model
```

- `app/llm/base.py`: the `LLMProvider` interface, request and response types.
- `app/llm/gemini.py`: `GeminiProvider`, built on the official
  [`google-genai`](https://pypi.org/project/google-genai/) SDK and the **Gemini Interactions
  API**. It is the only module that imports the Gemini SDK, and it is loaded lazily.
- Calls are stateless (`store=false`). The SDK handles retries for 429/5xx. Errors are mapped to
  provider-neutral types.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | From Phase 3 | *(empty)* | Gemini API key. Create one in [Google AI Studio](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` | Model ID. Leave empty for the default. |
| `LLM_TIMEOUT_SECONDS` | No | `120` | Per-request timeout |
| `LLM_MAX_RETRIES` | No | `2` | SDK retries for 429/5xx/timeouts |

- The default, `gemini-3.8-flash`, was the latest stable model in
  [Google's model list](https://ai.google.dev/gemini-api/docs/models) when this was written;
  change it with `GEMINI_MODEL`, never in code.
- The key is passed to the SDK explicitly (a `GOOGLE_API_KEY` in your shell is not used). It is
  stored as a secret and never logged, returned by the API, or printed by `check`.
- Use a **billing-enabled (paid tier) key in production**. At the time of writing, Google's
  terms allow content sent on the unpaid tier to be used to improve Google's products; check the
  current terms.

| Phase | Uses Gemini? |
|---|---|
| 1 · Website monitoring | **No.** Deterministic. |
| 2 · Persisted history (current) | **No.** Storage, incremental scans and change detection are deterministic. **`GEMINI_API_KEY` may be empty.** |
| 3 · Competitor analysis | Yes: topics, formats, audience, positioning, change summaries |
| 4 · Opportunity detection | Yes: relevance judgments and angles (scores are deterministic) |
| 5 · Blog research and generation | Yes |
| 6 · SEO, editing, fact-checking, quality | Yes |
| 7–10 · Publishing, scheduling, social, dashboard | Publishing itself never uses an LLM |

## Configuration reference

All settings are environment variables (or `.env`); see [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test` or `production` |
| `API_KEY` | *(empty)* | Required `X-API-Key` for `/api/v1/*` outside development |
| `DATABASE_URL` | local docker-compose DB | `postgresql+psycopg://USER:PASSWORD@HOST:PORT/DB`; stored as a secret, printed masked |
| `STORE_RAW_HTML` | `true` | Keep gzipped raw HTML for each captured version |
| `COMPETITORS_FILE` | `config/competitors.yaml` | YAML for `competitors import` |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `false` | Logging (JSON for production) |
| `CRAWLER_USER_AGENT` | `CompetitorMonitorBot/0.1 (+repo URL)` | Bot identity; include contact info |
| `CRAWLER_MIN_DELAY_SECONDS` | `3` | Minimum gap per host (robots `Crawl-delay` can raise it) |
| `CRAWLER_DEFAULT_SCAN_LIMIT` | `25` | New or changed pages fetched per scan |
| `CRAWLER_REVISIT_LIMIT` / `_AFTER_DAYS` | `5` / `7` | Stale captured pages re-checked per scan |
| `CRAWLER_MAX_SITEMAP_FILES` / `_URLS` | `20` / `10000` | Sitemap discovery bounds |

## Development

```bash
docker compose up -d db              # the tests need PostgreSQL
uv run pytest                        # offline: no external network, no real Gemini calls
uv run ruff check . && uv run ruff format --check .
uv run mypy app                      # strict type checking
LIVE_SCAN_URL=https://www.example.com uv run pytest -m live   # opt-in real-site scan
```

- Each test session creates a fresh database from the Alembic migrations (proving a fresh
  checkout builds the schema), and tables are emptied between tests. Point `TEST_DATABASE_URL`
  at another server if needed. Database tests are **skipped with a warning** when PostgreSQL
  isn't reachable, except in CI (`REQUIRE_DB_TESTS=1`), where that fails the run.
- HTTP, including the Gemini API, is mocked with [respx]; an autouse guard blocks every
  non-loopback socket.
- CI runs lint, format, type checks and tests against a PostgreSQL service
  (`.github/workflows/ci.yml`).

[respx]: https://lundberg.github.io/respx/

## Project layout

```
app/
  main.py, cli.py, config.py   FastAPI app factory, CLI, settings + competitor YAML import
  core/                        errors, logging, time utilities
  domain/                      shared types (content, competitors, scans, history views)
  crawling/                    fetcher, robots.txt, rate limiting, SSRF guard, sitemaps,
                               feeds, HTML signals, extraction, classification (no LLM)
  services/
    monitoring.py              scan orchestration (incremental when given known pages)
    scans.py                   run lifecycle: locking, scanning, recording
    history.py                 records scans: discovered URLs, versions, change events
    change_detection.py        word-level diffs and price-change detection
  db/                          models (by layer), sessions, advisory locks, queries, migrations
  llm/                         provider-agnostic LLM interface + Gemini provider (Phase 3+)
  api/                         HTTP routes and schemas
migrations/                    Alembic migrations
config/                        competitors.example.yaml
tests/                         unit, integration (incl. PostgreSQL) and opt-in live tests
```

## Licensing

Small patterns are adapted from MIT-licensed projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
