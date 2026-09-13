# Competitor Analysis Agent

An AI-powered competitor intelligence and content generation agent. It monitors competitors'
websites (and later their social channels), keeps a history of what they publish, finds content
opportunities, and generates and publishes original blog posts.

> **Status: Phase 3 of 10: AI competitor intelligence.** Scans stay deterministic (no LLM)
> and are stored in PostgreSQL. Gemini now analyzes what competitors publish: topics, formats,
> audiences, intent, angles, positioning, changes. Deterministic metrics then show trends,
> coverage and neglected subjects across competitors. Nothing is generated or published yet.
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

**AI analysis (Phase 3).** Turns the history into intelligence:

| Question | Where the answer comes from |
|---|---|
| What are competitors publishing? | Per-page analysis: summary, topics and subtopics, format, audience, intent, funnel stage, angle, themes, keywords, positioning claims, entities (`analyses`, `analysis`) |
| What topics do they focus on? Which are growing? | Topic shares and window-over-window trends per competitor and overall (`trends`, `/intelligence`) |
| What formats do they use? How is the mix shifting? | Format, audience and intent mixes, plus shifts between windows |
| What changed recently? | Change events (Phase 2) with model-written explanations of significant pricing and positioning changes |
| How does each competitor position itself? | A versioned profile whose statements all cite the competitor's own pages (`profile`) |
| Which subjects matter across competitors, and which are neglected? | Topic coverage across competitors; rising, dormant, declining, single-competitor and thin subtopics; plus a model-written briefing grounded in those numbers (`landscape`) |

The pipeline never sends the whole database to the model:

```
history ─► select what needs analysis ─► reuse analyses of minor edits (no LLM)
        ─► digest: page facts + text condensed to a budget (no LLM)
        ─► batch (count and size limits) ─► Gemini structured output ─► validate
        ─► normalize topics (aliases, taxonomy) ─► store ─► deterministic metrics
        ─► summaries, profiles, landscape briefing (Gemini, grounded in stored results)
```

- **Only what changed costs money.** A page is analyzed once per content version (and prompt
  version). Re-runs with no new content make no LLM calls. Pages whose text changed only
  slightly (the Phase 2 minor-edit rule) reuse their previous analysis without a call.
- **Deterministic preprocessing.** Each page becomes a bounded digest: URL, page type, title,
  description, reliable publication date, author, categories, tags, length, heading outline,
  and main text. Long pages are condensed extractively (the opening, then each section's start),
  never summarized by a model.
- **Batches** of up to `ANALYSIS_BATCH_SIZE` pages and `ANALYSIS_BATCH_MAX_CHARS` characters per
  call. A batch with unusable output is retried in halves; a page the model skipped is retried
  once; persistent failures are recorded, and the pages stay pending for the next run.
- **Budgets.** `LLM_MAX_TOKENS_PER_RUN` and `LLM_DAILY_TOKEN_BUDGET` are checked *before* each
  call; every call (tokens, latency, model, prompt version) is recorded in `llm_calls`.
  `analyze --dry-run` shows exactly what would be sent, and its token estimate, without a key.
- **Typed, validated output.** Each prompt has a Pydantic schema, sent to Gemini as a JSON
  Schema. Enums are enforced, and odd values are normalized rather than failing a batch. Results
  are stored as typed columns, not raw model responses.
- **Topic normalization.** A two-level taxonomy (topics → subtopics):
  1. Labels are matched by a normalized key: case, punctuation, plurals and common abbreviations,
     so "AI-Agents", "AI agents" and "Artificial intelligence agents" land on one topic.
  2. The model is shown the existing taxonomy and told to reuse its names.
  3. Every spelling ever seen becomes an alias.
  4. Duplicates the rules can't catch ("Agentic AI" vs "AI agents") are merged by
     `topics consolidate` (Gemini proposes, you apply) or `topics merge`.

  Merged topics keep resolving to their target. You can also seed names and synonyms from
  `config/topics.yaml`.
- **Grounding.** Profile statements must cite evidence ids (the competitor's analyzed pages or
  summarized changes); uncited statements are dropped and counted. Landscape findings must cite
  topics and competitors present in the metrics, or they are dropped. Page text is wrapped in
  delimiters it can't break out of, and treated as data, not instructions. The model has no
  tools.
- **Trends use reliable publication dates only.** A competitor whose captured, dated history
  doesn't reach back to the previous window is reported as `insufficient_history` instead of
  "rising". Scans capture newest content first, so a first scan would otherwise look like a
  sudden surge.
- **Descriptive, not prescriptive.** Phase 3 reports what competitors do. Scoring opportunities
  (what *you* should write) is Phase 4.

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

For AI analysis, set `GEMINI_API_KEY` in `.env`, then scan and analyze:

```bash
uv run python -m app scan --all
uv run python -m app analyze --all --dry-run                # review what would be sent first
uv run python -m app analyze --all
uv run python -m app landscape --refresh
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
uv run python -m app runs acme                            # recent runs (scans, analyses, reports)
uv run python -m app run 42                               # one run with its audit events

# AI analysis (needs GEMINI_API_KEY, except --dry-run)
uv run python -m app analyze acme --dry-run          # what would be sent, token estimate; no call
uv run python -m app analyze acme                    # analyze new/changed pages, summarize, profile
uv run python -m app analyze --all --limit 20        # every active competitor, 20 pages each
uv run python -m app analysis 123                    # one page's analysis
uv run python -m app trends acme --days 30           # topics, formats, audiences, shifts
uv run python -m app trends                          # across competitors: rising, neglected
uv run python -m app profile acme                    # evidence-backed positioning profile
uv run python -m app landscape --refresh             # new cross-competitor AI briefing
uv run python -m app usage --days 7                  # Gemini calls and tokens per day

# Topic taxonomy
uv run python -m app topics                          # topics with page and competitor counts
uv run python -m app topics show ai-agents           # trend, subtopics, recent pages
uv run python -m app topics import                   # seed from TOPICS_FILE (optional)
uv run python -m app topics consolidate              # Gemini proposes duplicate merges
uv run python -m app topics consolidate --apply      # ...and applies them
uv run python -m app topics merge agentic-ai ai-agents

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
| GET | `/api/v1/runs`, `/api/v1/runs/{id}` | Runs (scans, analyses, reports), with their audit events |
| POST | `/api/v1/competitors/{slug}/analyses` | Start an AI analysis run: `202` plus a run to poll, or `?wait=true`. Body: `{"limit": 20, "reanalyze": false, "change_summaries": true, "profile": true, "force_profile": false}`. `503` without `GEMINI_API_KEY` |
| GET | `/api/v1/competitors/{slug}/analysis-plan` | What a run would send, with token estimates (no LLM call) |
| GET | `/api/v1/competitors/{slug}/intelligence` | Topics, subtopics, formats, audiences, intents, trends, mix shifts, recent items and changes, latest profile (`?days=30`) |
| GET | `/api/v1/competitors/{slug}/profile`, `/profiles` | Latest profile / version history |
| GET | `/api/v1/analyses` | Latest analysis per page: `competitor`, `topic`, `content_format`, `published_since` |
| GET | `/api/v1/content/{id}/analyses` | Every analysis of one page |
| GET | `/api/v1/topics`, `/api/v1/topics/{slug}` | Taxonomy (`?parent=`, `?q=`) / one topic across competitors |
| POST | `/api/v1/topics/merge`, `/api/v1/topics/consolidate` | Merge `{"source", "target"}` / Gemini-proposed merges (`?apply=true` to apply) |
| GET / POST | `/api/v1/intelligence/landscape` | Cross-competitor metrics plus the latest briefing / generate a new briefing (`202` or `?wait=true`) |
| GET | `/api/v1/llm/usage` | Gemini calls and tokens per day, purpose and model (`?days=7`) |

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
| Analysis (Phase 3) | `topics`, `topic_aliases`, `content_analyses`, `content_analysis_topics`, `change_summaries`, `competitor_profiles`, `landscape_reports` | Model-produced interpretations, each with its run, model and prompt version; profiles and reports stored with the metrics they were grounded on |
| Operations | `runs`, `run_events`, `llm_calls` | What ran, when, with what result; every LLM call with its tokens |

Only the analysis layer holds LLM output, and it never modifies the layers below it.
Recommendations (Phase 4+) get their own tables.

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
| `GEMINI_API_KEY` | For AI analysis | *(empty)* | Gemini API key. Create one in [Google AI Studio](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` | Model ID. Leave empty for the default. |
| `GEMINI_ANALYSIS_MODEL` / `GEMINI_SYNTHESIS_MODEL` | No | `GEMINI_MODEL` | Per-route models: bulk per-page analysis vs. profiles, briefings, summaries, consolidation |
| `ANALYSIS_REASONING_EFFORT` / `SYNTHESIS_REASONING_EFFORT` | No | `low` / `medium` | Gemini `thinking_level` per route |
| `LLM_MAX_TOKENS_PER_RUN` / `LLM_DAILY_TOKEN_BUDGET` | No | `400000` / `2000000` | Hard budgets checked before each call (`0` daily = unlimited) |
| `LLM_TIMEOUT_SECONDS` | No | `120` | Per-request timeout |
| `LLM_MAX_RETRIES` | No | `2` | SDK retries for 429/5xx/timeouts |

- Structured outputs: every prompt's Pydantic schema is sent as a self-contained JSON Schema
  (nested models inlined, every field marked required so the model fills or explicitly nulls
  each one), and the reply is validated before use.
- Prompts live in [`app/prompts/`](app/prompts/), each with a `VERSION`, and that version is
  stored on everything it produces. Changing a prompt means bumping its version: pages then
  become pending again and are re-analyzed gradually, within the per-run limit.

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
| 1 · Website monitoring | **No.** Deterministic. Works with `GEMINI_API_KEY` empty. |
| 2 · Persisted history | **No.** Storage, incremental scans and change detection are deterministic. |
| 3 · Competitor analysis (current) | **Yes:** per-page analysis, change summaries, profiles, landscape briefings, topic consolidation. Metrics, trends and gaps stay deterministic. |
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
| `TOPICS_FILE` | `config/topics.yaml` | Optional seed taxonomy for `topics import` |
| `ANALYSIS_MAX_ITEMS_PER_RUN` | `40` | Pages analyzed per run; the rest stay pending |
| `ANALYSIS_BATCH_SIZE` / `_BATCH_MAX_CHARS` / `_ITEM_MAX_CHARS` | `6` / `40000` / `6000` | Pages per Gemini call and character budgets per call and per page |
| `ANALYSIS_MIN_WORDS` / `_MIN_WORDS_POSITIONING` | `80` / `20` | Minimum words for editorial pages / homepage, pricing, product and landing pages |
| `ANALYSIS_EXCLUDE_TYPES` | `["careers","legal","listing"]` | Page types never analyzed (JSON list) |
| `ANALYSIS_MAX_CHANGE_SUMMARIES_PER_RUN` | `10` | Significant changes explained per run |
| `ANALYSIS_TAXONOMY_PROMPT_LIMIT` | `150` | Existing topics shown to the analyzer |

## Development

```bash
docker compose up -d db              # the tests need PostgreSQL
uv run pytest                        # offline: no external network, no real Gemini calls
uv run ruff check . && uv run ruff format --check .
uv run mypy app                      # strict type checking
LIVE_SCAN_URL=https://www.example.com uv run pytest -m live   # opt-in real-site scan
uv run pytest -m llm_live            # opt-in: one real Gemini call (needs GEMINI_API_KEY)
```

- Each test session creates a fresh database from the Alembic migrations (proving a fresh
  checkout builds the schema), and tables are emptied between tests. Point `TEST_DATABASE_URL`
  at another server if needed. Database tests are **skipped with a warning** when PostgreSQL
  isn't reachable, except in CI (`REQUIRE_DB_TESTS=1`), where that fails the run.
- HTTP, including the Gemini API, is mocked with [respx]; an autouse guard blocks every
  non-loopback socket.
- The analysis pipeline is tested end to end with `tests/fakellm.py`, a deterministic stand-in
  for Gemini. It reads prompts like the real model, and can fail, omit documents or return
  invalid output on demand.
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
    change_detection.py        word-level diffs, price changes, diff excerpts
    analysis.py                AI analysis runs: select → digest → batch → Gemini → store
    digest.py                  deterministic page digests, condensing, batching
    topics.py, labels.py       taxonomy: label normalization, aliases, seeds, merges
    trends.py                  deterministic trends, mixes, coverage and gaps
    intelligence.py            competitor and landscape intelligence (read side)
    profiles.py, landscape.py  grounded syntheses (Gemini)
    change_summaries.py        explanations of significant changes (Gemini)
    llm_usage.py               token budgets and the LLM call ledger
  prompts/                     versioned prompts + their structured-output schemas
  db/                          models (by layer), sessions, advisory locks, queries, migrations
  llm/                         provider-agnostic LLM interface + Gemini provider
  api/                         HTTP routes and schemas
migrations/                    Alembic migrations
config/                        competitors.example.yaml, topics.example.yaml
tests/                         unit, integration (incl. PostgreSQL) and opt-in live tests
```

## Licensing

Small patterns are adapted from MIT-licensed projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
