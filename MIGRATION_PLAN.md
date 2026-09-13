# Migration Plan — Competitor Intelligence & Content Agent

**Status:** Approved · **Date:** 2026-09-13 · **Main repo:** `atulhooda/competitor-analysis-agent`

**Revision history**
- r1 (2026-09-13): initial analysis and plan.
- r2 (2026-09-13): **the primary LLM provider changed from Anthropic to Google Gemini**, at the owner's request. Gemini is the only LLM provider; no other provider is introduced unless the owner explicitly asks. Phase 1 stays deterministic (no LLM calls) and only establishes the provider-agnostic LLM interface and Gemini configuration.

The three source repositories were cloned to a temporary scratch directory for analysis only. They are not vendored, submoduled, or left beside the project.

---

## 0. Summary

| Decision | Choice |
|---|---|
| **Base repository** | **Repository 3 — `Str1nX03/Competitor-Research` (MIT)**. It is used as an *architectural seed*: package layout, settings pattern, LangGraph node conventions and prompt separation. It is not a feature base. Expect well over 90% of the final code to be new. |
| **Import (with attribution)** | Repository 1 — `gokborayilmaz/competitor-analysis-agent` (MIT). Competitor-profile schemas, and the "research each competitor in isolation → synthesize only from structured evidence" pattern. |
| **Reference only (no code)** | Repository 2 — `ShreyashSoni/agentic_blog_generator`. It has **no license**, so none of its code or prompt text can be copied. Its blog-pipeline *ideas* will be re-implemented independently. |
| **Stack** | Python 3.12 · FastAPI · PostgreSQL 16 · SQLAlchemy 2 + Alembic · compliant crawler (httpx + trafilatura) · **Google Gemini** via the official `google-genai` SDK (Interactions API), behind a provider-agnostic `app/llm` interface · LangGraph (content-generation workflow only) · APScheduler worker process |
| **Deliberately not used** | Redis, Celery, Kafka, microservices, ChromaDB, a vector DB in the MVP, any LLM provider other than Gemini (no OpenAI, no Anthropic), LangChain model wrappers, Tavily, Firecrawl, Upsonic, Flask |
| **Build order** | 10 vertical phases. Phase 1 is compliant competitor-website monitoring, with no database and no LLM. |

---

## 1. Repository analysis

### 1.1 Side-by-side

| Aspect | R1 · competitor-analysis-agent | R2 · agentic_blog_generator | R3 · Competitor-Research |
|---|---|---|---|
| Purpose | One-shot competitor snapshot report | Topic → blog post; WordPress publishing; social-post drafts | Idea/company → competitor discovery report |
| Size / history | ~360 LOC Python, 4 commits | ~6.2k LOC Python + ~1.3k LOC tests, 66 commits | ~550 LOC Python + ~1.3k LOC HTML/CSS/JS, 49 commits |
| Architecture | Single script, two phases | Flat packages (`agents/`, `services/`, `workflows/`, `memory/`, `prompts/`) plus 4 CLI entry scripts | `src/` package (`agents/`, `prompts/`, `config.py`, `utils.py`) + Flask app |
| Frameworks | Upsonic, Firecrawl SDK, Anthropic | LangGraph, LangChain (Anthropic + OpenAI), Tavily, ChromaDB, tenacity, requests | LangGraph, LangChain-Groq, langchain-tavily, Flask, flask-sock, xhtml2pdf, pydantic-settings |
| Agent architecture | A tool-using research agent per competitor (isolated context), then a tool-less synthesis agent. Pydantic structured output. | Linear 6-node graph: planner → research → outline → writer → editor → SEO | Two tiny graphs (researcher: 2 nodes; reporter: 1 node) chained inside a web handler |
| Backend / API | None (CLI) | None (CLIs only) | Flask routes + WebSocket, report list/get/delete, PDF endpoint. The README says FastAPI; that is stale. |
| Database | None (writes `report.json` / `.md`) | None (Markdown files on disk) | SQLite, one `reports` table, raw `sqlite3` |
| Vector DB | None | ChromaDB. The per-topic collection is deleted and recreated every run; default embeddings. | None |
| LLM integration | Upsonic `Agent(model="claude-sonnet-4-6")` | `get_llm()` factory (OpenAI or Anthropic via LangChain); sends an internal SSO bearer token | `ChatGroq` (`openai/gpt-oss-20b`), 700 max tokens |
| Web research | Firecrawl search (3 results) | Tavily "advanced" search (7 results), each LLM-summarized | Tavily, 1–2 results per query, 5 fixed questions per competitor |
| Scraping / crawling | Firecrawl scrape of the **homepage only** (map/crawl disabled) | None | None |
| Scheduling | None | None | None |
| Publishing | None | WordPress REST via Application Passwords (separate CLI over `.md` files) | PDF download |
| Error handling | None | `try/except` everywhere with **silent fabricated fallback content** | `CustomException(e, sys)` around every call; no retries or timeouts |
| Configuration | Python constants | `.env` + CLI flags; hard-requires a corporate SSO `TOKEN` | pydantic-settings + `.env` (good) |
| Testing | None | 60 tests, all for an approval module the workflow never calls. **57 pass / 3 fail.** | None (0-byte stubs, later deleted) |
| Observability | `print()` | stdlib `logging` | `print()`; LangSmith settings declared but never used |
| Code quality | Readable but **does not run** (broken import) | Docstrings and types; dead code, duplicated helpers, `sys.exit` inside library code | Small; contradictory prompts; README/code drift |
| Extensibility | Low (Upsonic lock-in, single file) | Medium (swappable node functions) | Medium (clear package seams) |
| License | **MIT** © 2024 Upsonic Teknoloji A.Ş. (file `LICENCE`) | **None** | **MIT** © 2026 Dravin Kumar Sharma |

### 1.2 Verified defects (each checked in code, git history, or by running)

**R1**
- Won't run: `main.py` does `from schemas import …`, but the file is named `shemas.py`.
- The README describes `map_website` discovery, but the code sets `enable_map=False`.
- Shallow and point-in-time: one homepage scrape and two searches per competitor. No history, no diffing.

**R2**
- **No license** (see §3).
- **Coupled to corporate-internal auth.** `utils/env_utils.py` shells out to `/usr/local/bin/appleconnect` with a hardcoded OAuth client ID. `validate_environment()` exits if `TOKEN` is missing, and the LLM factory sends `Authorization: Bearer $TOKEN`. `.env.example` references an internal gateway ("floodgate"), and the project depends on `apple-certifi`.
- **Silent fabricated fallbacks.** When Tavily or the LLM fails, "research" becomes lines like *"{topic} is an important subject in modern technology and business"*, the writer emits template filler sections, and SEO falls back to generic metadata. In an auto-publish system this would publish filler.
- **The WordPress publisher can create duplicate posts.** `create_post()` is wrapped in a tenacity `@retry` on any `RequestException`. A read-timeout that happens *after* WordPress created the post therefore re-POSTs it.
- The Markdown body is sent to WordPress unconverted. The `markdown` package is declared but never imported. There is also no SEO-plugin meta, no `future` (scheduled) status, and no slug pre-check. Auth errors get re-wrapped as generic publish errors.
- **The human-approval service is dead code.** It isn't wired into the graph. It's based on CLI `input()`, **auto-approves on timeout**, and the timeout can't fire while `input()` blocks (the failing test `test_timeout_calculation` confirms this). It also installs process-wide SIGINT/SIGTERM handlers that call `sys.exit`.
- LangGraph misuse:
  - Nodes mutate and return the whole state.
  - A ChromaDB client object lives in graph state, which isn't serializable, so checkpointing is impossible.
  - The "parallel writers" are a sequential loop.
- Thin RAG: research is truncated to 1,000 chars before summarizing and 500 chars per retrieved doc, and the collection is wiped every run.
- Prompts are loaded by CWD-relative path.
- Dependency drift between `pyproject.toml` and `requirements.txt` (e.g. `langchain-anthropic` 1.3.4 vs 1.4.0). `langgraph` and `PyYAML` are only installed transitively. `pytest` and `ipykernel` are listed as runtime deps. Requires Python ≥ 3.13.

**R3**
- The README describes FastAPI, `uvicorn main:app`, an "Assistant Agent" and a `tests/` directory. Git history shows the backend was switched to Flask on 2026-07-16 and the assistant agent deleted; the tests were always empty files.
- The competitor-extraction prompt asks for JSON, then for a comma-separated list. The output is parsed with `split(',')` and not stripped. "Find as many competitors as possible" leads to unbounded N × 5 sequential searches.
- It asks for revenue, profit and market share from single search snippets, which invites hallucinated figures.
- The pipeline blocks inside the WebSocket handler. PDF temp files are never deleted. There is no auth on DELETE. LLM output, derived from web content, is rendered via `marked.parse` → `innerHTML` without sanitization (XSS).
- The LangSmith settings are declared, but nothing reads them or loads `.env` into the process environment.

---

## 2. Scores (1 = absent/unusable, 10 = production-grade)

| # | Criterion | R1 | R2 | R3 |
|---|---|:-:|:-:|:-:|
| 1 | Competitor research | 5 | 1 | 4 |
| 2 | Web crawling | 3 | 1 | 1 |
| 3 | Web research | 5 | 5 | 4 |
| 4 | Social/content monitoring | 1 | 2¹ | 1 |
| 5 | Agent architecture | 5 | 6 | 3 |
| 6 | LangGraph implementation | 1² | 5 | 4 |
| 7 | RAG | 1 | 4 | 1 |
| 8 | Blog generation | 1 | 6 | 1 |
| 9 | SEO | 1 | 5 | 1 |
| 10 | CMS publishing | 1 | 4 | 1 |
| 11 | FastAPI/backend | 1 | 1 | 3³ |
| 12 | Database architecture | 1 | 2 | 2 |
| 13 | Scheduling | 1 | 1 | 1 |
| 14 | Production readiness | 1 | 2 | 2 |
| 15 | Extensibility | 3 | 5 | 4 |
| 16 | Code quality | 4 | 4 | 3 |
| 17 | Ease of modification | 6 | 5 | 7 |
| | **Total (/170)** | **41** | **59** | **43** |

¹ R2 generates social posts *from* blogs; it doesn't monitor anything. ² R1 uses Upsonic, not LangGraph. ³ Flask, not FastAPI.

None of the three is production-grade. R2 scores highest, but it can't legally serve as a base (§3, §4).

---

## 3. License assessment

*This is an engineering assessment, not legal advice. Confirm with counsel if the product's IP position is material, e.g. for fundraising diligence.*

| Repo | License found | Commercial/proprietary use? | Conditions | Verdict |
|---|---|---|---|---|
| R1 | MIT (`LICENCE`), © 2024 Upsonic Teknoloji A.Ş. | Yes | Keep the copyright and permission notice with copied or substantial portions | ✅ Code may be copied/adapted with attribution |
| R3 | MIT (`LICENSE`), © 2026 Dravin Kumar Sharma | Yes | Same | ✅ Code may be copied/adapted with attribution |
| R2 | **None.** No license file anywhere in the 66-commit history. The README shows an "MIT" badge that links to a `LICENSE` file that doesn't exist, and GitHub reports no license. | **No** | Default copyright applies (all rights reserved). GitHub's terms only allow viewing and forking on GitHub. A badge is not a grant. | ❌ **Do not copy code or prompt text** |

There is a second concern with R2. Its code references corporate-internal tooling (see §1.2), which suggests it may have been written in an employer context. So even a license added later by the repo author might not settle who owns the copyright.

**Consequences**
- R1 and R3 material that we adapt gets a header comment (`Adapted from <repo> (MIT, © <holder>)`). Both notices go into `THIRD_PARTY_NOTICES.md`.
- For R2, we take only non-copyrightable ideas: the stage decomposition, and the kinds of SEO checks. The WordPress integration will be written from the public WordPress REST API documentation. No R2 files, functions or prompt text will be pasted, paraphrased line-by-line, or used as a template. Its hardcoded client ID and internal URLs will not be propagated anywhere.
- New dependencies were checked for licenses (§9). All are MIT, BSD, Apache-2.0 or PSF, except `psycopg` (LGPL-3.0). Using LGPL unmodified as a dependency is compatible with proprietary and SaaS use.

---

## 4. Base repository decision

### Use Repository 3 as the base.

**Why R3**
1. **Legally usable** (MIT).
2. **Closest shape to the target.**
   - A real package with separated `agents/`, `prompts/`, config and utilities.
   - A web backend with persistence and a report-history API.
   - Its LangGraph nodes return **partial state updates** (idiomatic, and checkpoint-friendly). R2's don't.
   - Typed per-agent state, and pydantic-settings configuration.
3. **Its stack is our stack** — LangGraph, pydantic-settings, LangSmith-compatible tracing — without the parts we're dropping (Groq, Flask).
4. **Same domain as Phases 1–4** (competitor research).

**What "base" means in practice.** R3 contributes conventions and a handful of adapted modules: the settings pattern, prompt/agent separation, and LangGraph node style. Its Flask app, SQLite storage, Groq client, discovery agent and frontend are all replaced. We copy the few patterns we keep, with attribution. We do **not** merge R3's git history into your repo.

**Why not R2 (despite the highest score).** It has no license and provenance concerns. It is also coupled to corporate-internal auth, and its publisher and fallback behaviour are unsafe. Even with a license, most of it would be rewritten.

**Why not R1.** It's a ~360-line script that doesn't run, is built on Upsonic (which would become a second agent framework next to LangGraph), and has no server, persistence or package structure. Its best asset, the schemas plus the grounded-synthesis pattern, is imported instead.

---

## 5. Component disposition

### 5.1 Repository 3 (base, MIT)

| R3 file / module | Disposition | Target |
|---|---|---|
| `src/config.py` — `Settings(BaseSettings)` + cached `get_settings()` | **Reuse pattern, adapted.** New fields, `SecretStr` for secrets, grouped settings, `.env` loading. | `app/config.py` (Phase 1) |
| `src/prompts/*.py` — prompts kept out of agent code | **Reuse pattern.** Prompts become versioned files. The prompt text itself is rewritten. | `app/prompts/` (Phase 3+) |
| `src/agents/*.py` — class per agent, typed `TypedDict` state, nodes return partial dicts, `START`/`END` | **Reuse conventions** for the LangGraph workflow; the code is rewritten | `app/workflows/` (Phase 5) |
| `app.py` report history (list/get/delete) | **Concept kept**, becoming run history plus an audit trail | `app/api/v1/runs.py` (Phase 2) |
| `app.py` Markdown → HTML (`markdown` with `tables`, `fenced_code`) | **Concept kept.** Reimplemented with `markdown-it-py` + `nh3` sanitization for CMS output. | `app/publishers/render.py` (Phase 7) |
| `app.py` Flask routes, flask-sock WebSocket, raw `sqlite3` | **Rewritten** as FastAPI routers and SQLAlchemy/PostgreSQL; progress is exposed through run status | `app/main.py`, `app/api/` |
| `src/utils.py` `get_llm()` (Groq) | **Factory shape kept; implementation replaced.** `get_llm()` returns an `LLMProvider` interface whose only implementation is `GeminiProvider`. Nothing outside `app/llm/gemini.py` imports the Gemini SDK. | `app/llm/` (interface and config in Phase 1; first calls in Phase 3) |
| `src/utils.py` `web_search()` (Tavily) | **Removed.** Gemini's Google Search grounding handles blog research; our own fetcher handles monitoring. | — |
| `src/exception.py` `CustomException(e, sys)` | **Removed** (it hides error types) and replaced by a typed error taxonomy | `app/core/errors.py` |
| `src/agents/researcher_agent.py` (discovers competitors from an idea) | **Removed from the MVP.** "Suggest competitors" stays in the backlog. | — |
| `src/agents/reporter_agent.py` (free-form Markdown report) | **Replaced** by structured change summaries and digests | `app/services/analysis.py` (Phase 3) |
| `templates/`, `static/` (landing page, dashboard, `marked.js`) | **Removed** (XSS, and not needed until Phase 10) | — |
| `Procfile`, `reports.db`, `research.ipynb`, `.env_example` | **Removed** / replaced by `.env.example` | — |

### 5.2 Repository 1 (import, MIT)

| R1 file / module | Disposition | Target |
|---|---|---|
| `shemas.py`: `CompetitorProfile`, `CompetitiveAnalysisReport` | **Import, adapted.** Add evidence URLs, observed-at timestamps, confidence, and structured pricing tiers. | `app/domain/competitor_profile.py` (Phase 3) |
| `main.py` pattern: each competitor researched in its own context, then synthesis from structured profiles only ("do not add information not present") | **Reuse pattern** in the analysis services | `app/services/analysis.py` (Phase 3) |
| `config.py` `INDUSTRY`, `FOCUS_AREAS` | **Concept** folded into the company profile | `config/company_profile.yaml` (Phase 4) |
| Upsonic `Agent`/`Task`, `FirecrawlTools` | **Removed.** No second agent framework; our compliant crawler replaces Firecrawl. | — |
| Terminal/Markdown report printing, `report.json`, `example_report.md` | **Removed.** Output goes through the API (and later the dashboard). | — |

### 5.3 Repository 2 (reference only, no code or text copied)

| Idea observed | Independent implementation in the new system |
|---|---|
| Stage decomposition: plan → research → outline → write → edit → SEO | LangGraph workflow that adds fact-checking, an originality check, a quality gate and a bounded revision loop (Phase 5–6) |
| Per-section retrieval of research context | Drafting over stored research documents. Each claim is tagged with a source ID (and Gemini grounding metadata for web sources), so claims are traceable. |
| Prompts kept outside code | `app/prompts/` with version IDs recorded on every output |
| SEO checks (meta title/description length, slug rules, keyword density, FAQ) | `app/services/seo.py` (deterministic checks) plus one LLM call for copy |
| WordPress REST + Application Passwords + get-or-create taxonomy | `app/publishers/wordpress.py`, written from the WP REST docs: async, idempotent, SEO meta, scheduling |
| Plan/outline approval in the CLI | Replaced by an `ApprovalPolicy` on the **final article** (DB state machine) |
| Social-post drafts from a blog | Backlog (a distribution feature, not monitoring) |

### 5.4 Functionality removed
Flask UI and landing page; PDF export; WebSocket streaming; idea-driven competitor discovery (deferred); Upsonic/Firecrawl agentic scraping; ChromaDB; OpenAI and Groq providers; CLI approval prompts; the file-based Markdown → publish pipeline; social-post generation (deferred); the standalone editing CLI.

### 5.5 Functionality added (none exists in any source repo)
- Robots-compliant crawler with sitemap and RSS discovery
- Versioned content history and change detection
- Four strictly separated data layers
- Topic taxonomy and deterministic trend metrics
- Configurable opportunity scoring
- Fact-check, originality and quality gates
- CMS abstraction and an idempotent WordPress publisher with SEO meta and scheduling
- Application-level daily publishing limit
- Approval-policy extension point
- Scheduler worker with overlap locks
- Audit trail and structured logs
- API authentication
- Social platform adapters
- Dashboard

---

## 6. Target architecture

### 6.1 Principles
1. **Deterministic core, LLM at the edges.** Crawling, diffing, trend math, scoring, limits and publishing are plain Python and SQL. The LLM is used only where judgment or language generation adds value.
2. **The four data layers are never mixed:** raw → normalized → analysis → recommendation (§7). Every analysis or recommendation row records *how* it was produced: run, model, and prompt version.
3. **No silent fabrication.** A failed step fails loudly and is recorded. Nothing ever substitutes placeholder content.
4. **Side effects are isolated.** Only `PublishingService` talks to a CMS, and no LLM sits on that path.
5. **Everything is auditable:** which agent or tool ran, on what sources, why a topic was chosen, and what the gate decided.

### 6.2 Layers
```
            ┌──────────────────────── API (FastAPI) ─── CLI (Typer) ─── Scheduler worker (APScheduler) ──┐
            │                         thin: auth, validation, DTOs → call services                        │
            ├──────────────────────── Application services (orchestration) ────────────────────────────────┤
            │ monitoring · change_detection · analysis · trends · opportunities · generation · approval ·   │
            │ publishing (daily limit + idempotency) · runs/audit                                          │
            ├──────────── Agents / Workflows (LLM) ─────────────┬───────── Integrations (no LLM) ───────────┤
            │ content analyzer · change summarizer · relevance   │ crawling: fetcher, robots, sitemaps,     │
            │ judge · LangGraph blog workflow (plan, research,   │ feeds, extract, classify                  │
            │ outline, draft, edit, fact-check, SEO, judge)      │ publishers: WordPress (CMSPublisher)      │
            │                                                    │ social: YouTube, Instagram, X adapters    │
            ├────────────────────────────── Data layer (SQLAlchemy 2 / PostgreSQL 16) ───────────────────────┤
            │ repositories per layer: config · raw · normalized · analysis · recommendation · ops           │
            └──── External: competitor sites · Gemini API (+ Google Search grounding) · WordPress · social APIs ┘
```

### 6.3 Directory structure
```
competitor-analysis-agent/
├── pyproject.toml · uv.lock · .python-version · .env.example · .gitignore
├── README.md · MIGRATION_PLAN.md · THIRD_PARTY_NOTICES.md
├── docker-compose.yml              # Phase 2: postgres (pgvector/pgvector:pg16 image)
├── alembic.ini · migrations/       # Phase 2
├── config/                         # user data (real files gitignored; *.example.yaml committed)
│   ├── competitors.example.yaml
│   ├── company_profile.example.yaml    # Phase 4
│   └── scoring.example.yaml            # Phase 4
├── deploy/wordpress/               # Phase 7: optional mu-plugin exposing SEO/meta fields to REST
├── app/
│   ├── main.py                     # FastAPI app factory + lifespan
│   ├── __main__.py · cli.py        # `python -m app …` (Typer)
│   ├── config.py                   # Settings (pydantic-settings) + YAML loaders/validation
│   ├── core/                       # logging (structlog), errors, http client factory, retry policies, time/tz
│   ├── domain/                     # shared Pydantic types & enums (ContentType, ChangeType, CompetitorProfile…)
│   ├── api/                        # deps (auth, db session), v1 routers, request/response schemas
│   ├── db/                         # engine/session, repositories; models/ grouped by data layer
│   ├── crawling/                   # fetcher, robots, ratelimit, sitemaps, feeds, extract, classify
│   ├── services/                   # application services (see 6.2)
│   ├── llm/                        # LLMProvider interface, get_llm() factory, Gemini provider (only google-genai importer); later: prompt registry, usage/cost
│   ├── agents/                     # the few LLM-reasoning components (+ their tool definitions)
│   ├── workflows/                  # LangGraph blog_generation graph + state
│   ├── prompts/                    # versioned prompt files
│   ├── publishers/                 # base.py (CMSPublisher), wordpress.py, render.py (md→html, sanitize)
│   ├── social/                     # Phase 9: SocialSource protocol + platform adapters
│   └── scheduler/                  # worker, job registry, advisory locks
└── tests/  unit/ · integration/ · fixtures/ (html, sitemaps, feeds, wp responses) · conftest.py
```
Differences from your suggested layout:
- There is no separate `tools/` package. The only tool-using agent is the blog researcher, so its tools live beside it.
- `models/` and `database/` are merged into `db/`.
- `crawling/` and `social/` are integrations, kept separate from services.

### 6.4 Agents vs services (your proposed list, resolved)

| Proposed | Resolution | Why |
|---|---|---|
| Competitor Monitoring Agent | **Service** (`crawling/` + `services/monitoring.py`) | Discovery, fetching and diffing are deterministic. An LLM adds cost and nondeterminism. |
| Research Agent | **Merged** into the Blog Research step | Only generation needs open-ended research |
| Content Analysis Agent | **LLM function** (single structured call per item, batched) | Classification and extraction, not autonomy |
| Trend Detection Agent | **Service** (SQL/Python) + an optional LLM narrative | Growth and saturation are arithmetic |
| Opportunity Agent | **Service** (weighted scoring) + one LLM relevance/angle judgment | Scores must be explainable and configurable |
| Blog Research Agent | **Agent** (Gemini with Google Search grounding / URL context + internal-corpus tool) | Deciding what to look up benefits from reasoning |
| Blog Writer / Editor | **LangGraph nodes** | Generation |
| Fact Checker | **LangGraph node** (claims ↔ cited sources) | Verification needs reading comprehension |
| SEO Agent | **Mostly service** (length, slug, headings, links, density) + one LLM copy call | Most SEO rules are mechanical |
| Quality evaluation | **Service checks + LLM judge** node | A hybrid gate |
| Publishing Agent | **Service. No LLM.** | Side effects must be deterministic and idempotent |

---

## 7. Data architecture (PostgreSQL 16)

### 7.1 Tables by layer

**Config**
- `competitors`: `id`, `slug` (unique), `name`, `website_url`, `is_self` (your own site, for coverage and internal links), `active`
- `sources`: `id`, `competitor_id`, `kind` (`website` · `rss` · `sitemap` · `page_watch` · `youtube` · `instagram` · `x` · `import`), `url_or_handle`, `settings` JSONB (include/exclude patterns, limits), `status` (`active` · `blocked` · `paused`), `etag`, `last_modified`, `last_checked_at`

**1. Raw (as collected, append-only)**
- `fetches`: every HTTP attempt — `url`, `status`, headers, `elapsed_ms`, `bytes`, `error`, `robots_decision`, `run_id`
- `raw_documents`: `url`, `content_hash`, `body` (TOAST-compressed), `content_type`, `fetched_at`. Stored only when the hash is new. A retention policy is configurable. Access goes through a `RawStore` interface, so it can move to object storage later.

**2. Normalized (facts extracted from sources)**
- `content_items`: `competitor_id`, `canonical_url` (unique per competitor), `content_type` (`blog_post` · `landing_page` · `pricing` · `case_study` · `resource` · `product` · `changelog` · `press` · `docs` · `other`), `first_seen_at`, `last_seen_at`, `removed_at`, `published_at`, `current_version_id`
- `content_versions`: `content_item_id`, `version_no`, `raw_document_id`, `title`, `description`, `author`, `published_at`, `modified_at`, `language`, `text_md`, `headings` JSONB, `categories[]`, `tags[]`, `page_meta` JSONB (OpenGraph/JSON-LD), `word_count`, `text_hash`, `extractor_version`
- `change_events`: `content_item_id`, `from_version_id`, `to_version_id`, `change_type` (`new` · `updated` · `removed` · `pricing_changed`), `diff_stats` JSONB, `detected_at`. Detected deterministically.
- `social_posts` / `social_post_metrics` (Phase 9): `platform`, `external_id` (unique per platform), `posted_at`, `post_type` (image, video, reel, carousel, text, short), `caption`, `hashtags[]`, `url`; metrics as a time series

**3. Analysis (interpretations and derived metrics; each row has `run_id`, `method` = `llm`|`deterministic`, `model`, `prompt_version`)**
- `content_analyses`: per version — summary, primary topic, format (how-to, listicle, case study, comparison, opinion, news…), funnel stage, target audience, keywords[], positioning claims[], entities
- `topics` (canonical taxonomy; `merged_into_id` for human merges) and `content_topics` (version ↔ topic, confidence, is_primary)
- `competitor_profiles`: versioned positioning, pricing and audience snapshots (adapted R1 schema), with evidence URLs
- `change_summaries`: an LLM explanation and significance score for a `change_event` (e.g. pricing or messaging shifts)
- `trend_snapshots`: topic × competitor × period — item count, share of voice, growth, competitor coverage (deterministic)

**4. Recommendation (outputs and actions)**
- `content_opportunities`: `topic_id`, `factors` JSONB, `weights_version`, `score`, `rationale`, `suggested_angle`, `evidence` (version IDs), `status` (`new` · `selected` · `dismissed` · `generated`)
- `generated_articles`: `opportunity_id`, `status` (`drafting` · `needs_review` · `awaiting_approval` · `approved` · `rejected` · `scheduled` · `published` · `failed`), `title`, `slug` (unique), `meta_title`, `meta_description`, `keywords[]`, `category`, `tags[]`, `body_md`, `faq`, `internal_links`, `external_references`, `cta`, `featured_image_suggestion`, `quality_report`, `fact_check_report`, `originality`, `workflow_thread_id`, `approved_by`/`approved_at`
- `publications`: `article_id`, `target`, `idempotency_key` (unique), `mode` (`draft` · `publish` · `schedule`), `status` (`in_progress` · `succeeded` · `failed` · `unknown`), `go_live_date` (local date used for the daily limit), `scheduled_for`, `remote_id`, `remote_url`, `attempts`, `last_error`

**Ops**
- `runs`: `kind`, `trigger` (`schedule` · `api` · `cli`), `status` (`running` · `succeeded` · `partial` · `failed`), params, stats, timings
- `run_events`: per step or tool call — component, event, data (URLs, factor scores, gate results, token usage and cost), level. This is the audit trail.

### 7.2 Storage decisions
- **PostgreSQL only.** One database for relational data, raw documents, workflow checkpoints (LangGraph Postgres saver) and job locks (advisory locks). No Redis.
- **pgvector is deferred.** The MVP doesn't need vector search:
  - topics are canonicalized by the LLM against the existing taxonomy (cached prompt);
  - originality uses n-gram shingling, which is better than embeddings for detecting copying;
  - "RAG" for generation means SQL selection of topic-tagged competitor content, Postgres full-text search, and research documents placed directly in context.

  We still run the `pgvector/pgvector:pg16` image, so adding it later is one migration plus an `Embedder` module, not an infrastructure change. Revisit at Phase 3/4 with real data. If embeddings are needed, Gemini's embedding models (e.g. `gemini-embedding-001`) keep this on the single provider.

---

## 8. Cross-cutting design

### 8.1 Compliant crawling rules (Phase 1)
- **Honest identity:** a configurable bot User-Agent with a contact URL. No browser spoofing, no proxy rotation, no stealth/headless evasion.
- **robots.txt:** fetched and cached per host (24 h TTL). Disallow and `Crawl-delay` are honored. If robots.txt returns 4xx, crawling is allowed. If it returns 5xx or is unreachable, crawling is paused for that host (conservative).
- **Politeness:**
  - one in-flight request per host;
  - a minimum delay per host (default 3 s, or the robots `Crawl-delay` if larger), plus a global concurrency cap;
  - conditional GET (`If-None-Match` / `If-Modified-Since`).
- **Retries:** only on 408/429/5xx/timeouts, honoring `Retry-After` (capped), with exponential backoff and jitter, at most 3 attempts. Other 4xx responses are never retried.
- **Stop conditions:** a 401/403 or a detected CAPTCHA/challenge marks the source `blocked` and stops it. Nothing is ever bypassed.
- **Safety limits:**
  - maximum response size, and a content-type allowlist;
  - limited, same-site redirects;
  - SSRF guard (http/https only; private, loopback and link-local IPs blocked);
  - XML parsed with `defusedxml`.
- **Discovery order:**
  1. RSS/Atom (configured, autodiscovered from `<link rel="alternate">`, or common paths)
  2. sitemaps (from robots `Sitemap:` lines or common paths; indexes and `.gz` supported)
  3. explicitly tracked pages (pricing, homepage, product pages)

  Candidates are filtered by date and include/exclude patterns, with a cap per run.
- **Extraction:** trafilatura (main text as Markdown, title, author, date, categories, tags), plus OpenGraph and JSON-LD (`datePublished`, `@type`). "Thin" pages (likely JS-rendered) are flagged. A headless-render adapter is added **only** if a real competitor needs one.
- **Classification:** deterministic, from URL patterns, JSON-LD type and feed membership (Phase 1). LLM refinement of format and topics comes in Phase 3.
- **Terms of Service:** reviewing each competitor site's ToS remains the operator's responsibility. This is documented in the README.

### 8.2 LLM usage policy (interface in Phase 1; first calls in Phase 3)
- **One provider: Google Gemini**, via the official **`google-genai`** SDK. No OpenAI, Anthropic or other provider is added unless you explicitly ask for one. LangChain chat wrappers are not used; LangGraph orchestrates plain Python functions that call our interface.
- **Provider-agnostic interface.** Agents and services depend only on `app/llm` (`LLMProvider`, `LLMRequest`, `LLMResponse`, `StructuredResponse`, typed errors), obtained through `get_llm()`.
  - `app/llm/gemini.py` is the **only** module that imports the Gemini SDK, and it is imported lazily, so Phase 1 code paths never load it.
  - Adding another provider later means adding one module; callers don't change.
- **API surface: the Gemini Interactions API** (`client.aio.interactions.create`). Google made it GA in June 2026 and recommends it for new projects; new models and tools launch there. `generateContent` is legacy (still supported) and is not used.
- **Model:** `GEMINI_MODEL`, defaulting to `gemini-3.8-flash` (the latest stable model, and the one used in Google's quickstart). Every request can override the model, so later phases can send bulk work to a cheaper model such as `gemini-3.5-flash-lite` purely through configuration. Whether to do that is your decision, once quality is measured.
- **Reasoning depth:** set per route via `reasoning_effort` → Gemini `thinking_level` (`minimal` · `low` · `medium` · `high`). Low for bulk classification; high for writing, fact-checking and judging.
- **Structured outputs:** `response_format` with a JSON Schema generated from the Pydantic model, and the result validated with Pydantic. Responses are never parsed by stripping code fences.
- **Stateless calls:** `store=false` on every request. We keep our own history in Postgres and don't rely on server-side conversation state.
- **Retries and timeouts:**
  - The SDK retries 429 and 5xx responses itself, honoring `Retry-After`. This is configured through `HttpRetryOptions` (`LLM_MAX_RETRIES`) and `HttpOptions.timeout` (`LLM_TIMEOUT_SECONDS`).
  - There is no second retry layer.
  - SDK errors are mapped to our typed errors (auth, rate limit, unavailable, invalid request, bad response).
- **Research and grounding (Phase 5):** Gemini's **Google Search grounding** and **URL context** tools. Sources are persisted, and each drafted claim is tagged with a source ID so the fact-checker can verify it.
- **Cost levers (Phase 3):** only new or changed versions are analyzed; token budgets apply per run; usage (input, output, thought and cached tokens) is logged to `run_events`. Cheaper service tiers or batch processing can be used for bulk analysis if the volume justifies it.
- **Data terms:** use a billing-enabled (paid-tier) API key in production. At the time of writing, Google's terms allow content sent on the unpaid tier to be used to improve Google's products; check the current terms.
- **Untrusted input:** competitor pages and search results are treated as data. They are delimited in prompts. Agents that read them have no side-effecting tools.

### 8.3 Opportunity scoring (Phase 4)
Factors, each 0–100 and stored per opportunity with its evidence:

| Factor | Source | Direction |
|---|---|---|
| Competitor activity | Competitor items on the topic in the window (log-scaled, normalized) | + |
| Growth | Recent window vs prior window (smoothed) | + |
| Business relevance | LLM rubric judgment against `company_profile.yaml` (products, ICP, positioning), with rationale | + |
| Audience relevance | LLM rubric judgment against the ICP | + |
| Novelty | How recently the topic emerged across competitors | + |
| Search opportunity | Optional keyword-data provider. If null, the factor is dropped and the weights renormalize. | + |
| Existing company coverage | Your own published items on the topic (your site is crawled as `is_self`) | − |
| Saturation | Share of competitors covering the topic × volume | − |

`score = Σ wᵢ·f′ᵢ / Σ wᵢ`, where `f′ = f` for positive factors and `100 − f` for negative ones.
- Weights live in versioned `config/scoring.yaml`, and every score records its `weights_version`.
- Hard filters: excluded topics, and topics already generated or queued within N days.

This design is what lets the system answer "why was this topic selected?"

### 8.4 Blog generation workflow (Phase 5–6, LangGraph)
```
START → plan (topic analysis, search intent, angle vs competitor coverage)
      → research (Gemini Google Search grounding / URL context + internal competitor corpus; sources persisted)
      → outline (H1/H2/H3, FAQ candidates, internal-link targets from your site)
      → draft (claims tagged with source IDs) → edit (voice/tone from company profile)
      → fact_check (unsupported claims flagged)
      → seo_package (title, slug, meta, keywords, headings check, FAQ, links, CTA, category/tags, image suggestion)
      → quality_gate (deterministic checks + LLM judge + originality vs competitor corpus and your own site)
          ├─ pass → persist → ApprovalPolicy
          ├─ fixable & attempts < MAX_REVISIONS → revise → fact_check …
          └─ otherwise → persist as needs_review
```
- The Postgres checkpointer lets a failed run resume from its last completed node.
- Revision loops are bounded.
- Originality is measured as n-gram containment against each related competitor article, with thresholds for overlap and the longest verbatim run.

### 8.5 Human approval (the extension point)
Generation ends at the quality gate. At that point `ApprovalService.submit(article)` asks one `ApprovalPolicy`:
- `AUTO_PUBLISH=true` → `AutoApprovalPolicy`. Approves **only** if every hard check passed and the judge score clears its threshold. Otherwise the article goes to `needs_review`.
- `AUTO_PUBLISH=false` → `ManualApprovalPolicy`. The article moves to `awaiting_approval`, and a human acts through `POST /articles/{id}/approve|reject` (later, the dashboard).

Publishing only ever consumes `approved` articles, and re-checks that status itself. Approvals can take days, so they are a durable DB state transition, not a paused LangGraph thread. That keeps the approval logic out of the workflow.

### 8.6 Publishing, idempotency and the daily limit (Phase 7–8)
- **`CMSPublisher` protocol:** `find_by_slug`, `create`, `update`, `get`, `ensure_terms`, `healthcheck`.
  - `WordPressPublisher` is the first implementation.
  - `PublishingService` is the only caller. It is CMS-agnostic, so adding a CMS means adding an implementation.
- **WordPress specifics:**
  - Application Password over HTTPS, for a dedicated least-privilege user.
  - Markdown → HTML via markdown-it-py, then **sanitized with `nh3`**, so injected web content can't become stored XSS on your site.
  - Status mapping: `draft`, `publish`, or `future` + `date_gmt` for scheduled posts.
  - Categories and tags are fetched or created.
  - The slug is pre-checked.
  - SEO meta goes through a configurable adapter (`none` / Yoast / Rank Math). This needs a tiny mu-plugin that exposes those meta keys to REST; it ships in `deploy/wordpress/`.
- **Idempotency:**
  1. A `publications` row is written first, with a unique `idempotency_key`, status `in_progress`.
  2. The CMS call is made.
  3. Outcomes:
     - success → `succeeded`;
     - definite 4xx → `failed`;
     - **ambiguous** (timeout or connection loss after sending) → `unknown`.

  A POST is **never retried blindly**. An `unknown` row is reconciled by looking the post up by slug, and by our article ID stored in post meta when the mu-plugin is installed. Only GETs and updates to a known post ID are retried automatically.
- **Daily limit (`MAX_BLOGS_PER_DAY`), enforced inside `PublishingService`:**
  - A single transaction takes `pg_advisory_xact_lock(target, go_live_date)`.
  - It counts that day's `publish`/`schedule` publications with status `in_progress|succeeded|unknown`, then refuses with `DailyLimitReached` once the count reaches the cap.
  - Drafts don't count. Scheduled posts count toward the day they go live, in `PUBLISH_TIMEZONE`.
  - A concurrency test proves that exactly N publications succeed.
  - Approved articles that exceed the cap stay queued for the next day's capacity.

### 8.7 Scheduling (Phase 8)
- A separate `python -m app worker` process runs APScheduler 3 with cron triggers, all from env. The API process never runs jobs, so multiple uvicorn workers can't duplicate them.
- Every job takes a Postgres advisory lock: if the lock is already held, the run is skipped. Every run is recorded in `runs`.
- Any job can also be run manually via the CLI or API.
```
SCHEDULE_TIMEZONE=UTC
SCHEDULE_SCAN_CRON="0 */6 * * *"          # check competitors
SCHEDULE_ANALYZE_CRON="30 */6 * * *"      # analyze new/changed content (batch)
SCHEDULE_TRENDS_CRON="0 2 * * *"          # trends
SCHEDULE_OPPORTUNITIES_CRON="0 6 * * *"   # morning opportunities
SCHEDULE_GENERATE_CRON="0 7 * * *"        # generate BLOGS_TO_GENERATE_PER_DAY
SCHEDULE_PUBLISH_CRON="0 * * * *"         # drain approved queue within the daily limit
BLOGS_TO_GENERATE_PER_DAY=2
MAX_BLOGS_PER_DAY=2
PUBLISH_TIMEZONE=UTC
AUTO_PUBLISH=false
```
An empty cron value disables that job.

### 8.8 Observability
- `runs` and `run_events`, always on and queryable through the API, record:
  - which agent or step ran, and which tools it called (with arguments summarized);
  - which competitor and URLs were involved;
  - the topic factor scores and rationale;
  - gate results and publication outcomes;
  - LLM tokens and cost.
- structlog JSON logs carry bound context (`run_id`, `job`, `competitor`, `source`, `article_id`), and secrets are redacted.
- **LangSmith is optional.** If `LANGSMITH_TRACING=true` and a key is set, calls through the `app/llm` interface and LangGraph runs are traced. Otherwise it's a no-op, and the app runs fully without it.

### 8.9 Error handling
- Error taxonomy: `TransientError` (timeouts, 429, 5xx, connection), `PermanentError` (4xx, validation, robots-disallowed, blocked) and `AmbiguousOutcomeError` (a non-idempotent write with an unknown result).
- Retries (bounded, `Retry-After`-aware) apply **only** to idempotent operations. The crawler uses an explicit retry loop; LLM calls use the Gemini SDK's built-in retries. Structured output is validated, with one repair attempt (Phase 3); after that the item fails. There is never a placeholder.
- Per-item isolation: one failing competitor, source or page marks the run `partial` instead of aborting it.
- Duplicates are prevented by unique constraints (competitor + canonical URL, version hash, article slug, publication idempotency key) and by deduplicating opportunities against coverage and the queue.
- After repeated 403s or challenge pages, a host is marked `blocked` until someone resets it manually.

### 8.10 Security
- Secrets come only from the environment (`SecretStr`). `.env.example` is committed; `.env` is gitignored. Nothing secret appears in code or logs.
- An `X-API-Key` header is required on `/api/v1/*`, which covers approve and publish.
- Security controls referenced elsewhere in this plan: SSRF guard (§8.1); sanitized CMS HTML (§8.6); untrusted content kept away from side-effecting tools (§8.2); HTTPS-only WordPress with a least-privilege account (§8.6).
- `uv.lock` is committed.

### 8.11 Social monitoring feasibility (Phase 9, official/permitted routes only)

| Platform | Route | Feasible? |
|---|---|---|
| YouTube | YouTube Data API v3 (API key, daily quota) + public channel RSS | ✅ Videos/Shorts, titles, descriptions, tags, publish time, view/like/comment counts |
| Instagram | Graph API **Business Discovery**. Requires your own IG Business/Creator account linked to a Facebook Page, plus a Meta app. | ✅ For competitor Business/Creator accounts: captions, media type, timestamps, like/comment counts, permalinks |
| X / Twitter | Official X API, paid access tier | ⚠️ Works; the cost depends on current X pricing |
| LinkedIn | Official APIs only expose pages you administer; scraping violates LinkedIn's terms | ❌ Via API. Use manual/CSV import, or a licensed data provider after vendor and legal due diligence. |
| TikTok | Research API is restricted to approved researchers | ❌ For a commercial MVP |
| Podcasts, newsletters, changelogs, press pages | RSS / sitemaps (reuses the website pipeline) | ✅ |

---

## 9. Dependencies

### 9.1 Final set (versions current on PyPI as of 2026-09-13; exact pins go in `uv.lock`)

| Package | Version | License | Phase | Purpose |
|---|---|---|---|---|
| fastapi / uvicorn[standard] | 0.141 / 0.52 | MIT / BSD-3 | 1 | API server |
| pydantic / pydantic-settings | 2.13 / 2.15 | MIT | 1 | Validation, settings |
| httpx | 0.28 | BSD-3 | 1 | Async HTTP (crawler, WordPress, social APIs) |
| trafilatura | 2.2 | Apache-2.0 | 1 | Main-content and metadata extraction |
| protego | 0.6 | BSD-3 | 1 | robots.txt (wildcards, crawl-delay) |
| feedparser | 6.0 | BSD-2 | 1 | RSS/Atom |
| defusedxml | 0.7 | PSF | 1 | Safe sitemap XML parsing |
| pyyaml | 6.0 | MIT | 1 | Competitor / company / scoring config |
| structlog | 26.1 | MIT/Apache | 1 | Structured logs |
| typer | 0.27 | MIT | 1 | CLI |
| sqlalchemy / alembic | 2.0 / 1.20 | MIT | 2 | ORM, migrations |
| psycopg[binary,pool] | 3.3 | LGPL-3.0 | 2 | Postgres driver, sync and async; shared with the LangGraph checkpointer |
| google-genai | 2.23 | Apache-2.0 | 1 (interface only; first calls in Phase 3) | The single LLM provider: the official Gemini SDK, Interactions API |
| langgraph / langgraph-checkpoint-postgres | 1.2 / 3.1 | MIT | 5 | Content workflow + resumable checkpoints |
| langsmith *(optional extra)* | latest | MIT | 5 | Tracing when enabled |
| markdown-it-py / nh3 | 4.2 / 0.3 | MIT | 7 | Markdown → HTML, sanitization |
| apscheduler | 3.11 | MIT | 8 | Cron scheduling in the worker process |
| **dev:** pytest, pytest-asyncio, respx, ruff, mypy | 9.1, 1.4, 0.23, 0.16, 2.3 | MIT/Apache/BSD | 1 | Tests, HTTP mocking, lint, types |

**Conditional (added only on a demonstrated need):** `pgvector` + an embedding model; a headless renderer for JS-only competitor sites. Social APIs are called over `httpx`, so no platform SDKs are needed.

### 9.2 Dropped from the source repos
`upsonic`, `firecrawl-py` (R1) · `chromadb`, `openai`, `langchain-openai`, `langchain-community`, `langchain-anthropic`, `tavily-python`, `tiktoken`, `tokenizers`, `requests`, `markdown`, `ipykernel`, `apple-certifi`, `python-dotenv` (R2) · `langchain_groq`, `langchain_tavily`, `flask`, `flask-sock`, `gunicorn`, `xhtml2pdf`, `websockets`, `jinja2`* (R3).
\* Jinja2 may return in Phase 10 if the dashboard is server-rendered.

### 9.3 Conflicts and resolutions

| Conflict | Resolution |
|---|---|
| Three agent stacks (Upsonic; LangGraph + LangChain wrappers ×2) | LangGraph for the one workflow that needs it; every model call goes through `app/llm` → Gemini |
| Four LLM providers (Anthropic, OpenAI, Groq, Upsonic multi-provider) | Google Gemini only (`google-genai`); the legacy `google-generativeai` package is not used (PyPI marks it Inactive) |
| Two web frameworks (Flask in R3's code, FastAPI in R3's README) | FastAPI |
| Two storage engines (SQLite, ChromaDB) plus files on disk | PostgreSQL only |
| Three research/scrape vendors (Firecrawl; Tavily via two different packages) | Our own compliant crawler for monitoring; Gemini Google Search grounding for research |
| Sync `requests` vs async code | httpx everywhere |
| Python ≥ 3.13 (R2) vs 3.10 (R3) | 3.12 baseline; every chosen dependency supports 3.12–3.13 |
| R2 pin drift and undeclared direct deps; R3 unpinned | A single `pyproject.toml` + committed `uv.lock`; no source-repo lockfiles reused |

---

## 10. Change summary

### 10.1 Architecture changes
- **Before:** a one-shot script (R1); CLI-driven file pipelines (R2); a Flask app running agents inside a request handler (R3).
- **After:** a layered service (API / CLI / worker → services → agents & integrations → data). Long-running work happens in the worker process, and every run is recorded and auditable.

### 10.2 Database changes
- **Before:** R1 wrote a JSON file; R2 used ChromaDB plus Markdown files; R3 used a SQLite `reports` table (the committed DB is empty).
- **After:** the PostgreSQL schema in §7, managed with Alembic. There is no data to migrate.

### 10.3 API changes
R3's Flask routes (`/`, `/product`, `/ws/research`, `/api/reports*`, `/download-pdf`) are removed. New REST API under `/api/v1` (API-key protected):

| Phase | Endpoints |
|---|---|
| 1 | `GET /health` · `GET /api/v1/competitors` · `POST /api/v1/competitors/{slug}/scan` |
| 2 | competitors & sources CRUD · `POST /scans` → `run_id` · `GET /runs/{id}` (+ events) · `GET /content?competitor=&type=&since=` · `GET /content/{id}/versions` · `GET /changes?since=` |
| 3 | `GET /content/{id}/analysis` · `GET /topics` · `GET /trends` · `GET /competitors/{id}/profile` |
| 4 | `POST /opportunities/refresh` · `GET /opportunities` · `PATCH /opportunities/{id}` |
| 5–6 | `POST /articles` (from an opportunity) · `GET /articles` · `GET /articles/{id}` |
| 7 | `POST /articles/{id}/approve` · `POST /articles/{id}/reject` · `POST /articles/{id}/publish` · `GET /publications` |
| 8 | `GET /schedule` · `GET /limits` |
| 9 | `/social/accounts` · `/social/posts` |

### 10.4 Agent changes
| Source agent | Becomes |
|---|---|
| R1 research agent (Upsonic + Firecrawl tools) | The deterministic crawler (Phase 1–2), plus structured profile extraction (Phase 3) |
| R1 analysis agent | Grounded synthesis from structured, evidence-linked profiles (Phase 3) |
| R3 ResearcherAgent (competitor discovery) | Dropped (backlog: "suggest competitors") |
| R3 ReporterAgent | Structured change summaries and digests (Phase 3) |
| R2 planner / research / outline / writer / editor / SEO (ideas only) | Independently implemented LangGraph nodes, with fact-check, originality and quality-gate nodes added (Phase 5–6) |
| R2 approval service | `ApprovalPolicy` on final articles (Phase 7) |

---

## 11. Testing strategy
- **Unit tests** (no network, no DB) cover the pure logic:
  - robots rules and crawl-delay (fake clock), fetcher retry/Retry-After/size limits/SSRF guard;
  - sitemap indexes, `.gz` and XXE payloads; RSS and Atom parsing and autodiscovery;
  - extraction on fixture HTML; classification tables; change detection;
  - scoring math; daily-limit logic; approval policies; SEO validators; HTML sanitization; publisher request building.
- **HTTP contract tests** (`respx`):
  - WordPress create, update, taxonomy and scheduling;
  - **timeout-after-create → no second POST, reconciliation by slug**.
- **Integration tests** (real Postgres via `docker compose`, which tests skip if it's unavailable):
  - migrations up/down, repositories, idempotent upserts;
  - the **concurrent daily-limit test** (N+1 parallel publishes → exactly N succeed);
  - advisory-lock job overlap.
- **LLM code** runs behind the `app/llm` interface.
  - Unit tests mock the Gemini API at the HTTP level with `respx`, so the real SDK's request building and error mapping are exercised without any network call or API key. An autouse guard blocks real sockets.
  - A small opt-in eval set (`pytest -m llm_eval`, which costs money) guards prompt changes from Phase 3.
- **Phase 1 isolation:** a test asserts that the crawler, the monitoring service, the API and the CLI run with `GEMINI_API_KEY` empty and never import the Gemini SDK.
- **Workflow tests** run the LangGraph graph with fake nodes: routing, the bounded revision loop, failure and resume.
- **API tests** use an httpx `AsyncClient` against the app factory.
- **Live smoke tests** (`pytest -m live`, opt-in) run against real, robots-permitting sites.
- **CI:** a GitHub Actions workflow (ruff + mypy + pytest) on every push. Your remote is GitHub.

---

## 12. Phased delivery plan

At every phase: run the tests → run the app → verify against real inputs → fix → update docs → commit, then wait for your approval before starting the next phase.

### Phase 1 — Competitor website monitoring (no DB, no LLM)
**Scope**
- Project skeleton: uv/pyproject, ruff, mypy, pytest, CI, `.env.example`, `.gitignore`, README, `THIRD_PARTY_NOTICES.md`.
- `app/config.py` settings, and competitors loaded from a validated `config/competitors.yaml`.
- `app/crawling/`: polite fetcher, robots, per-host rate limiting, sitemap and feed discovery, extraction, page classification.
- `app/services/monitoring.py`: `scan_competitor(slug, since, limit) → ScanResult`. The result includes items with URL, type, title, date, author, categories, tags, word count and discovery source, plus the pages robots skipped and any errors.
- CLI: `python -m app scan <slug> --since 7d --limit 20`.
- API: `GET /health`, `GET /api/v1/competitors`, `POST /api/v1/competitors/{slug}/scan`.
- **LLM groundwork, with no calls:** the `app/llm` interface (`LLMProvider`, request and response types, typed errors), a lazily imported `GeminiProvider` on the Interactions API, the `get_llm()` factory, and the `GEMINI_API_KEY` / `GEMINI_MODEL` settings. Crawling, sitemaps, RSS, robots, extraction, URL normalization and classification stay deterministic and never touch this module.

**Acceptance criteria**
- `pytest` is green with no network, covering: robots disallow and crawl-delay, 429 + `Retry-After`, sitemap index and `.gz`, RSS/Atom, extraction fixtures, classification, and the SSRF guard.
- A real scan of your competitors lists their recent posts, pricing and landing pages.
- The API returns the same result as the CLI.
- The logs show robots decisions and per-host pacing.
- Everything above works with `GEMINI_API_KEY` empty. The Gemini provider's unit tests pass with a mocked API.

### Phase 2 — Persist competitor content
- Postgres via docker compose; SQLAlchemy models for the config, raw, normalized and ops layers; Alembic.
- Competitors are seeded from YAML.
- Versioning with text hashes, and change events (`new` / `updated` / `removed` / `pricing_changed`).
- Run history, and the query API ("what did X publish this week?").

### Phase 3 — Competitor analysis
- First Gemini calls through `app/llm`: structured outputs, per-route model and reasoning effort, usage tracking, and cost controls.
- Per-item content analysis (topics, format, audience, keywords).
- Topic taxonomy.
- Versioned competitor profiles (R1 schema) and summaries of pricing/messaging changes.
- Deterministic trend snapshots (frequency, topic growth, formats, strategy shift).

### Phase 4 — Opportunity detection
- Company profile; your own site crawled as `is_self`.
- Computation of each factor; configurable weights; LLM relevance judgment and angle.
- Ranked, explainable opportunities.

### Phase 5 — Blog generation
- The LangGraph workflow (plan → research → outline → draft → edit) with the Postgres checkpointer.
- Research sources persisted; claims tagged with source IDs (Gemini grounding metadata for web sources).

### Phase 6 — SEO and quality checks
- SEO packaging (all the blog output fields you listed), fact-check, originality, LLM judge, the bounded revision loop, and the quality report.

### Phase 7 — CMS publishing
- `CMSPublisher`, `WordPressPublisher` (draft/publish/schedule, taxonomy, slug, SEO meta via mu-plugin), idempotency and reconciliation.
- `ApprovalPolicy` and the approve/reject API.

### Phase 8 — Scheduling and daily limits
- Worker process, env-configured cron, advisory locks, the generate-N job, the publish-queue job, and the transactional daily limit, with its concurrency test.

### Phase 9 — Social monitoring
- `SocialSource` adapters for YouTube (API + RSS), Instagram (Business Discovery) and X (if you provide paid API access), plus CSV import for LinkedIn.
- Social posts feed into the same analysis and trend layers.

### Phase 10 — Dashboard
- Server-rendered (FastAPI + Jinja2 + HTMX) unless you prefer an SPA: competitor timeline, changes, trends, opportunities, and the article review/approve queue.

---

## 13. Consolidation procedure
1. Create branch `feat/phase-1-website-monitoring` from `main`. Its first commit is this `MIGRATION_PLAN.md`.
2. Scaffold the new project structure. No source repository is added as a submodule, subtree or dependency.
3. Copy the few adopted patterns from R3 (settings, prompt/agent separation, LangGraph node conventions), adapted with header attribution, as each phase needs them.
4. Import R1's profile schemas and grounded-synthesis pattern in Phase 3, adapted with header attribution.
5. Implement all R2-inspired capabilities independently (§3). No R2 files are opened during implementation beyond the analysis already done.
6. Create `THIRD_PARTY_NOTICES.md` with the R1 and R3 MIT notices.
7. Delete the temporary clones after approval. The final repository depends only on PyPI packages and external APIs.

---

## 14. Inputs needed from you
1. **Approval** of the base choice, the architecture and this plan.
2. **2–5 competitor URLs and your own site URL.** These are needed to verify Phase 1 against real sites; your site is later used for coverage and internal links. Pricing, product or other pages you want tracked explicitly are also useful.
3. *(Phase 3, not blocking):* a billing-enabled `GEMINI_API_KEY`. Phase 1 doesn't need one. Say so if you'd rather run bulk analysis on a cheaper Gemini model (e.g. Flash-Lite) from day one.
4. *(Phase 7, not blocking):* your WordPress version, SEO plugin (Yoast, Rank Math or none), and whether you can install a small mu-plugin.

---

## Appendix — what was verified, and how
- All source files in the three repos were read in full.
- Git histories were inspected: R2 has no license file in its entire history; R3's FastAPI → Flask switch and assistant-agent deletion are both in its history.
- R2's test suite was run in an isolated environment: **57 passed, 3 failed**. All tests target the approval module; there are no tests for the agents, the workflow, SEO or WordPress.
- R2 was grep-verified: the approval service has no callers outside tests; `markdown` is never imported; there is no checkpointer or interrupt usage.
- R3's committed `reports.db` contains only its schema. R3 never loads `.env` or reads its LangSmith settings.
- Current PyPI versions and license metadata were checked for every proposed dependency.
