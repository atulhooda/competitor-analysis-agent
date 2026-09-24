# Competitor Analysis Agent

An AI-powered competitor intelligence and content generation agent. It monitors competitors'
websites (and later their social channels), keeps a history of what they publish, finds content
opportunities, and generates and publishes original blog posts.

> **Status: Phase 8 of 10: scheduling and the autonomous pipeline.** Scans are deterministic
> and stored in PostgreSQL; Gemini analyzes what competitors publish (Phase 3); opportunities
> are scored deterministically (Phase 4); an **approved** opportunity becomes a researched,
> cited, edited draft (Phase 5), which Phase 6 fact-checks, scores and revises until it is
> `ready` or `needs_review`. Phase 7 publishes a ready article once its exact version is
> approved: to **your own website's GitHub repository**, as a branch, an MDX file and a pull
> request that a merge takes live (WordPress was the first adapter and is being retired).
> Phase 8 runs all of it on a schedule, within daily limits, and **everything automatic is
> off by default**. **Phase 8 introduces autonomous scheduling and pipeline
> orchestration. Social media automation is intentionally deferred to Phase 9.** See
> [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the architecture and roadmap.

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
- **Descriptive, not prescriptive.** Phase 3 reports what competitors do. Phase 4 turns that
  into recommendations for you.

**Content opportunities (Phase 4).** Answers "given what competitors are doing, what should we
create next?" with a ranked list of opportunities. Each one has:

- a 0–100 score that breaks down into named parts;
- the gaps behind it;
- a suggested format, audience and intent;
- the competitor pages it rests on;
- a status you control (`new → reviewed → approved / rejected → used`).

See [Content opportunities](#content-opportunities) for the methodology.

**Article drafts (Phase 5).** Writes an article for an opportunity you approved:

```
approved opportunity → brief (deterministic) → research (Google Search + URL context)
  → outline → draft → editorial pass → stored article, with sources and citations
```

- **Traceable.** Every article links to its opportunity, the assessment it was briefed from,
  the company profile version it was written with, and the evidence behind them.
- **Cited.** Each research-backed statement cites a stored source that Gemini actually read.
- **Checkpointed.** A failed or interrupted article resumes where it stopped.

**Phase 5 generates drafts but does not publish them.** See
[Article drafts](#article-drafts).

**Articles you write yourself.** `articles import <file>` turns one Markdown file with YAML
frontmatter into a `ready` article that goes through the same publishing machinery. It needs
no Gemini. What can be checked without it is checked (length, citations, structure, the site's
MDX contract); the fact-check, originality and judge gates are recorded as **not run**, and
the article, its version, the pull request and the publication record all say the piece was
written by a person and not machine-fact-checked. See
[Importing an article you wrote](#importing-an-article-you-wrote).

**Article validation (Phase 6).** Checks a completed draft and prepares it for a person to
publish:

```
completed article → fact-check → originality → SEO package → metrics → Gemini judge
  → decision (score + gates) → bounded revisions → ready / needs_review
```

- **Fact-checked.** Every cited claim is checked against the source it cites. Support needs a
  quote from the source, and agreement alone never counts. Uncited factual claims are flagged.
- **Measured.** Similarity to competitor pages, readability, structure, citations and SEO
  are computed in code. Gemini contributes one scored component, its quality rubric.
- **Gated.** An article is `ready` only when no mandatory gate fails (contradicted claims,
  unsupported claims, uncited claims, citation integrity, severe overlap, missing SEO fields,
  malformed content, minimum score).
- **Revised within a bound.** Failing articles are revised at most `QUALITY_MAX_REVISIONS`
  times. Every revision is a new version, validated again, and recommended only if it scores
  better.

**Phase 6 validates and prepares articles but does not publish them.** See
[Article validation](#article-validation).

**Publishing (Phase 7 + the GitHub adapter).** Sends an approved, ready article to your
website's repository:

```
ready article → approval (this exact version and quality report) → render (MDX)
  → preflight → branch blog/<slug> → src/content/blog/<slug>.mdx → pull request
  → Vercel preview verified → (merge → production deployment → live URL verified)
```

- **Approved, exactly.** An approval covers one version and one quality report. A new
  version or a new validation voids it.
- **A pull request by default.** The post goes live only when you ask for it and
  `PUBLISH_ALLOW_DIRECT_PUBLISH` is on; then the pull request is squash-merged and the live
  page is checked before the publication counts.
- **Never twice.** One publication per version and repository; a branch, a file, a pull
  request and a merge are each looked up before they are made, so a lost answer or a crash
  never creates a duplicate.

**Phase 7 publishes only approved, ready article versions.** See [Publishing](#publishing).

**Scheduling and the autonomous pipeline (Phase 8).** Runs the whole chain on a schedule:

```
schedule → scan → analyze → opportunities → generate (top N) → validate → READY gate
  → approval policy → daily publishing limit → WordPress draft → publish
```

- **Off by default.** `SCHEDULER_ENABLED=false`, `AUTOMATED_PUBLISHING_ENABLED=false`,
  `PUBLISH_AUTO_APPROVE=false`, `PUBLISH_ALLOW_DIRECT_PUBLISH=false`: nothing runs, and
  nothing is sent to WordPress, until you turn it on.
- **Daily limits, per local day.** `MAX_ARTICLES_GENERATED_PER_DAY` (3) is applied before any
  article is written; `MAX_ARTICLES_PER_DAY` (1) counts successful public posts and is
  enforced atomically inside the publisher.
- **Safe to repeat.** One job per scheduled occurrence, one running job per type, checkpoints
  per stage: a rerun, retry or crash never duplicates an article, an approval or a post.

See [Scheduling and the autonomous pipeline](#scheduling-and-the-autonomous-pipeline).

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

For content opportunities, describe your company, then generate:

```bash
cp config/company.example.yaml config/company.yaml  # your company profile (gitignored)
uv run python -m app company import
uv run python -m app opportunities generate         # works without GEMINI_API_KEY (no interpretation)
uv run python -m app opportunities                  # the ranked list
```

To write a draft, approve an opportunity, check its brief, then generate (needs
`GEMINI_API_KEY`):

```bash
uv run python -m app opportunities approve 12
uv run python -m app articles brief 12              # the deterministic brief (no Gemini)
uv run python -m app articles generate 12           # research, outline, draft, edit
uv run python -m app articles show 1                # the draft, with its sources (not published)
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

# Company profile (what opportunities are scored against)
uv run python -m app company import                  # new version from COMPANY_FILE (no-op if unchanged)
uv run python -m app company show                    # the current version
uv run python -m app company versions                # version history

# Content opportunities
uv run python -m app opportunities generate          # score, store, interpret the top candidates
uv run python -m app opportunities generate --no-interpret   # deterministic only, no Gemini call
uv run python -m app opportunities generate --window-days 90 --force
uv run python -m app opportunities                   # ranked list (new, reviewed, approved)
uv run python -m app opportunities list --status approved --min-score 60 --topic ai-agents --competitor acme
uv run python -m app opportunities show 12           # breakdown, gaps, why, interpretation, events
uv run python -m app opportunities evidence 12       # the stored evidence behind the current score
uv run python -m app opportunities history 12        # every assessment, with why the score changed
uv run python -m app opportunities approve 12 --note "for Q4"   # also: review, reject, use, expire, reopen
uv run python -m app opportunities list --origin editorial   # or: competitors

# Editorial topics: article ideas from your company profile alone (needs GEMINI_API_KEY)
uv run python -m app editorial propose --count 10    # ideas → opportunities (status new)
uv run python -m app editorial propose --dry-run     # see the ideas, save nothing (one call)
uv run python -m app editorial                       # list editorial topics

# Article drafts (needs GEMINI_API_KEY, except `brief`; nothing is published)
uv run python -m app articles brief 12               # preview the brief for opportunity 12
uv run python -m app articles generate 12            # write a draft for approved opportunity 12
uv run python -m app articles generate 12 --regenerate   # a new attempt after a failed or cancelled one
uv run python -m app articles                        # list (also: --status, --opportunity, --since)
uv run python -m app articles show 1                 # status, steps, brief, issues, Markdown preview
uv run python -m app articles sources 1              # retrieved sources, their facts, citation counts
uv run python -m app articles versions 1             # outline, draft and edited versions (--show ID)
uv run python -m app articles steps 1                # the checkpoint log (fingerprints, prompts, tokens)
uv run python -m app articles resume 1               # continue from the first unfinished step
uv run python -m app articles cancel 1               # stop for good

# An article you wrote yourself (no Gemini; nothing is published)
uv run python -m app articles import drafts/my-post.md          # → a ready article + its own opportunity
uv run python -m app articles import drafts/my-post.md --opportunity 12   # attach it to an existing one
uv run python -m app articles import drafts/my-post.md --json   # ids, slug, word count, what it did

# Article validation (needs GEMINI_API_KEY; nothing is published)
uv run python -m app articles validate 1             # fact-check, originality, SEO, score, revise → ready / needs_review
uv run python -m app articles quality 1              # score breakdown, gates, issues, metrics, judge, versions (--version ID)
uv run python -m app articles fact-check 1           # every claim check (--verdict contradicted, --version ID)
uv run python -m app articles originality 1          # passages similar to stored competitor/company pages
uv run python -m app articles seo 1                  # keywords + evidence, meta tags, slug, links, FAQ, checks
uv run python -m app articles revisions 1            # the edited version and every revision, with scores
uv run python -m app articles revise 1 --note "..."  # one more revision of the recommended version

# Approval and publishing (a pull request on the site's repository by default)
uv run python -m app articles approval 1             # recommended version, score, gates, approval state
uv run python -m app articles approve 1 --note "..." # approve that exact version (asks to confirm; --yes)
uv run python -m app articles reject 1 --note "..."  # reject it, with a reason
uv run python -m app articles approvals 1            # every decision, and why it stopped applying
uv run python -m app articles preflight 1            # every check, WordPress included (read-only)
uv run python -m app articles publish 1 --dry-run    # preflight + rendered HTML + WordPress request; changes nothing
uv run python -m app articles publish 1              # create or update the WordPress draft
uv run python -m app articles publish 1 --status publish   # make it public (needs PUBLISH_ALLOW_DIRECT_PUBLISH)
uv run python -m app articles publication 1          # post id, URL, what was mapped, every attempt
uv run python -m app articles publications 1         # every publication (one per version and site)

# Scheduling and the pipeline (Phase 8)
uv run python -m app pipeline run --dry-run          # plan: no fetch, no Gemini call, no CMS change
uv run python -m app pipeline run                    # the full pipeline, now (same locks and limits)
uv run python -m app schedule run scan               # one job now: scan, analyze, opportunities,
                                                     #   generate_articles, quality_check, publish
uv run python -m app schedule status                 # dashboard: today's counts, allowances, next runs
uv run python -m app schedule list                   # schedules, next runs, last job
uv run python -m app schedule pause --reason "..."   # pause schedules (manual runs still work)
uv run python -m app schedule resume
uv run python -m app jobs list --status failed       # recent jobs
uv run python -m app jobs show 12                    # stages, checkpoint, attempts, report
uv run python -m app jobs retry 12                   # continue a failed job from its checkpoint
uv run python -m app jobs cancel 12                  # queued: now; running: at its next checkpoint
uv run python -m app worker                          # the scheduler process (SCHEDULER_ENABLED=true)

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
| GET / PUT | `/api/v1/company-profile` | The current company profile / save it (a new version only if it changed: `{"created": false}` otherwise) |
| GET | `/api/v1/company-profile/versions` | Company profile version history |
| POST | `/api/v1/opportunities/generate` | Start a generation run: `202` plus a run to poll, or `?wait=true`. Body: `{"window_days": 60, "interpret": true, "force": false}`. `409` without a company profile or while a generation is running |
| GET | `/api/v1/opportunities` | Ranked by score. Filters: `status` (repeatable; default `new`, `reviewed`, `approved`), `min_score`, `topic` (slug or part of the name), `competitor` (slug in the evidence), `created_since`, `scored_since`, `limit`/`offset` |
| GET | `/api/v1/opportunities/{id}` | Current assessment (score breakdown, gaps, suggestion and reasons, Gemini interpretation) and the event timeline |
| GET | `/api/v1/opportunities/{id}/evidence` | Evidence rows of the current assessment, or of a past one (`?assessment_id=`) |
| GET | `/api/v1/opportunities/{id}/history` | Every assessment: score, company profile version, what changed |
| PATCH | `/api/v1/opportunities/{id}` | Change status: `{"status": "approved", "note": "..."}`. `409` if the transition isn't allowed |
| GET | `/api/v1/opportunities/{id}/brief` | The deterministic brief an article would get (no Gemini, nothing stored) |
| POST | `/api/v1/articles` | Write a draft for an approved opportunity: `202` with the article and its queued run (runs in the background), or `?wait=true`. Body: `{"opportunity_id": 12, "regenerate": false}`. `200` with the existing article if the opportunity already has one; `409` if the opportunity isn't approved; `503` without `GEMINI_API_KEY` |
| GET | `/api/v1/articles` | Newest first. Filters: `status` (repeatable), `opportunity_id`, `created_since`, `created_until`, `limit`/`offset` |
| GET | `/api/v1/articles/{id}` | Status, progress, current step, brief, steps (prompt versions, models, tokens), runs, content (Phase 6's recommended version once validated, else the edited version, or the draft until then), outline, issues, quality score, failure details (`?include_markdown=true` for a preview) |
| GET | `/api/v1/articles/{id}/sources` | Retrieved sources with their facts and citation counts (`?all=true` for earlier research runs) |
| GET | `/api/v1/articles/{id}/versions`, `…/versions/{version_id}` | Every outline, draft and edited version / one version with its claim → source citations and editor notes |
| GET | `/api/v1/articles/{id}/steps` | The checkpoint log |
| POST | `/api/v1/articles/{id}/resume` | Continue from the first step needing work (`202`), or `200` when nothing needs redoing. `409` while running, for a cancelled article, or when the article's token budget is spent |
| POST | `/api/v1/articles/{id}/cancel` | Stop for good; a run in progress stops before its next step |
| POST | `/api/v1/articles/{id}/validate` | Validate a completed article: `202` with the article (`validating`) and its queued run, or `?wait=true`. Resumes a failed validation. `409` for an article still being written, cancelled, running or out of budget; `503` without `GEMINI_API_KEY` |
| POST | `/api/v1/articles/{id}/revise` | One more revision of the recommended version, validated like the others. Body: `{"note": "..."}` (optional). `409` before the first validation |
| GET | `/api/v1/articles/{id}/quality` | Status and current step, score, breakdown, gates, issues, metrics, judge rubric, every validated version's score, revision count, recommended version (`?version_id=` for another version) |
| GET | `/api/v1/articles/{id}/fact-check` | Every claim check with its verdict, explanation, evidence, confidence, model and prompt version (`?verdict=` repeatable, `?version_id=`) |
| GET | `/api/v1/articles/{id}/originality` | The similarity report: flagged passages, the page and the overlapping text (`?version_id=`) |
| GET | `/api/v1/articles/{id}/seo` | The SEO package and its checks (`?version_id=`) |
| GET | `/api/v1/articles/{id}/revisions` | The edited version and every revision: parent, reason, issues addressed, changes, tokens, score, whether it's recommended |
| POST | `/api/v1/articles/{id}/approve` | Approve the recommended version and its quality report. Body: `{"note": "...", "approver": "..."}` (optional). `409` unless the article is `ready` |
| POST | `/api/v1/articles/{id}/reject` | Reject it. Body: `{"note": "..."}` (required: the reason) |
| GET | `/api/v1/articles/{id}/approval` | The approval as it stands (`not_ready`, `pending`, `approved`, `rejected`, `invalidated`), with the version, score, gates and what blocks publishing |
| GET | `/api/v1/articles/{id}/approvals` | Every decision, oldest first, with why and when it stopped applying |
| POST | `/api/v1/articles/{id}/preflight` | Every publishing check, WordPress included (read-only). Body: `{"status": "draft"}` (optional) |
| POST | `/api/v1/articles/{id}/publish` | `202` with `{"publication_id": ..., "status": "queued"}`; the CMS work runs in the background (`?wait=true` to wait). `?dry_run=true`: preflight, rendered HTML and the WordPress request, changing nothing. Body: `{"status": "draft" / "pending" / "publish"}`. `409` if not ready or not approved, or if `publish` isn't allowed; `503` if WordPress isn't configured |
| GET | `/api/v1/articles/{id}/publication` | The latest publication: status, post id, public URL, what was mapped, the last preflight, every attempt |
| GET | `/api/v1/articles/{id}/publications` | Every publication, newest first |
| POST | `/api/v1/pipeline/run` | Start the pipeline now: `202` with the queued job, which runs in the background. Body: `{"job_type": "full_pipeline"}` (or one stage), `{"dry_run": true}` → `200` with the plan. A job of a type already running is skipped |
| GET | `/api/v1/jobs` | Recent jobs, newest first (`?status=`, `?job_type=`, `?limit=`) |
| GET | `/api/v1/jobs/{id}` | One job: stages, checkpoint, attempts, heartbeat, error, report |
| POST | `/api/v1/jobs/{id}/retry` | Retry a failed job: `202` with a new job continuing from its checkpoints. `409` unless it failed |
| POST | `/api/v1/jobs/{id}/cancel` | Cancel a queued job, or stop a running one at its next checkpoint. `409` once finished |
| GET | `/api/v1/schedule` | The configured schedules, their next runs and last job |
| GET | `/api/v1/schedule/status` | The dashboard: switches, today's generated / ready / published and the remaining allowances, today's jobs, next runs, warnings |
| POST | `/api/v1/schedule/pause`, `/api/v1/schedule/resume` | Pause or resume scheduled runs. Body: `{"reason": "..."}` (optional) |

`/api/v1/*` requires the `X-API-Key` header whenever `API_KEY` is set. Outside development,
requests are refused until it is. Interactive docs are served at `/docs`.

> Changed in Phase 2: the Phase 1 endpoint `POST /api/v1/competitors/{slug}/scan` is now
> `POST /api/v1/competitors/{slug}/scans`, and scans are persisted.

## Content opportunities

Phase 4 answers one question: *given what competitors are doing, what should we create next?*

### What is computed, and what Gemini writes

| Computed deterministically (no LLM) | Written by Gemini (top candidates only) |
|---|---|
| Topic frequency, growth, and which competitors are growing on it | Working title |
| Competitor coverage and count | Recommended angle |
| Publishing frequency, recency | Why now |
| Format, audience and intent distributions | Target audience, recommended format, search intent |
| Saturation, overlap, seven gap types | Differentiation strategy, strategic rationale |
| Strategic fit with your company profile | Confidence (the model's own estimate, 0–1) |
| The 0–100 score, its breakdown and the ranking | |
| A suggestion (format, audience, intent) with "why" bullets | |

Gemini never computes a statistic or a score. It is shown the computed numbers and the
competitor pages. Before anything is stored, every sentence it writes that contains a number
not in that evidence is deleted. A title with an invented statistic falls back to the topic
name. If Gemini isn't configured or fails, every opportunity is still complete: score,
breakdown, gaps, suggestion and evidence.

### Your company profile

Opportunities are scored against your company profile, never a hardcoded industry.
[`config/company.example.yaml`](config/company.example.yaml) documents every field:

- description and products;
- target audiences;
- core, adjacent and excluded topics;
- preferred formats;
- positioning, differentiators and tone.

Copy it to `config/company.yaml` (gitignored) and run `company import`, or send it with
`PUT /api/v1/company-profile`. Keep secrets out of it.

- **Versioned.** Saving an identical profile does nothing; any change creates a new version.
- **Recorded per score.** Every assessment records the profile version it was scored with. The
  next `generate` re-scores against the latest version, and the opportunity's history says
  "company profile changed" where that moved the score.
- **Scoring fields.** Description, products, audiences, topics and formats affect scoring.
  Positioning and differentiators only feed Gemini's interpretation. Tone is kept for writing
  in later phases.

### Pipeline

```
latest page analyses + topic taxonomy + company profile + scoring config    (PostgreSQL)
  ─► signal engine: one candidate per canonical topic, plus core topics nobody covers   (no LLM)
  ─► qualify (relevance, pages, minimum score) ─► deduplicate ─► cap
  ─► store: one opportunity per topic; a new assessment only if the score or its basis changed;
     evidence rows; events; expire what no longer qualifies; reopen what qualifies again
  ─► Gemini: interpret the top new or changed opportunities (batched, budgeted, reused when unchanged)
  ─► API / CLI
```

- **One run at a time.** A Postgres advisory lock prevents overlapping runs.
- **Recorded.** Each run is logged in `runs` with kind `opportunities`.
- **Execution.** Runs happen in the background through the API, or synchronously from the CLI.

### Scoring

A candidate scores five positive dimensions, each a 0–1 value times its weight, minus a
saturation penalty. The positive weights are rescaled to total 100, so:

- the score is always 0–100;
- the breakdown always adds up to the score.

Defaults are below. Every weight and threshold can be changed in
[`config/scoring.example.yaml`](config/scoring.example.yaml).

| Dimension | Max points | Value (0–1) |
|---|---|---|
| momentum | 20 | 0.6 × growth + 0.4 × breadth. Growth = clamp(0.5 + log₂((recent+1)/(previous+1)) / 4): flat → 0.5, 4× → 1. Breadth = share of compared competitors publishing more in the last 60 days than in the 60 before. Reliable publication dates only; with too little history, a neutral 0.3 |
| strategic_fit | 25 | Match with your profile. A core topic: 1.0 (same), 0.85 (contains), 0.7 (overlaps). An adjacent topic: 60% of that. A subtopic matching a core topic: 0.5. Words all in your description or products: 0.35. Plus up to 0.2 when the topic's competitor pages have keywords matching your core topics. An excluded topic scores 0 |
| audience_fit | 15 | Share of the topic's pages aimed at your best-matched audience, where 15% of pages counts as full fit (0.5 if you list no audiences) |
| content_gap | 25 | The strongest gap × its gap weight (below) |
| recency | 15 | 0.5^(days since the last competitor page / 30) |
| saturation | −15 | A penalty (below) |

**Saturation** isn't simply "many pages":

- raw = 0.35 × volume + 0.30 × breadth + 0.20 × frequency + 0.15 × format variety, where:
  - volume is the log-scaled page count (30 pages ≈ full);
  - breadth is the share of competitors covering the topic;
  - frequency is pages per week in the window (1 per week = full);
  - variety is the number of distinct formats / 5.
- effective = raw × (1 − 0.5 × relief), where relief is the strongest of the freshness, depth,
  audience and intent gaps.

So a crowded topic whose coverage is stale, shallow, or misses your audience still leaves room.
A crowded topic that competitors already cover well is penalized.

**Gaps.** Each gap type is scored 0–1 and stored separately:

| Gap | Score | Meaning |
|---|---|---|
| topic | 1 − covering / total competitors (with ≥ 2 competitors) | Few competitors cover it. Once ≥ 20 pages are analyzed, a core topic of yours that no competitor covers becomes its own candidate, with topic gap 1 |
| audience | 1 − best audience share / 0.5 | Your audiences are underserved |
| intent | 1 − valuable-intent share / 0.2 | Commercial and comparison intents (configurable) are rare |
| format | 1 − valuable-format share / 0.2 | Your preferred formats are rare (default: tutorial, guide, comparison, case study) |
| depth | max(fragmentation, shallowness) | Fragmentation: most subtopics (≥ 3) touched by a single page. Shallowness: the median page has under 800 words (full gap at 200) |
| freshness | (median page age − 90) / (365 − 90) | Competitor content is old |
| differentiation | (dominant format share − 0.5) / 0.5 | Everyone covers it the same way (≥ 2 competitors, ≥ 4 pages) |

Audience, intent, format and depth gaps are scaled by min(1, pages / 4), so one or two pages
can't make a strong gap. Low coverage alone is not an opportunity. Strategic fit must be at
least 0.2, and a topic needs at least 2 competitor pages.

**Qualification, deduplication, ranking.**

- **Rejection.** A candidate is rejected when:
  - your profile excludes it;
  - its strategic fit is below 0.2;
  - it has fewer than 2 pages;
  - its score is below 40;
  - it is a near-duplicate of a better candidate;
  - it falls beyond the top 25.

  Rejections are counted by reason in the run summary.
- **Deduplication.** Candidates are canonical taxonomy topics (Phase 3 aliases and merges), so
  spelling variants are already one topic. Near-duplicates the taxonomy kept apart ("Workflow
  automation" and "Automating workflows") are detected from shared word stems (Jaccard ≥ 0.75
  over names and aliases). The lower-scored one is folded into the other as a related topic.
  No LLM is involved.
- **Suggestion.** Every opportunity gets a deterministic suggestion:
  - the valuable or preferred format competitors use least;
  - the intent they neglect (when the intent gap ≥ 0.3);
  - your most underserved audience, or your best-served one when no audience gap applies;
  - "why" bullets with exact numbers, e.g. "topic growth: +200% (2 → 6 items, 60-day windows)"
    or "covered by 2 of 3 competitors (8 pages)".

### Gemini's role

- **Top candidates only.** Only the top 10 (`interpretation.candidates`) opportunities are
  sent. They must be `new`, `reviewed` or `approved`, and score at least 50. Calls go 4 per
  batch to `GEMINI_SYNTHESIS_MODEL`. Everything else stays deterministic.
- **What the prompt contains.** Your company profile and, per opportunity:
  - its score breakdown, signals, gaps and deterministic suggestion;
  - up to 8 competitor pages (summary, format, audiences, angle), wrapped in delimiters and
    treated as untrusted data.
- **How output is checked.** Output is structured (a Pydantic schema), then:
  - sentences with numbers that aren't in the evidence are removed;
  - cited pages must be evidence rows of that assessment;
  - an answer left with no angle or "why now" is marked `failed`.
- **Reuse.** An interpretation is reused, with no call, while the evidence pages, the
  suggestion, the company profile, the prompt version and the model are all unchanged.
- **Failures never touch scores.**
  - Without `GEMINI_API_KEY`, interpretations are `skipped`.
  - On a budget, rate-limit or outage error, the remaining ones are skipped and the run ends
    `partial`.
  - Unusable output is retried in halves, then marked `failed`.
- **Budgets.** Calls share Phase 3's token budgets and the `llm_calls` ledger (purpose
  `opportunity_interpretation`).

### Provenance: why was this recommended?

`opportunities show`, `GET /api/v1/opportunities/{id}` and `…/evidence` answer from stored data
only:

- **Score breakdown.** Points per dimension, with the value and the signal behind each.
- **Gaps and reasons.** Every gap with its score and detail, plus the deterministic "why" bullets.
- **Evidence rows**, stored per assessment:
  - topic metrics and the topic trend;
  - the competitor pages (URL, title, date, format, audiences, summary, angle, and the analysis
    and content version ids);
  - competitor positioning profiles and strong gaps;
  - the company profile version and what matched;
  - near-duplicate topics.
- **The basis.** The scoring config fingerprint, company profile version, window, and the
  analysis ids the score was computed from.
- **Gemini's interpretation**, with its model, prompt version and the evidence it cited.

### Lifecycle, history and idempotency

| Status | Meaning | Can change to |
|---|---|---|
| `new` | Just found | reviewed, approved, rejected, expired |
| `reviewed` | Looked at, undecided | new, approved, rejected, expired |
| `approved` | To be written (Phase 5 will pick these up) | reviewed, rejected, used, expired |
| `rejected` | Not for us | reviewed |
| `used` | Turned into content | nothing (final) |
| `expired` | No longer qualifies | new, reviewed |

- **One opportunity per topic.** Re-running generation updates the existing opportunity instead
  of adding a duplicate. It's keyed by canonical topic, or `core:<topic>` for a core topic
  nobody covers.
- **Immutable assessments.** A new assessment is stored only when the score moves by at least
  1 point (`min_score_change`) or its basis changes: analyses, company profile, scoring config
  or window. Each one records what changed, for example `72 → 88: momentum +9.1 pts (recent
  items 2 → 6; growing competitors 1 → 2); analysed pages: 4 added, 0 dropped`. `opportunities
  history` lists them all.
- **Idempotent.** A re-run with nothing new creates no assessment, event or Gemini call.
  `--force` re-assesses and re-interprets anyway, recorded as "recalculated on request".
- **Expiry.** The next generation expires a `new` or `reviewed` opportunity that no longer
  qualifies, and records why (for example "strategic fit 0.1 is below the minimum 0.2" or
  "topic merged into 'AI agents'"). If it qualifies again, it's reopened as `new`. Approved,
  rejected and used opportunities are re-scored but never changed automatically.
- **Staleness.** An open opportunity that no generation has re-confirmed within 30 days
  (`expires_after_days`) is flagged `stale` in listings.
- **Events.** Every status change is an event: who, when, from → to, and a note.

### Limitations

- **Coverage.** Signals only reflect what was scanned and analyzed. Pages beyond the scan
  limits and undated pages count only partly. Growth needs reliably dated history in both
  windows.
- **No demand data.** There's no search volume, keyword difficulty or traffic data; demand is
  inferred from competitor activity alone.
- **Your own content.** Your site isn't compared, so a topic you've already covered can still
  be recommended. Mark it `used` or `rejected`.
- **Lexical matching.** Relevance matching uses word stems, not meaning. Synonyms need taxonomy
  aliases or `topics merge`.
- **Uncalibrated.** Weights and thresholds are reasoned defaults, not calibrated against
  results. Gemini's confidence is the model's own estimate.
- **Digits only.** The number check only works on digits; "three times" isn't caught.

## Editorial topics

Competitor monitoring only finds topics your competitors write about. Editorial topics fill
the rest of the blog: Gemini proposes article ideas from your company profile alone, and
each accepted idea becomes an ordinary opportunity (key `editorial:<topic>`) that takes the
same path as every other one: approval, article, validation, article approval, publishing.

```
company profile + what is already covered (opportunities, articles, your site's posts)
  → Gemini proposes ideas: topic, title, angle, audience, format, key points
    (about a third more than asked for: some won't pass the checks)
  → deterministic checks: excluded topics, strategic fit, near-duplicates of anything
    covered or of each other, numbers not in your profile
  → the best ideas by strategic fit → opportunities, status new
```

- **Scores are never Gemini's.** An idea's score is its strategic fit to your profile × 100,
  from its topic, keyword and title: on a core topic 70–100, which clears
  `PIPELINE_MIN_OPPORTUNITY_SCORE` (60); on an adjacent topic 42–60, which waits for a
  person. Gemini also names the profile topic each idea serves, but the claim counts only when
  the idea's own words share at least two words with that topic (audience words most of your
  topics carry, such as "clinics", don't count); it then scores like a partial match (70
  core, 42 adjacent).
- **No duplicates.** The ideas are checked against every opportunity (any status, so rejected
  and expired ones never come back), every article, and your site's posts (read from
  `PUBLISH_SITE_URL`'s sitemap, politely: robots.txt, rate limits and the SSRF guard apply).
  If the sitemap can't be read, the run still checks the database and reports it.
- **No invented facts.** Ideas contain no statistics: sentences with numbers your profile
  doesn't contain are removed, and a title with one falls back to the topic. The article's
  research step finds sourced numbers later.
- **The brief reads them like any opportunity.** The idea is stored as the opportunity's
  interpretation (title, angle, audience, format, intent), its key points lead the brief's,
  and there is no competitor evidence, so none is cited.
- **Owned by the editorial planner.** `opportunities generate` never expires, rescores or
  re-interprets an editorial opportunity. Unapproved ones expire after the scoring
  configuration's `expires_after_days` (30); approved ones stay until written.
- **One Gemini call per run**, recorded like every call (`LLM_DAILY_TOKEN_BUDGET` applies).
  A failed call fails the run and saves nothing. `--dry-run` shows the ideas without saving
  any. Every idea, kept or rejected with the reason, is kept in the run's summary
  (`GET /api/v1/editorial/runs/{id}`).

In the pipeline, the `editorial` stage runs between `opportunities` and `generate` and
proposes just enough ideas that today's `MAX_EDITORIAL_ARTICLES_PER_DAY` has opportunities to
write (at most `EDITORIAL_TOPICS_PER_RUN`). Ideas still waiting count: with
`PIPELINE_APPROVE_OPPORTUNITIES=false` they wait for you, and the stage proposes nothing more
until they are approved, rejected or expired. `MAX_EDITORIAL_ARTICLES_PER_DAY=0`, the
default, turns the stage off.

Limitations: ideas come from the model's general knowledge, steered by your profile; there is
no search-volume data. Duplicate detection matches word stems, so two different wordings of
the same question can both pass (review before approving). A competitor topic that appears
later can overlap an editorial one: the competitor engine doesn't check editorial topics.

## Article drafts

Phase 5 turns an approved opportunity into a complete blog draft and stores it.
**Phase 5 generates drafts but does not publish them.** Publishing is Phase 7.

### Flow

| Step | Status | How |
|---|---|---|
| brief | stored when the article is created | Deterministic, from stored data. No LLM |
| research | `researching` | Web search finds sources; a page-reading tool reads them |
| outline | `outlining` | Structured sections |
| draft | `drafting` | Structured content with citation markers |
| edit | `editing` | Editorial pass, then the completion checks |
| | `completed` | Stored; a person reviews it |

The five steps are written by **one provider**: Gemini by default, or Claude — see
[Claude: the second writer](#claude-the-second-writer) for `WRITING_PROVIDER`, the
`split` routing and the automatic fallback. `articles.writer` records which one wrote the
article, and `articles show` prints it.

Only an opportunity with status `approved` can get an article. Its generation runs in the
background (`POST /api/v1/articles`, or synchronously with `articles generate`).

### One live article per opportunity

- An opportunity has at most one article that is in progress or completed. A database
  constraint enforces it, even for concurrent requests.
- Asking again returns the existing article.
- A **failed** article is resumed (`articles resume`).
- A **cancelled** article is final.
- `regenerate` starts a new attempt after a failed or cancelled one; the old attempt is kept.
- A generation run stops if its opportunity stops being approved.

### The brief

The brief is built from stored data only:

- the opportunity and its score;
- the assessment's gaps and suggestion;
- Gemini's Phase 4 interpretation, if there is one;
- the evidence rows (competitor pages and positioning);
- the company profile (description, products, audiences, positioning, differentiators, tone).

It defines:

- the topic, working title, target audience and search intent;
- the angle, content type and desired outcome;
- the key points to cover and the competitor weaknesses to address;
- the differentiation strategy and things to avoid;
- the evidence: competitor pages, as context, never as facts.

`provenance` names the input behind each choice. The same inputs always give the same brief.
Preview it with `articles brief` before spending anything. It's stored when the article is
created.

### Research

1. **Discover.** One Gemini call with Google Search grounding.
   - Gemini identifies the claims that need evidence (at most `ARTICLE_RESEARCH_MAX_QUERIES`
     research questions) and proposes candidate sources.
   - It prefers official documentation, primary sources, research, regulators and reputable
     organizations, and avoids vendor marketing.
   - The actual Google queries are recorded.
2. **Screen** (deterministic). Each URL must be a public http(s) address:
   - no credentials, non-standard ports, `javascript:`/`file:` URLs, or private, loopback or
     link-local hosts;
   - the domain must resolve, checked with the Phase 1 SSRF guard.

   Duplicates are dropped. Your site and competitors' sites are recognized by domain. The most
   authoritative candidates come first, capped at `ARTICLE_RESEARCH_MAX_SOURCES`.
3. **Read.** Gemini's URL context tool retrieves the pages (at most
   `ARTICLE_RESEARCH_MAX_URL_CONTEXT_CALLS` calls) and extracts facts with supporting excerpts.
   This process never fetches a page itself.

A URL becomes a source **only if the URL tool reports that it retrieved the page**, so a made-up
or broken URL never reaches the article. Every candidate's outcome is recorded (retrieved, not
retrieved, rejected, skipped) with the reason. Research needs at least
`ARTICLE_RESEARCH_MIN_SOURCES` usable sources, or the step fails and keeps what it found.
Facts from competitor or company pages are marked `attribution_required`: the article may
state them only as attributed claims.

### Citations and provenance

- **The chain.** `article claim → source label → stored source → URL`.
  - The content marks research-backed statements with inline labels such as `[S3]`.
  - Every draft and edited version stores each claim with the sources it cites
    (`article_citations`, with foreign keys to `article_sources`).
  - Phase 6 fact-checking can read these directly.
- **Unknown labels.** A citation label that doesn't match a stored source is removed and
  recorded as an issue. Sources are never invented.
- **Numbers.** Sentences with a number found in neither the research nor the brief are flagged
  for review.
- **Format.** The content is structured JSON (sections → paragraphs, lists, subheadings), not
  HTML, so Phase 7 can render HTML, Markdown or CMS formats. The Markdown preview in
  `articles show` is a review aid only.

### Safety

- **Untrusted content.** Web pages, competitor context and drafts are untrusted content. They
  are wrapped in delimiters they can't close or imitate. Every prompt separates the system
  instructions from them and says to ignore instructions embedded in them.
- **Tools.** Only research calls have tools (search, URL reading). The outline, draft and edit
  calls have none, and nothing in the pipeline executes or publishes anything.
- **Competitor text.** Competitor pages inform differentiation only. The writing prompts get
  Phase 3 summaries and angles, never competitor page text. Copying, close paraphrase and
  unattributed competitor claims are forbidden in the prompts.
- **Company facts.** These come only from the company profile; anything else is treated as
  unknown.

### Resumability, idempotency and prompt versions

- **Checkpoints.** Every step execution is a row in `article_steps`. It holds a fingerprint
  of the step's inputs (the upstream outputs, the step's prompt version, the model and the
  relevant settings), plus its output, tokens and status.
- **When a step runs.** A run executes a step only if no succeeded execution matches its
  fingerprint.
  - A failure or crash after research resumes at the outline.
  - A failed draft is retried with the same research and outline.
  - Running a finished article again makes no Gemini call.
- **Crashes.** A run whose process died is detected (its lock is free) and marked
  interrupted, and the article resumes.
- **Prompt versions.** Each prompt has its own version (`article-research/1`,
  `article-outline/1`, `article-draft/1`, `article-edit/1`; the brief builder is
  `article-brief/1`).
  - A new editorial prompt version re-runs only the edit.
  - A new outline version re-runs the outline, then the draft and edit only if the outline
    actually changed.
- **Versions.** Outline, draft and edited versions are immutable. Re-running adds a version,
  so "what did the model write before the latest edit?" is always answerable.
- **Slugs.** Deterministic from the title (ASCII, collision-safe). A slug is not a public URL.

### Token budget

- **Per article.** `ARTICLE_MAX_TOKENS` limits the whole article, across every step and
  resume, within the daily budget.
- **Research.** Research has its own cap, `ARTICLE_RESEARCH_MAX_TOKENS`.
- **Checking.** Budgets are checked before every call.
- **When it runs out.** The step fails safely: finished steps are kept, the article is
  `failed`, and resuming requires raising the budget.
- **Recording.** Every call is recorded in `llm_calls` (purposes `article_research`,
  `article_outline`, `article_draft`, `article_edit`) with its prompt version and tokens.

In a live run, a 1,900-word article with 3 sources and 28 citations used 6 calls and about
47k tokens.

### Completion checks

Before an article is `completed`, deterministic checks require:

- a non-empty title and content;
- at least two sections, including a body section, each with a heading;
- no empty blocks;
- at least `ARTICLE_MIN_WORDS` words;
- every citation matching a stored source;
- the opportunity and company profile version still existing.

These checks are a baseline, not a quality score (that's Phase 6). An edit that fails them
fails the step, and a resume retries it.

### Limitations

- **Sources.** Research depends on what Google Search and URL context return. Paywalled,
  script-rendered or blocked pages can't be read. With fewer usable sources than the minimum,
  research fails rather than writing without evidence.
- **Fact-checking is Phase 6.** Citations show which source a claim relies on; whether the
  source supports it is checked by [Article validation](#article-validation). The Phase 5
  number check only sees digits.
- **Titles.** The URL tool doesn't report page titles, so a source's title is as Gemini read
  it from the page.
- **Query cap.** The cap is enforced on research questions and in the prompt, while Gemini
  decides the exact queries. The queries it ran are recorded.
- **One language.** Articles are written in English.

## Importing an article you wrote

Not every post has to be written by Gemini. `articles import` takes one Markdown file you
wrote yourself and turns it into a `ready` article that goes through **the same** publishing
machinery as a generated one: the site's MDX contract, the cover image, the pull request, the
merge, the live check, the publication record and its idempotency. It needs no Gemini key and
makes no LLM call. It publishes nothing: `articles approve` and `articles publish` still do.

### The file

YAML frontmatter, then the body in Markdown:

```markdown
---
title: How small support teams hand work to AI agents
description: A practical guide to splitting customer support work between an AI agent and the people who own the hard cases.
primary_keyword: AI agents for customer support
target_audience: founders
content_type: guide            # guide | listicle | comparison | research | tutorial | article
tags:                          # 3 to 6; lowercased when published
  - ai agents
  - customer support
  - handoff
slug: ai-agents-customer-support-handoff   # optional: derived from the title otherwise
sources:                       # one per [S…] marker the body uses; at least one
  - label: S1
    title: Zendesk CX Trends
    url: https://example.com/cx-trends
  - label: S2
    title: Intercom support benchmark
    url: https://example.org/benchmark
---

Support teams of three or four people carry the same ticket volume as teams twice their
size, and the difference is almost always where the work is split. An AI agent that answers
the repetitive half of the queue buys a small team the hours it needs [S1].

## What an agent should answer first

Start with the tickets that already have one correct answer: order status, password resets,
plan changes [S2].

- Order and delivery status
- Password and login problems

### The rule of the single source

An agent is only as good as the page it reads from. Keep one canonical help centre article
per answer and let the agent quote it.

## Designing the handoff

The handoff is the part teams get wrong, so the agent hands over the whole conversation, its
own confidence and the article it used [S1].

1. The agent says it is handing over, by name
2. The transcript lands in the ticket
```

- **Structure.** Paragraphs before the first `##` are the introduction (the site renders the
  title, so the body has no H1); every `##` opens a section; `###` is a subheading; `-`/`*`
  and `1.` are lists; a wrapped line continues its paragraph or list item.
- **Citations.** `[S1]` marks the source of the claim it follows, exactly as the writing step
  marks one. Each label must be defined in `sources`, and each listed source must be cited.
  When the post is published, a marker becomes an inline link to that page.
- **Prose, not markup.** The body is escaped before it reaches the site, so a file containing
  a tag (`<BlogCTA />`), a brace (`{…}`) or an `import`/`export` line is refused rather than
  published as literal text. Inline Markdown (`**bold**`, links) is not interpreted either.
- **It is parsed into the same `ArticleContent`** the writing step produces, so everything
  downstream — rendering, the MDX file, the cover, the pull request — is unchanged.

### What the import creates

In one transaction:

- an **opportunity of its own**, key `manual:<label key of the title>`, approved, scored by
  strategic fit to your company profile, with the profile as its evidence and a `created`
  event by the actor `import` — or the opportunity you name with `--opportunity <id>`
  (`new` and `reviewed` ones are approved, since a person decided to write the piece);
- the **article** (origin `imported`, status `ready`) with its deterministic brief, one
  **final version** (recommended) with its sources and claim → source citations, and the
  **step rows** that say where it came from: prompt version `import/1`, no LLM call, no token
  (there are no outline and draft steps, because those steps never happened);
- an **authored quality report**.

### The authored quality report

The checks this project can run without Gemini are run and recorded as gates that pass:
`content_valid` (`ARTICLE_MIN_WORDS`, sections, headings, no empty block), `citation_integrity`
(every `[S…]` resolves, at least one source), `mdx_safe` (the file is composed exactly as
publishing would compose it and checked with the site's own safety check), and `seo_fields`
(keyword, meta title, meta description and slug, from the frontmatter).

The gates that need Gemini — `no_contradicted_claims`, `unsupported_claims`, `uncited_claims`,
`originality` and the `minimum_score` they feed — are recorded as **not run**, with the reason
*"written by a person, not fact-checked by the agent"*. There is no combined score, and none is
invented: `articles quality` prints "written by a person: no agent score".

**It is not a loophole.** An authored report is valid only on an imported article, and
`app/services/approval_rules.py` says so as a first-class rule, checked on every approval and
again immediately before the CMS is called:

- a report marked `authored` on an article whose origin is `generated` blocks approval and
  publication ("it must pass its own gates");
- a gate recorded as *not run* in a report that is **not** marked `authored` blocks them too,
  so the marker alone can never buy a generated article a pass.

The agent also never rewrites an imported article: `articles resume` and `articles generate
--regenerate` refuse it and point back at the file.

### Provenance

Nothing claims a fact-check that never happened:

- `articles show` and `articles import` say the article was written by a person and what was
  and was not checked; `articles quality` marks the report as authored and prints each Gemini
  gate as *not run*; `articles versions` shows the version's prompt version `import/1` and
  "by hand" where a model would be;
- `articles preflight` adds a non-blocking `provenance` check saying the same;
- the publication record carries `details.authored`;
- the **pull request** says: *"Written by hand and imported into the competitor-analysis
  agent, which checked its length, its citations, its structure and this file against the
  site's contract… It was **not** fact-checked, originality-checked or scored by the agent's
  Gemini quality gates"*, and its quality line reads "none (written by a person, not scored by
  the agent)".

### Idempotency and limits

- Importing the same file twice creates nothing the second time: the title's opportunity
  already has a live article, which the command reports instead (versions are never
  overwritten). Edit the piece, cancel the article, and import again for a new attempt.
- The file is refused, with every problem named at once, when it is shorter than
  `ARTICLE_MIN_WORDS`, cites a source it doesn't define, lists a source it never cites, has
  no usable slug, or contains anything MDX would read as code.
- An imported article can still be validated with Gemini (`articles validate`) once credits
  allow: that produces an ordinary report which replaces the authored one.

## Article validation

Phase 6 turns a completed draft into a checked, scored article that a person can publish.
**Phase 6 validates and prepares articles but does not publish them.** Publishing is Phase 7.

### Flow

```
completed article (Phase 5)
  ├─ fact_check    every cited claim vs its source; uncited factual claims    (Gemini + code)
  ├─ originality   word n-gram overlap with stored competitor/company pages   (code)
  ├─ seo           keywords, meta tags, slug, headings, FAQ, links, tags       (Gemini + code)
  ├─ metrics       structure, length, readability, citations, claims, SEO     (code)
  ├─ judge         quality rubric, 8 dimensions scored 1-5 with reasons       (Gemini)
  └─ decision      combined score, mandatory gates, issues in priority order  (code)
while the best version fails a gate and automatic revisions < QUALITY_MAX_REVISIONS:
  revise the best version (issues most serious first) → validate the revision
recommended version = the best: passing first, then the highest score, then the earliest
→ ready (every gate passes) or needs_review
```

Only `completed` articles (and `ready` / `needs_review` ones, to validate again) can enter.
Articles still being written (`queued`, `researching`, `outlining`, `drafting`, `editing`) and
cancelled ones are refused. While Phase 6 runs, the article is `validating` or `revising`.
`current_step`, the run, the score, the revision count and the recommended version are exposed
by the API and CLI. A new Phase 5 edit (for example after an edit-prompt change) clears the
validation: the article is `completed` again and must be revalidated.

### Fact-checking

Claims come from `article_citations` (claim → source). For each (claim, cited source) pair:

1. **Stored notes.** Gemini checks the claim against the source's stored research: the facts
   and verbatim excerpts read in Phase 5. The prompt (`fact-check/1`) treats the source as
   untrusted data and asks for a verdict from the source text only.
2. **Evidence check.** A `supported`, `partial` or `contradicted` verdict needs an evidence
   quote. Code checks that the quote is in the stored notes (word for word, or at least 80% as
   one contiguous run, allowing for punctuation). Without a verified quote the verdict doesn't
   count, and the claim is unsettled.
3. **Re-reading.** Unsettled claims are checked against the page itself with Gemini's URL
   context tool, for at most `FACT_CHECK_MAX_REREADS` sources per version. The URL must pass
   the Phase 1 SSRF guard first (public addresses only; no localhost, private, link-local,
   `javascript:` or `file:` URLs), and the tool must report the page as retrieved. This process
   fetches nothing itself, and URLs from the article text are never fetched.
4. **Fallback.** Whatever is still unsettled is `unsupported`. Agreement alone never verifies
   a claim.

Verdicts are `supported`, `partial` (the source supports part of it, or less than it says),
`unsupported` and `contradicted`. A claim citing several sources takes its best verdict. Each
check is stored in `article_claim_checks`: claim, source, verdict, explanation, evidence and
whether it was verified, confidence, re-read or not, model, prompt version and time. Rows are
never overwritten. An identical (claim, source) pair keeps its earlier verdict (same prompt
and model), so a revision only re-checks the sentences it changed. Only verdicts a model
actually decided are reused.

**Uncited claims.** Sentences without a citation that look factual are extracted in code:
numbers, percentages, dates, quantities, research references ("a study found"), legal
statements and named organizations. Gemini then classifies which of them need a source.
Those are stored as `needs_verification`, and the rest as `not_required` (advice, common
knowledge). Nothing is deleted automatically: they become issues for the revision.

**Computed from the verdicts, never by the model:** `citation_coverage` (cited claims ÷
cited + uncited claims needing a source), the supported, partial, unsupported and
contradicted claim ratios, the uncited factual claim ratio, and citation integrity (markers in
the text, claim → source records and stored sources must agree).

### Originality

Originality is a **similarity signal**, not a plagiarism verdict. It is deterministic, with no
LLM:

- **Shingles.** Text is normalized (citation markers removed, Unicode folded, lowercased) and
  split into runs of `ORIGINALITY_NGRAM_SIZE` words (default 8), hashed stably.
- **Common phrases are ignored.** A run found on `ORIGINALITY_COMMON_DOC_FREQUENCY` or more
  stored pages (boilerplate, stock phrases, terminology) doesn't count. Neither does a run
  that is 75% or more stopwords.
- **Per passage.** Each paragraph or list item of at least `ORIGINALITY_MIN_PASSAGE_WORDS`
  words is compared with every stored page. Its similarity is the share of its runs found in
  the best-matching page (containment). Each flag stores the passage, the page (competitor
  slug or "company", URL, content item), the longest shared text and the similarity.
- **Thresholds.** A passage is flagged at `ORIGINALITY_FLAG_THRESHOLD` (0.25). At
  `ORIGINALITY_MAX_OVERLAP` (0.5) the overlap is severe and the article can't be `ready`. The
  0-1 score goes from 1 (at or below the flag threshold) to 0 (at the severe level).

The corpus is the current version of every active page of the monitored competitors. Your
own site is compared too when it's monitored like a competitor: its pages are recognized by
the company profile's website domain, labelled "company", and offered as internal links.

### SEO package

- **Keywords.** Candidates are derived from stored data, with their provenance: the
  opportunity topic (weight 4), competitor subtopics (2), keywords of the competitor pages
  behind the opportunity (1, plus a capped bonus for pages sharing it), and company topics (1).
  Prominence in the article adds to that. Gemini picks the primary and secondary keywords
  among them and explains the choice. Code rejects a keyword that isn't a candidate or a
  rewording of candidate and heading words, and falls back to the top candidate. The evidence
  stored with the primary keyword lists the inputs it rests on.
- **Meta tags.** Meta title (at most `SEO_TITLE_MAX_CHARS`) and meta description
  (`SEO_DESCRIPTION_MIN_CHARS`-`SEO_DESCRIPTION_MAX_CHARS`). The final slug is built from the
  primary keyword, and the article's slug is updated (made unique, like Phase 5's).
- **Headings.** H1 (the title), H2 (section headings) and H3 (subheadings), with hierarchy
  problems and duplicates.
- **FAQ.** 3-6 questions answered only with what the article says. An answer with a number
  the article doesn't contain is dropped.
- **Links.** Internal links only to stored pages of your own site, and external links only
  to the article's stored research sources (never competitor or company pages). Gemini
  chooses by id among what it's offered, so a URL can never come from the model.
- **Category, tags, image.** The category is one of the topic options. Tags must relate to
  the candidates. The image suggestion is a concept, purpose and alt text (at most 125
  characters); no image is generated.
- **Checks.** Keyword in the meta title, H1, introduction, an H2 and the slug (content words,
  any order or inflection); meta lengths; hierarchy; duplicates; at least two H2s; FAQ; external
  links; and keyword density at most `SEO_MAX_KEYWORD_DENSITY`. Repetition is flagged as
  stuffing, never rewarded. The SEO score is the share of checks passed.

### Metrics and readability

Every number is computed in code:

- **Structure:** title, H1/H2/H3 counts, introduction and conclusion, hierarchy, duplicates,
  the Phase 5 completion checks, and paragraphs of at most 150 words.
- **Length:** words, sentences, paragraphs, lists.
- **Readability:** Flesch reading ease, `206.835 − 1.015 × (words ÷ sentences) − 84.6 ×
  (syllables ÷ words)`, and Flesch-Kincaid grade, `0.39 × (words ÷ sentences) + 11.8 ×
  (syllables ÷ words) − 15.59`. Syllables are counted with a vowel-group heuristic. The 0-1
  value maps reading ease 20 → 0 and 60 → 1.
- **Citations, claims, originality and SEO,** as above.

### The Gemini judge

The judge (`quality-judge/1`) scores eight dimensions from 1 to 5, each with an explanation and
specific issues: factual_support, audience_value, clarity, structure, originality,
search_intent_alignment, strategic_alignment and readability. It is given the article, the
brief, the research and the computed fact-check, originality and metrics results, which it
must treat as authoritative. It must not invent evidence, and it gives no overall score. A
dimension it leaves out counts as 1. Its 0-1 value is the mean of `(score − 1) ÷ 4`, computed
in code.

### Score and gates

```
score = Σ weight × value        weights (QUALITY_WEIGHTS) rescaled to total 100
```

| Component | Default weight | Value (0-1) |
|---|---|---|
| fact_support | 20 | (supported + ½ partial) ÷ cited claims |
| citation_coverage | 20 | cited ÷ (cited + uncited claims needing a source) |
| originality | 20 | the originality score |
| structure | 10 | share of structure checks passed |
| readability | 10 | Flesch reading ease, 20 → 0 and 60 → 1 |
| seo | 10 | share of SEO checks passed |
| gemini_judgment | 10 | the rubric mean, 1 → 0 and 5 → 1 |

The breakdown (weight, value and points per component) is stored with each report. An article
is `ready` only if **every** gate passes; otherwise it is `needs_review`:

| Gate | Fails when |
|---|---|
| content_valid | the Phase 5 completion checks fail (malformed content) |
| citation_integrity | a citation doesn't resolve to a stored source, or text and records disagree |
| no_contradicted_claims | more than `QUALITY_MAX_CONTRADICTED` (0) contradicted claims |
| unsupported_claims | more than `QUALITY_MAX_UNSUPPORTED_RATIO` (10%) of cited claims unsupported |
| uncited_claims | more than `QUALITY_MAX_UNCITED_CLAIMS` (3) factual claims without a citation |
| originality | a passage at or above `ORIGINALITY_MAX_OVERLAP` similarity |
| seo_fields | primary keyword, meta title, meta description or slug missing |
| minimum_score | score below `QUALITY_MIN_SCORE` (70) |

A gate can also be recorded as **not run**, which only an *authored* report — the deterministic
checks of an article a person wrote and imported — may do; see
[Importing an article you wrote](#importing-an-article-you-wrote).

### Revisions and the recommended version

- **Issues first.** Issues are listed most serious first: contradicted claims, unsupported
  (then partially supported) claims, uncited factual claims, citation problems, overlap with
  competitor pages, structure, search intent (and the judge's audience and strategy points),
  SEO, then readability and style.
- **What the revision gets.** The revision prompt (`article-revision/2`) receives them with
  the source evidence, as fenced data, along with the brief, the outline, the research and the
  current length. It may not add facts, numbers or sources.
- **A new version.** Each revision is an immutable `article_versions` row of kind `revision`,
  with its parent, reason, the issue ids it says it addressed (only real ids are kept), its
  changes, model, prompt version, tokens and time. It then goes through the same checks as a
  draft, and a revision that fails them (too short, broken structure) is rejected.
- **Validated again.** Each revision is validated in full. Unchanged claims keep their
  verdicts.
- **The best, not the latest.** The recommended version passes the gates first, then has the
  highest score, then is the earliest. A worse revision is kept but never recommended, and
  the article's content (API, CLI) is the recommended version.
- **Bounded.** At most `QUALITY_MAX_REVISIONS` (2) automatic revisions are made from one
  edited version, counting earlier validations'. An unusable revision, or one identical to a
  version already validated, counts as an attempt. A second attempt on the same version is a
  new attempt, not a replay. `articles revise` / `POST …/revise` asks for one more, optionally
  with a note.

### Checkpoints, resume and idempotency

Phase 6 reuses Phase 5's job system: the article lock (one run per article, for generation and
validation alike), `runs` (kind `article_quality`), and `article_steps`. Each step is a
checkpoint with a fingerprint of its inputs, so an unchanged step is reused:

| Step | Fingerprint |
|---|---|
| fact_check | prompt version, model and limits, version content, its citations, the research |
| originality | algorithm version and thresholds, version content, the stored corpus (page versions) |
| seo | prompt version, model and limits, version content, the SEO inputs (brief, topics, keywords, pages, sources) |
| metrics | metrics version, content, the fact-check, originality and SEO outputs |
| judge | prompt version, model, content, brief, research, the fact-check and originality outputs, the metrics except SEO |
| decision | policy (weights, thresholds), the outputs above, the version |
| revision | prompt version, writing settings, parent content, its issues, brief, research, outline, note, attempt |

Downstream steps depend on the **outputs** of upstream steps. So validating again makes no
Gemini call, and a failed step resumes where it stopped (`articles validate` again). A prompt
change re-runs only what depends on it: a new SEO prompt redoes the SEO step (and the metrics
and decision if the package changed), never the fact-check or the judge. A new judge prompt
never redoes the originality check.

### Token budget

Phase 6 calls count towards the article's `ARTICLE_MAX_TOKENS` and have their own cap,
`QUALITY_MAX_TOKENS`, across every validation and revision (also within
`LLM_MAX_TOKENS_PER_RUN` and the daily budget). Budgets are checked before each call. If the
budget runs out while validating the edited version, the article is `failed` at that step.
If it runs out during a revision, revising stops: the versions already validated decide the
outcome, and the article records why revisions stopped. Calls are recorded in `llm_calls`
(purposes `fact_check`, `claim_classification`, `seo_package`, `quality_judge`,
`article_revision`). Claims are checked in batches (`FACT_CHECK_BATCH_SIZE`), and a batch that
returns unusable output is split and retried.

In a live run, the 1,877-word Phase 5 article (28 citations) was validated in 15 calls and
about 60k tokens. The edited version failed two gates (4 unsupported claims, 6 uncited
factual claims). One revision fixed them (29 of 30 claims supported), and the score rose from
80.7 to 90.8, so it became the recommended version. Validating again reused every step.

### Safety

- All source, competitor and article text is untrusted: it is fenced in the prompts (tags
  inside it are defused), and every prompt says it is data, never instructions.
- The only URLs ever read are stored research sources, re-read through Gemini after the SSRF
  check. Link suggestions can only point at stored pages and sources.
- API keys are never logged or stored.

### Limitations

- **Fact-checking checks support, not truth.** A claim is checked against the source it
  cites, so a supported claim is only as good as its source. Re-reading depends on the page
  still being readable by Gemini's URL tool.
- **Signals, not judgments.** Originality is a similarity signal against *stored* pages only,
  not the web. Readability is a formula tuned for English.
- **Heuristics.** Keyword placement checks content words in any order. The uncited-claim
  extraction is signal-based, so a factual sentence without numbers, dates or research words
  can be missed.
- **Revisions can shorten.** A revision that removes unsupported content can shorten the
  article (1,877 → 1,298 words in the live run). It must stay above `ARTICLE_MIN_WORDS`.
- **Your own site.** Your site is compared and linked only if it is monitored like a
  competitor.

## Publishing

Phase 7 turns an approved, ready article version into a post on your website. The core
(approval, preflight, idempotency, the publication ledger, verification) is target-neutral;
an adapter knows the target. **The GitHub adapter is the publishing target** (`CMS_PROVIDER=
github`, the default): the site's own repository. The WordPress adapter was the first one
and is kept only until it is removed. **Phase 7 publishes only approved, ready article
versions.** Nothing is posted to social media.

### The flow

```
article (ready)
  → recommended version → its current quality report → a live approval of both
  → render (target-neutral: safe HTML, MDX-safe Markdown, metadata)
  → preflight (article, version, quality, approval, content, citations, links, SEO fields,
               target status, the target's config and access, the existing post, the slug,
               category, tags, idempotency)
  → idempotency check → create or update the draft → verify it → record it
  (→ make it public, only if asked and allowed → verify → mark the opportunity used)
```

### GitHub: the site's repository

The website (Next.js, deployed by Vercel) keeps its posts as `src/content/blog/<slug>.mdx`
files. The adapter ([app/cms/github/](app/cms/github/)) publishes the way a person would:

| Step | What happens |
|---|---|
| File | `<slug>.mdx` in the content directory: the site's frontmatter contract (`title`, `description`, `publishedAt` as a quoted ISO date, the configured author, `category` from a fixed mapping, 3–6 lowercase `tags`, `draft: false`), the article body (intro paragraphs, H2/H3, lists, inline source links, FAQ), a "Related reading" list of existing site pages, exactly one `<BlogCTA />` and a closing italic byline. Two keys the site ignores carry ownership: `agentPublication` (the publication's marker) and `agentSource` |
| Draft | Branch `blog/<slug>` from `main`, commit `Add blog post: <title>`, a pull request whose description carries the article's provenance (score, opportunity, sources, expected URL). The branch's **Vercel preview deployment** must succeed and serve the post before the draft counts as verified |
| Publish | Only with `PUBLISH_ALLOW_DIRECT_PUBLISH=true`: the pull request is **squash-merged**, the production deployment of the merge commit must succeed, and `https://<site>/blog/<slug>` must return the page (200, title, canonical URL, published metadata). Only then is the publication `published`; merged but not verified stays an unknown outcome the next run resolves |
| Category | Never the model's choice: guides, how-tos and everything else → `Playbook`; comparisons → `Comparison`; data-led research → `Industry Data`. `Product`, `Announcements` and `Research Paper` are never chosen automatically |
| Slug | The Phase 6 SEO slug under the site's rules (lowercase ASCII, `[a-z0-9-]`, no leading, trailing or double dash). The filename is the URL; a slug already used by a post that isn't the agent's blocks publishing |
| Internal links | Only to pages in the site's sitemap (read at preflight) or its blog directory; others become plain text. Source citations are ordinary Markdown links to the stored sources |
| Safety | Prose is escaped so it can never become JSX, an `import`/`export` line, a Markdown block or an autolink; the whole file is validated (frontmatter round-trip, one CTA, no stray brace or tag) before it is written. The site's own build is the last gate |
| Idempotency | The branch, the file, the pull request and the merge are each looked up before they are made. A lost answer, a crash after the branch, after the commit or after the pull request, a lost pull request number or a merge that timed out all reconcile against `main`, the `blog/*` branches, open pull requests and the marker in the frontmatter |
| Protection | If Vercel's Deployment Protection blocks the preview, the run stops with a clear error and nothing is merged: nothing is bypassed |
| Token | `GITHUB_TOKEN`, a fine-grained personal access token for that one repository with *Contents* and *Pull requests* read/write, sent to `api.github.com` only. The site and its previews are fetched without credentials |
| Cover image | With `PUBLISH_COVER_IMAGES=true`: one picture per post — a Gemini illustration or a Pexels stock photo (`COVER_IMAGE_SOURCE`) — committed to the same branch as `public/blog/covers/<slug>.<ext>` and named in the frontmatter (`coverImage`, plus `coverWidth`/`coverHeight`). See [Cover images](#cover-images) |

`articles publish <id> --dry-run` shows the exact MDX file, branch and pull request title
without touching GitHub.

### The blog index gate

Before a post is merged, the agent loads the **blog index** on the pull request's preview
deployment, not just the post's own page. The merge is refused, the pull request stays open
and the run fails with the reason when the index:

- doesn't list the new post;
- no longer lists a post that is live today (a count would miss a build that loses one post,
  because the new one makes up the number);
- hides its post cards: an inline `opacity:0`, `display:none` or `visibility:hidden` on the
  element `PUBLISH_INDEX_CONTAINER_ID` (default `blog-posts`).

The last check exists because of a real incident (2026-09-24): the site's post grid faded in
only once 15% of it was on screen, which stops happening past a few dozen posts. Every card
stayed at `opacity:0` and the blog looked empty to visitors while the agent kept publishing.
With the gate, a regression like that stops publishing and says why, instead of piling posts
onto a page nobody can see. `PUBLISH_INDEX_PATH` (default `/blog`) names the index.

### Cover images

Off by default (`PUBLISH_COVER_IMAGES=false`): with it off, nothing here runs and a post is
published exactly as it was before. With it on, each published post gets one cover picture,
from one of two sources (`COVER_IMAGE_SOURCE`).

| | Common to both sources |
|---|---|
| Where it goes | `public/blog/covers/<slug>.<ext>` on the post's own branch, committed before the `.mdx` file, so the pull request and its preview deployment show the post with its cover |
| Frontmatter | `coverImage: '/blog/covers/<slug>.<ext>'` between `tags` and `draft`, with `coverWidth` / `coverHeight` as unquoted numbers read from the picture's own header (the site renders the detail header at the picture's natural aspect ratio; cards crop to 1200×630) |
| File types | `png`, `jpeg` or `webp` only, between 100 bytes and 8 MB. Anything else is refused and never committed |
| Obtained once | One `article_covers` row per (article, version), unique in the database. A retry after a lost answer, a second publication of the same version and a follow-up pull request all reuse it: the image model is called, or Pexels searched, exactly once per version |
| Committed once | The adapter looks the file up on the branch first, like the branch, the post and the pull request, and asks for the bytes only when it is missing |
| Provenance | Each row records its source (`gemini` \| `pexels`), and for a photo its id, page, photographer and their page. The publication's details carry the same, and the pull request repeats the credit line |
| Failure | Never fatal. No key, a budget stop, a refusal, an unusable answer or no usable photo is a warning on the publication, and the post is published without a cover |
| Dry runs | Make nothing: `--dry-run` shows the file as it stands, without a cover |

**`COVER_IMAGE_SOURCE=gemini` (the default): a generated illustration**

| | |
|---|---|
| What | One editorial illustration per article version, drawn by `GEMINI_IMAGE_MODEL` from the article's title, primary keyword, audience and your company profile's positioning and tone |
| The prompt | [app/prompts/cover_image.py](app/prompts/cover_image.py), versioned like every other prompt. Its hard rules: **no text, letters, numbers, logos or watermarks** (models render text badly, and the site prints the title over the cover anyway), no identifiable people, nothing that could pass as a real photo of a real clinic, patient or medical record, nothing clinical and nothing that claims a result. The article's own words are **untrusted data** inside a delimited block, never instructions |
| Alt text | Deterministic, from the article's keyword — it describes the illustration and claims nothing |
| Cost | One image call per published version, in the same `llm_calls` ledger (purpose `cover_image`) and the same `LLM_DAILY_TOKEN_BUDGET` as every text call |

**`COVER_IMAGE_SOURCE=pexels`: a stock photo** ([app/images/pexels.py](app/images/pexels.py))

| | |
|---|---|
| What | One [Pexels](https://www.pexels.com/) photo per article version. **No model is called**: `PEXELS_API_KEY` is the only new requirement, and free |
| Choosing it | 2–4 searches built by fixed rules from the article's primary keyword, who it is for and the brief's content format, tried in order: keyword + audience, keyword alone, keyword + the subject its format suggests, then a plain business scene. The article's words become *search words* (ASCII, lowercased, stop-words dropped) and are sent as a query parameter — never as an instruction to anything |
| Which photo | `orientation=landscape`, at least 1200px wide, wider than it is tall, and of those the one whose aspect ratio is closest to 16:9 (ties go to the lower photo id, so the choice is deterministic). The **sized variant** (`large2x`, else `landscape`, else `large`) is downloaded, never the multi-megapixel original |
| Never twice | The photo's id is stored, and any photo an earlier cover used is skipped (an index on `article_covers (source, source_id)`). A search with no usable photo falls through to the next query; if none of them has one, the post is published without a cover |
| Two hosts | The key goes to `api.pexels.com` in the `Authorization` header and nowhere else; the picture is fetched from `images.pexels.com` with **no credential**. Redirects are never followed, and a link to any other host is refused before a request is made. Timeouts and bounded retries follow `CMS_REQUEST_TIMEOUT` / `CMS_MAX_RETRIES` |
| Alt text | Deterministic, from the article's keyword, and says the picture is a stock photograph |
| Attribution | The Pexels licence doesn't require it. The photographer, their page and the photo's page are recorded on the row and in the publication's details, and the pull request gets a `Cover photo: <photographer> on Pexels — <url>` line. **Nothing is written into the frontmatter**: the site's `BlogFrontmatter` (`src/lib/blog.ts`) has no credit field, so a visible credit on the page would need a new field there *and* in the post template — a change to the website repository, which this agent never makes |

The picture's bytes live in `article_covers` only. They never enter a rendered document, a
CMS payload, a publication's details, a run summary or a log line — the publication records
the cover's name, type, size, alt text, SHA-256 and provenance. `PEXELS_API_KEY` is held as
a secret: it is never logged, never put in an error message, never stored and never part of
a payload.

The API queues publishing in the background (`202`, `{"publication_id": ..., "status":
"queued"}`); the CLI runs it straight away. One run per article at a time: publishing shares
the article's lock with generation and validation, and a second request returns the
publication already in progress.

### Approval

- **Explicit.** A person approves (`articles approve` / `POST …/approve`) or rejects (with a
  reason) an article that is `ready`. `needs_review` must be resolved in Phase 6 first: there
  is no manual override.
- **Exact.** A decision records the article, **the recommended version, the quality report**,
  the approver, the method (`manual` or `auto`), the channel (API, CLI, policy), the time and
  a note. Decisions are never deleted or edited.
- **Invalidated, never reused.** An approval authorizes publication only while the article's
  recommended version and current quality report are the ones it names. Each of these voids
  it, with the reason recorded:
  - a new recommended version (a better revision);
  - a new quality report for the same version (for example, new weights);
  - a new Phase 5 edit;
  - cancellation;
  - a later decision.

  Validating again with nothing changed keeps the same report, so the approval stands.
- **States.** `not_ready`, `pending` (ready, no decision yet), `approved`, `rejected`,
  `invalidated`. At most one decision per article is live, enforced by the database.
- **Auto-approval** (`PUBLISH_AUTO_APPROVE`, off by default). A publish request for a ready
  article with no decision records an automatic approval (method `auto`). It never overrides
  a rejection, and it still needs an explicit publish request: nothing publishes by itself.
- **Authored reports.** An article written by a person and imported carries a report whose
  Gemini gates are recorded as not run; it is approvable and publishable, and only ever on an
  imported article (see [Importing an article you wrote](#importing-an-article-you-wrote)).
  The pull request and the publication record say the piece wasn't machine-fact-checked.

### WordPress setup (the earlier adapter, `CMS_PROVIDER=wordpress`)

1. Use a WordPress site served over **HTTPS**. Application Passwords need HTTPS (http is
   accepted here only for `localhost`).
2. Create a user for publishing, or use your own: an **Author** can create drafts and
   publish their own posts, an **Editor** can also create categories and tags. Drafts need
   `edit_posts`; making posts public needs `publish_posts`.
3. In WordPress, go to **Users → Profile → Application Passwords**. Name one (e.g.
   "competitor agent"), click **Add New Application Password**, and copy the password shown
   (it is shown once).
4. Put it in `.env`, never in code or in git:

   ```env
   WORDPRESS_BASE_URL=https://blog.example.com
   WORDPRESS_USERNAME=your-user
   WORDPRESS_APPLICATION_PASSWORD=abcd efgh ijkl mnop qrst uvwx
   ```

5. Check it: `uv run python -m app articles preflight <id>` shows whether the site is
   reachable, who you are signed in as, and what that user can do.

The REST API only is used. Nothing scrapes WordPress or drives a browser. Your normal
WordPress password is never used or stored, and credentials are sent only to
`WORDPRESS_BASE_URL`, over HTTP Basic auth with the Application Password (redirects are
never followed).

### Drafts and direct publishing

- **Draft by default.** `PUBLISH_DEFAULT_STATUS=draft`. `pending` leaves the post "Pending
  Review" in WordPress.
- **Making a post public.** Use `--status publish` / `{"status": "publish"}`, which needs
  **`PUBLISH_ALLOW_DIRECT_PUBLISH=true`**. With `PUBLISH_DRAFT_FIRST=true` (default) the
  draft is written and verified first, then made public and verified again.
- **Fully automatic publication** needs both switches, `PUBLISH_AUTO_APPROVE=true` and
  `PUBLISH_ALLOW_DIRECT_PUBLISH=true` (plus `PUBLISH_DEFAULT_STATUS=publish`), and still
  an explicit publish request. Phase 7 has no scheduler.
- **A public post is never taken back to draft.** Updating a public post needs the publish
  target.

### Preflight and dry run

- **Preflight** (`articles preflight` / `POST …/preflight`) runs every check and prints
  `READY` or `BLOCKED` with a line per check. It reads WordPress through a client that
  refuses any change.
- **Dry run** (`articles publish --dry-run` / `?dry_run=true`) adds the rendered HTML and the
  exact WordPress request body, including the ownership marker, term ids, slug, status and
  excerpt. It never includes credentials.

Neither changes anything, in WordPress or in the database.

### What is published, and how

- **What.** Only `article.recommended_version_id`, rendered from its stored content, with its
  approval valid at the moment WordPress is called (checked again right before the change).
  Never an older or rejected version, and never content from a request.
- **Rendering** (`app/services/article_render.py`, CMS-neutral). The structured article
  becomes HTML built here tag by tag:
  - every piece of text is escaped, so a model can't inject markup or scripts;
  - H2 for sections, H3 for subheadings, paragraphs, ordered and unordered lists;
  - citations: `[S2]` becomes `[1]`, linking to a **Sources** section that lists only the
    sources this version cites, with their stored URLs (no internal ids are published);
  - the FAQ from the SEO package.
- **Links.** Only Phase 6's validated links, and only to allowed targets:
  - internal links go to stored pages of your site;
  - external links go to the article's stored research sources;
  - http(s) only;
  - each is placed on the first occurrence of its anchor text, and internal links that don't
    fit are listed under "Related reading";
  - no URL is ever added while publishing.
- **SEO mapping.**

  | Article / SEO package | WordPress |
  |---|---|
  | Title | Post title |
  | SEO slug | Slug |
  | Meta description | Excerpt |
  | Category | Category id, looked up by name |
  | Tags | Tag ids, looked up by name, without duplicates |
  | `WORDPRESS_DEFAULT_AUTHOR_ID` | Author |

  No SEO plugin is assumed. The meta title and description are kept in the publication
  record and shown by `articles publication`.
- **Categories.** A missing category **blocks** publishing, rather than filing the post in
  the wrong one, unless `WORDPRESS_CREATE_MISSING_TERMS=true` (then it's created).
  `WORDPRESS_DEFAULT_CATEGORY_ID` applies only when the SEO package has no category.
- **Tags.** Missing tags are left out, with a warning, unless creation is allowed.
- **Images.** Phase 6 suggests an image concept and alt text but doesn't generate images, so
  posts go out without a featured image. The suggestion is kept in the publication record,
  and no image URL is invented.
- **Slug.** A slug used by a post this system didn't create **blocks** publishing: it is
  never silently changed. A public post keeps its existing slug when a new version updates
  it.

### Idempotency and reconciliation

- **One publication per version.** Each (article, version, CMS, site) has a single
  publication row, keyed by a deterministic idempotency key.
- **The post id is stored at once.** The WordPress post id is saved as soon as it is known,
  so no later run creates another post.
- **Ownership marker.** Every post carries an opaque marker (an HTML comment with a random
  id, not a database id). A post without it is someone else's and is never updated.
- **Lost responses.** If WordPress saves a post but the response is lost (a timeout), the
  creation is **not retried blindly**:
  1. the post is looked up by slug, then by marker;
  2. if it's there, it is adopted;
  3. only if it isn't is the creation retried.

  If the lookup itself fails, the run stops and the next run reconciles first.
- **Publishing the same version again** checks the post and changes nothing if it already
  holds this version (`action: none`).
- **Attempts are recorded.** Every change attempt (`create`, `update`, `publish`,
  `reconcile`, `terms`) is logged with its outcome: `succeeded`, `failed`, or `unknown` (then
  reconciled).

### Version safety

A new recommended version is never published automatically. If version 4 is published and
a revision makes version 5 recommended:

1. the approval of version 4 no longer applies;
2. WordPress keeps version 4;
3. version 5 needs its own approval and an explicit publish.

Version 5 then **updates the same WordPress post**, through a new publication row. That
requires the article's earlier publication record on that site, the post to still be this
article's (its marker), and the new approval. The version 4 publication stays as history
(`superseded_by`).

### Publication lifecycle and the opportunity

- **Statuses.**

  | Status | Meaning |
  |---|---|
  | `queued` | Accepted; waiting for its run |
  | `preflight` | Checks running |
  | `blocked` | A check failed; nothing was sent |
  | `submitting` | A change is in flight |
  | `draft_created` | WordPress holds this version as a draft (or pending review) |
  | `published` | This version is public |
  | `failed` | The attempt failed; the post's own state is in `external_status` |
  | `cancelled` | Stopped before completion |

- **Recorded on success.** The CMS, the post id, the public URL, the published time and
  what was mapped.
- **The opportunity** becomes `used` only after a **confirmed** public publication. It stays
  as it was while a publication is queued, preflighted, only a draft, or after a failure.

### Failures

- **Retried (transient):**
  - timeouts, network errors, rate limits (`429`, honoring `Retry-After`) and server
    errors: bounded retries with backoff (`CMS_REQUEST_TIMEOUT`, `CMS_MAX_RETRIES`);
  - reads and updates are retried directly; a creation is retried only after
    reconciliation.
- **Never retried (fix the cause, then publish again):**
  - wrong credentials (`401`), missing permissions (`403`) and invalid requests
    (`400`/`422`);
  - slug collisions and missing categories.
- **What's kept.** Errors are saved on the publication and its attempts, without
  credentials. A CMS that can't be reached during preflight fails the run without changing
  anything.

### Security

- **Credentials come only from settings.** They never appear in logs, the database, API
  responses, error messages or test output: the password is a secret setting, HTTP headers
  and bodies are never logged, and errors carry only the method, path, status and
  WordPress's own error code and message.
- **The site URL** must be https (http only for localhost) and may not contain credentials.
- **A dry-run client refuses every change** before it is sent.

### Limitations

- **SEO plugins.** Meta title and description aren't written to Yoast or Rank Math: no
  plugin is assumed. The description goes to the excerpt.
- **Images.** No featured image (none is generated).
- **Removing the marker.** A post whose marker was removed in WordPress is treated as
  someone else's and won't be updated. Deleted or trashed posts must be restored in WordPress.
- **One CMS per run.** WordPress only, one site at a time (`WORDPRESS_BASE_URL`); changing
  the site starts a new publication history.
- **After publishing, the opportunity is `used`.** Phase 5 won't write to it again (use a
  Phase 6 revision for changes).

## Scheduling and the autonomous pipeline

Phase 8 introduces autonomous scheduling and pipeline orchestration. Social media automation is
intentionally deferred to Phase 9.

### How it runs

```
APScheduler (worker process) → JobService (queue, locks, heartbeat, retries)
  → PipelineService (the stages) → the Phase 2–7 services
```

The scheduler holds no business logic. Each stage calls the same service a person would:

| Stage | Service | Notes |
|---|---|---|
| `scan` | Phase 2 scans | Every active competitor; one failing is a warning |
| `analyze` | Phase 3 analysis | Incremental: already analyzed pages cost nothing |
| `opportunities` | Phase 4 scoring | Zero opportunities is a normal result |
| `editorial` | Editorial topics | Tops up the editorial ideas for today's editorial allowance; off by default |
| `generate` | Phase 5 articles | The top opportunities within today's allowances (one per origin) |
| `quality` | Phase 6 validation | With its revision loop; `needs_review` is a decision, not an error |
| `approval` | Phase 7 approvals | Reports where each ready article stands; approves nothing |
| `publish` | Phase 7 publishing | Only through `PublishingService`; the daily slot is reserved atomically |

Job types: `scan`, `analyze`, `opportunities`, `editorial`, `generate_articles`,
`quality_check`, `publish` (approval + publish) and `full_pipeline`.

### Turning it on

Nothing runs by default. A cautious setup that writes drafts every morning for a person to
review:

```bash
SCHEDULER_ENABLED=true
FULL_PIPELINE_SCHEDULE=0 6 * * *     # 06:00 every day, in SCHEDULER_TIMEZONE
AUTOMATED_PUBLISHING_ENABLED=true    # the pipeline may send articles to WordPress...
# PUBLISH_ALLOW_DIRECT_PUBLISH=false (default): ...as drafts only
# PUBLISH_AUTO_APPROVE=false (default): only articles a person approved
```

Then start the worker next to the API: `uv run python -m app worker`. Fully automatic public
posting needs all four of `AUTOMATED_PUBLISHING_ENABLED`, `PUBLISH_AUTO_APPROVE`,
`PUBLISH_ALLOW_DIRECT_PUBLISH` and a non-zero `MAX_ARTICLES_PER_DAY`. Check what a run would
do with `pipeline run --dry-run` first.

### Schedules and time zones

- **Cron.** Standard 5 fields (`minute hour day month weekday`) or `@hourly`, `@daily`,
  `@weekly`, per job: `FULL_PIPELINE_SCHEDULE`, `SCAN_SCHEDULE`, `ANALYSIS_SCHEDULE`,
  `OPPORTUNITY_SCHEDULE`, `EDITORIAL_SCHEDULE`, `ARTICLE_GENERATION_SCHEDULE`,
  `QUALITY_SCHEDULE`, `PUBLISH_SCHEDULE`. Expressions are parsed, never evaluated; a bad one stops the app at
  startup.
- **Time zone.** Schedules run in `SCHEDULER_TIMEZONE` (default `Asia/Kolkata`), DST included.
  Everything is stored in UTC.
- **One job per occurrence.** Each scheduled time has a unique key, so two workers or a restart
  can't run it twice.
- **Missed runs.** After an outage, one catch-up run for the latest missed time (within
  `SCHEDULER_CATCH_UP_HOURS`), never one per missed time. A new schedule doesn't fire
  retroactively.
- **Pause.** `schedule pause` records each scheduled time as a skipped job until `schedule
  resume`; manual runs still work. `SCHEDULER_ENABLED=false` fires nothing at all.

### Daily limits

Both limits use the calendar day in `SCHEDULER_TIMEZONE`, computed from stored timestamps (no
midnight job). `0` means none, never unlimited.

| Limit | Counts | Enforced |
|---|---|---|
| `MAX_ARTICLES_GENERATED_PER_DAY` (3) | Articles created today from competitor opportunities, by the pipeline or by hand | Before any article is created: the best N opportunities are selected under a lock. The rest stay eligible for later runs |
| `MAX_EDITORIAL_ARTICLES_PER_DAY` (0) | Articles created today from editorial topics | The same selection, under the same lock, with its own allowance: one origin never uses the other's |
| `MAX_ARTICLES_PER_DAY` (1) | Successful public posts today. Not drafts, failed, blocked or deferred attempts | Inside Phase 7's publisher, in the transaction that starts the CMS change, under a lock. Two publishers racing for the last slot publish exactly one; the other waits for a later run with nothing sent |

A publication made by hand isn't limited, but uses the day's allowance. Drafts don't use it;
a run still sends at most `MAX_ARTICLES_PER_DAY` of them.

### Selection

Never random. Opportunities are ranked by score, then evidence, then strategic fit:
approved ones, plus (with `PIPELINE_APPROVE_OPPORTUNITIES`, the default) new or reviewed ones
scoring at least `PIPELINE_MIN_OPPORTUNITY_SCORE`. The pipeline approves only the ones it
selects, recorded as actor `pipeline`. An opportunity that has an article, in any state, is
never selected again. Publishing takes ready articles of approved, unused opportunities that
were never published on the site, in the same order.

### Approval and publishing safety

- The pipeline never approves an article. Without `PUBLISH_AUTO_APPROVE` it stops before
  publishing and lists the articles awaiting approval (`articles approve <id>`).
- Only `ready` articles with a live approval of their exact version are sent, through Phase 7:
  preflight, idempotency key, reconciliation after a lost response, draft first.
- `AUTOMATED_PUBLISHING_ENABLED=false` keeps the pipeline away from WordPress entirely.
- Two consecutive publishing failures stop the stage (the CMS is probably down).

### Jobs, checkpoints and recovery

- **Jobs** (`jobs list`, `GET /api/v1/jobs`): type, status (`queued`, `running`, `completed`,
  `completed_with_warnings`, `failed`, `cancelled`, `skipped`), trigger, attempts, heartbeat,
  error, stages and the pipeline report.
- **Checkpoints.** Each stage's result is saved when it ends (`scan_complete` …
  `publishing_complete`), with per-item progress while it runs. A continued job skips what's
  done: no Gemini work is repeated, and an article is never created twice.
- **Locks.** One running job per type (a second is skipped), and at most
  `MAX_CONCURRENT_PIPELINES` jobs that can spend Gemini tokens.
- **Crashes.** A running job whose heartbeat is older than `JOB_STALE_AFTER_MINUTES`, and whose
  process is gone, is continued from its checkpoint by the worker. Ctrl-C requeues running jobs.
- **Retries.** Transient failures (network, Gemini unavailable or rate limited, WordPress
  429/5xx) are retried with exponential backoff, `JOB_MAX_ATTEMPTS` attempts in all.
  Invalid keys or configuration, rejected credentials, failed quality gates and spent token
  budgets are never retried. `jobs retry` continues a failed job from its checkpoints.
- **Budgets.** `LLM_DAILY_TOKEN_BUDGET` is checked before each Gemini stage and item; a spent
  budget ends the stage as `skipped_due_to_budget`.
- **Priority.** Recovered and retried jobs first, then scheduled ones, then manual ones.

### Planning mode

`pipeline run --dry-run` (or `POST /api/v1/pipeline/run {"dry_run": true}`) shows what would
run now: the competitors and the local analysis estimate, the ranked opportunities and which
fit today's allowance, the validations, and each publication with why it would or wouldn't be
sent. No site is fetched, **no Gemini call is made**, nothing is written to WordPress and no
allowance is used. It doesn't simulate opportunity detection: it shows the opportunities that
exist now.

### Observability

- `schedule status` (`GET /api/v1/schedule/status`): what ran today, what failed, generated /
  ready / published with the remaining allowances, the next runs, and warnings (disabled,
  paused, WordPress or Gemini not configured, failed or stuck jobs).
- Structured log events for every job and stage (`job.*`, `pipeline.stage_*`, `worker.*`,
  `publishing.deferred_daily_limit`), never with a secret.

### Limitations

- **One pipeline at a time by default.** Raise `MAX_CONCURRENT_PIPELINES` only with enough
  Gemini quota and database connections (each running job holds two).
- **The worker is required for schedules and retries.** The API and CLI run the jobs they
  start; a job requeued for a retry waits for a worker.
- **A stage's failure stops the pipeline.** If every competitor's scan fails, nothing after it
  runs that time.
- **Failed articles are not resumed automatically** after the run that created them: resume
  them with `articles resume <id>`.
- **No metrics backend.** Status comes from the database and the logs.

## Data model

PostgreSQL, managed with Alembic migrations (`migrations/`), in separate layers:

| Layer | Tables | Contents |
|---|---|---|
| Configuration | `competitors` | Who is monitored, and how (options stored as validated JSON) |
| Raw | `raw_documents` | HTML exactly as fetched (gzip), stored only for captured versions |
| Normalized | `content_items`, `content_versions`, `change_events` | One row per URL (lifecycle + reliable dates); immutable snapshots; the change log |
| Analysis (Phase 3) | `topics`, `topic_aliases`, `content_analyses`, `content_analysis_topics`, `change_summaries`, `competitor_profiles`, `landscape_reports` | Model-produced interpretations, each with its run, model and prompt version; profiles and reports stored with the metrics they were grounded on |
| Recommendations (Phase 4) | `company_profiles`, `opportunities`, `opportunity_assessments`, `opportunity_evidence`, `opportunity_events` | Versioned company profiles; one opportunity per topic with its status; immutable scored assessments (breakdown, gaps, signals, suggestion, Gemini interpretation); the evidence each assessment rests on; the status and scoring timeline |
| Generation (Phase 5) | `articles`, `article_steps`, `article_versions`, `article_sources`, `article_citations` | Article drafts linked to their opportunity, assessment and company profile version; the checkpoint log; immutable outline/draft/edited versions; retrieved research sources with their facts; claim → source citations. `articles.origin` says who wrote it (`generated`, or `imported` for a file a person wrote). Never published |
| Validation (Phase 6) | `article_claim_checks`, `article_originality_flags`, `article_quality_reports`; revision rows in `article_versions` | One verdict per (claim, cited source) and per uncited factual claim, with evidence and provenance; flagged passages with the page they overlap; each version's score, breakdown, gates and issues, linked to the steps it came from; `authored` marks the deterministic report of an imported article, whose Gemini gates are recorded as not run. The article points at its recommended version and report. Never published |
| Publishing (Phase 7) | `article_approvals`, `publications`, `publication_attempts`, `article_covers` | Decisions on an exact article version and quality report (never deleted; invalidated with a reason; one live per article); one publication per article version and CMS site (idempotency key, post id, URL, status, what was mapped, last preflight); every CMS change attempted and its outcome; one generated cover picture per article version, with the prompt, prompt version and model that made it. No credentials |
| Scheduling (Phase 8) | `jobs`, `scheduler_state`; `publications.limit_day` | One row per execution (type, trigger, status, the scheduled time and its unique key, attempts, heartbeat, error kind, stage checkpoints and progress, the report; no secrets); the persisted pause switch; the local day an automated publication reserved |
| Operations | `runs`, `run_events`, `llm_calls` | What ran, when, with what result (article runs carry `article_id`); every LLM call with its tokens |

LLM output lives in the analysis layer, in the `interpretation` of opportunity assessments and
in the generation layer. None of it modifies the layers below it, and no score depends on it.

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
| `GEMINI_ANALYSIS_MODEL` / `GEMINI_SYNTHESIS_MODEL` | No | `GEMINI_MODEL` | Per-route models: bulk per-page analysis vs. profiles, briefings, summaries, consolidation, opportunity interpretation |
| `GEMINI_IMAGE_MODEL` | No | `gemini-3.1-flash-image` | The image model, used only for blog cover images (`PUBLISH_COVER_IMAGES`) and never for text. Also available: `gemini-3-pro-image`, `gemini-2.5-flash-image` |
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

### Claude: the second writer

Articles can be written by **Anthropic Claude** instead of, or alongside, Gemini.
`app/llm/claude.py` implements the same provider protocol as `app/llm/gemini.py` with the
official [`anthropic`](https://pypi.org/project/anthropic/) SDK, and is the only module that
imports it (loaded lazily). The writing services can't tell the two apart:

- **Structured output** goes through the SDK's `messages.parse()` helper, which sends each
  prompt's Pydantic model as a `json_schema` output format and returns it validated.
- **Tools.** The project asks for `google_search` and `url_context`; on Claude those become
  the server-side **web search** and **web fetch** tools, and what they did is reported as the
  same `Grounding` — queries run, URLs requested, the final URL and status of each read, and
  the citations on the answer — so research, citations and provenance work unchanged.
- **Usage** is reported as input (uncached + cache reads + cache writes), output and thinking
  tokens, so the `llm_calls` ledger and the token budgets keep working. Each row records the
  provider that served it.
- **Errors** map onto the same neutral taxonomy. HTTP 402 (`billing_error`) becomes
  `LLMBillingError`, deliberately *not* `LLMInvalidRequestError`: an account that cannot pay
  is a different problem from a request that is malformed, and only the first is worth
  retrying on the other provider. Gemini's own HTTP 402 ("prepayment credits are depleted")
  maps there too.
- Claude **generates no images**: cover pictures stay with Gemini or Pexels.

#### Who writes which article

| `WRITING_PROVIDER` | Effect |
|---|---|
| `gemini` (default) | Gemini writes every article. Nothing changes from before. |
| `claude` | Claude writes every article. |
| `split` | Articles from **editorial topics** are written by Claude, articles from **competitor opportunities** by Gemini. |

`split` is how "10 from Claude, 4 from Gemini a day" is expressed in the system's own terms:
the two opportunity origins already have their own daily allowances
(`MAX_EDITORIAL_ARTICLES_PER_DAY` and `MAX_ARTICLES_GENERATED_PER_DAY`). Set them to 10 and 4,
with `ARTICLE_GENERATION_SCHEDULE` every two hours and `MAX_ARTICLES_PER_RUN=1`, and the day
produces 14 posts spread over the day, 10 of them Claude's.

`CLAUDE_ARTICLE_SHARE` (default 10) caps how many articles Claude writes in one
`SCHEDULER_TIMEZONE` day; past the cap the rest of the day's editorial articles go to Gemini.
`CLAUDE_ARTICLE_SHARE=0` turns Claude off without unsetting its key.

Routing is a property of the **article**, not of a call: one article is researched, outlined,
drafted and edited by one provider, so its steps stay consistent and a resume continues with
the provider whose checkpoints are on disk instead of paying for them again. The step
fingerprint includes the model, so a step Gemini produced is never reused as if Claude had
written it.

#### Fallback

If the chosen provider fails a step with an error **no retry fixes** — bad credentials,
billing/credits, a configuration problem, or a rejected request — the article falls back to
the other **configured** provider, that provider redoes the step, and the run continues with
it. Rate limits, 5xx, unusable output and budget stops are *not* fallback reasons: those are
worth retrying on the same provider.

A fallback is never silent:

- `articles.writer` is rewritten to the provider that actually produced the content, and
  `articles show` prints `written by: claude` plus a `fell back · …` line per handover;
- the run summary carries `writer` and `writer_fallbacks`, and the pipeline's job summary
  repeats both (`writers`, `writer_fallbacks`) and raises a stage warning;
- both providers' calls stay in the `llm_calls` ledger with their own `provider` and `model`,
  and the failed step row is kept next to the one that replaced it.

A provider with no API key is never chosen and never fallen back to: **with
`ANTHROPIC_API_KEY` unset, behaviour is identical to before.** The pipeline stops a stage on a
systemic LLM failure as it always did, except when another provider took over in the same run
and produced an article.

**Validation stays on Gemini.** Fact-checking, originality, SEO, the judge and revisions are
unchanged, on `GEMINI_QUALITY_MODEL`. That is deliberate: the judge is one component of the
quality score (`gemini_judgment` in `QUALITY_WEIGHTS`), the fact-check re-read step depends on
Gemini's URL-context retrieval statuses, and scores are only comparable across articles while
one model produces them. Moving validation too would re-baseline every gate and threshold.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | For writing with Claude | *(empty)* | Anthropic API key. Without it Claude is never used and nothing falls back to it. |
| `CLAUDE_MODEL` | No | `claude-sonnet-5` | Model ID for the writing steps. Use a 4.6-or-later model: the provider sends adaptive thinking and the `_20260209` web tools. |
| `WRITING_PROVIDER` | No | `gemini` | `gemini`, `claude` or `split` (see above) |
| `CLAUDE_ARTICLE_SHARE` | No | `10` | With `split`: articles Claude may write per `SCHEDULER_TIMEZONE` day |

`LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_MAX_TOKENS_PER_RUN` and
`LLM_DAILY_TOKEN_BUDGET` apply to both providers.

| Phase | Uses Gemini? |
|---|---|
| 1 · Website monitoring | **No.** Deterministic. Works with `GEMINI_API_KEY` empty. |
| 2 · Persisted history | **No.** Storage, incremental scans and change detection are deterministic. |
| 3 · Competitor analysis | **Yes:** per-page analysis, change summaries, profiles, landscape briefings, topic consolidation. Metrics, trends and gaps stay deterministic. |
| 4 · Content opportunities | **Yes, top candidates only:** title, angle, why now, format, audience, rationale. Signals, relevance, gaps, scores and ranking are deterministic; works without a key. |
| 5 · Article drafts | **Yes:** research (Google Search grounding and URL context), outline, draft, editorial pass. The brief, URL screening, citation checks and completion checks are deterministic. Nothing is published. |
| 6 · Article validation | **Yes:** claim verdicts (with URL context re-reads), uncited-claim classification, SEO wording, the quality rubric, revisions. Evidence checks, originality, metrics, the score, gates and version selection are deterministic. Nothing is published. |
| 7 · Publishing (current) | **Only the cover picture** (`PUBLISH_COVER_IMAGES`, off by default): an image model draws it, from a prompt the article can only supply data to — and not even that with `COVER_IMAGE_SOURCE=pexels`, where the photo is chosen by fixed rules. Approval, rendering, preflight, the CMS calls and reconciliation are deterministic. |
| 8–10 · Scheduling, social, dashboard | Publishing decides nothing with an LLM; only a post's cover illustration is generated |

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
| `COMPANY_FILE` | `config/company.yaml` | Company profile YAML for `company import` |
| `SCORING_FILE` | `config/scoring.yaml` | Optional opportunity scoring configuration (weights, thresholds, Gemini candidates); built-in defaults without it |
| `EDITORIAL_TOPICS_PER_RUN` | `10` | Editorial ideas kept per proposal run (Gemini is asked for about a third more) |
| `GEMINI_WRITING_MODEL` | `GEMINI_MODEL` | Model for article research, outline, draft and editing (Gemini's turn) |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic key. Without it Claude never writes and nothing falls back to it |
| `CLAUDE_MODEL` | `claude-sonnet-5` | Model for article research, outline, draft and editing (Claude's turn) |
| `WRITING_PROVIDER` | `gemini` | `gemini`, `claude`, or `split` (editorial topics → Claude, competitor opportunities → Gemini) |
| `CLAUDE_ARTICLE_SHARE` | `10` | With `split`: articles Claude may write per `SCHEDULER_TIMEZONE` day |
| `WRITING_REASONING_EFFORT` / `RESEARCH_REASONING_EFFORT` | `medium` / `low` | Gemini `thinking_level` for writing and for research |
| `ARTICLE_MAX_TOKENS` | `400000` | Token budget per article, across every step and resume |
| `ARTICLE_TARGET_WORDS` / `ARTICLE_MIN_WORDS` | `1500` / `600` | Length the writer aims for / the minimum to complete |
| `ARTICLE_MAX_CONTEXT_CHARS` | `40000` | Research material (facts, sources) included in writing prompts |
| `ARTICLE_RESEARCH_MAX_QUERIES` | `6` | Research questions (Google Search queries requested) |
| `ARTICLE_RESEARCH_MAX_SOURCES` | `10` | Candidate sources read and kept |
| `ARTICLE_RESEARCH_MAX_URL_CONTEXT_CALLS` | `2` | Page-reading calls (URL context; up to 20 pages each) |
| `ARTICLE_RESEARCH_MAX_TOKENS` | `150000` | Research token cap (within the article budget) |
| `ARTICLE_RESEARCH_MIN_SOURCES` | `2` | Usable sources required to continue (0 allows an article without external sources) |
| `GEMINI_QUALITY_MODEL` | `GEMINI_MODEL` | Model for fact-checking, SEO and the judge (revisions use `GEMINI_WRITING_MODEL`) |
| `QUALITY_REASONING_EFFORT` | `low` | Gemini `thinking_level` for fact-checking, SEO and the judge |
| `QUALITY_MAX_TOKENS` | `300000` | Phase 6 tokens per article, across validations and revisions (within `ARTICLE_MAX_TOKENS`) |
| `QUALITY_MAX_REVISIONS` | `2` | Automatic revisions from one edited version (0 = validate only) |
| `QUALITY_MIN_SCORE` | `70` | Minimum score for `ready` |
| `QUALITY_MAX_CONTRADICTED` / `_UNSUPPORTED_RATIO` / `_UNCITED_CLAIMS` | `0` / `0.1` / `3` | Gate limits |
| `QUALITY_WEIGHTS` | see [Score and gates](#score-and-gates) | Component weights as JSON (rescaled to 100) |
| `FACT_CHECK_BATCH_SIZE` / `_MAX_REREADS` / `_MAX_UNCITED_CANDIDATES` | `8` / `4` / `30` | Claims per call; sources re-read per version; uncited sentences checked per version |
| `ORIGINALITY_NGRAM_SIZE` / `_FLAG_THRESHOLD` / `_MAX_OVERLAP` | `8` / `0.25` / `0.5` | Shingle size; flag and severe similarity |
| `ORIGINALITY_COMMON_DOC_FREQUENCY` / `_MIN_PASSAGE_WORDS` | `3` / `12` | Pages that make a phrase common; shortest passage checked |
| `SEO_TITLE_MAX_CHARS` / `SEO_DESCRIPTION_MIN_CHARS` / `_MAX_CHARS` | `60` / `70` / `160` | Meta tag lengths |
| `SEO_MAX_KEYWORD_DENSITY` | `0.03` | Keyword density above which repetition is stuffing |
| `CMS_PROVIDER` | `github` | The publishing target: `github` (the site's repository) or `wordpress` (the earlier adapter) |
| `PUBLISH_SITE_URL` | *(empty)* | The public site the articles appear on (live-URL verification, internal links) |
| `GITHUB_REPO` / `GITHUB_TOKEN` | *(empty)* | The site's repository (`owner/name`) and a fine-grained token with Contents and Pull requests read/write on it (secret; never logged) |
| `GITHUB_BASE_BRANCH` / `GITHUB_CONTENT_DIR` / `GITHUB_BRANCH_PREFIX` | `main` / `src/content/blog` / `blog/` | Where posts live and the branch each is proposed on |
| `GITHUB_API_URL` | `https://api.github.com` | The API host (GitHub Enterprise only) |
| `GITHUB_DEPLOY_TIMEOUT_SECONDS` / `GITHUB_DEPLOY_POLL_SECONDS` | `900` / `15` | Waiting for Vercel's preview or production deployment |
| `PUBLISH_AUTHOR_NAME` / `_ROLE` / `_INITIALS` / `_LINKEDIN` | `Engageo Team` / `AI Content` / `EN` / *(empty)* | The byline of agent-written posts (configuration, never model output) |
| `PUBLISH_CTA_TITLE` / `_BODY` / `_LABEL` / `_HREF` | see `.env.example` | The one call-to-action block every post ends with |
| `PUBLISH_BYLINE` | see `.env.example` | The closing italic byline paragraph |
| `PUBLISH_COVER_IMAGES` | `false` | Give each published post one cover picture, commit it to the post's branch and name it in the frontmatter |
| `COVER_IMAGE_SOURCE` | `gemini` | Where that picture comes from: `gemini` (an illustration from `GEMINI_IMAGE_MODEL`) or `pexels` (a stock photo, no model call) |
| `PEXELS_API_KEY` | *(empty)* | Only for `COVER_IMAGE_SOURCE=pexels`: a free key from [pexels.com/api](https://www.pexels.com/api/), sent to `api.pexels.com` only (secret; never logged or stored). Empty: covers are skipped with a warning |
| `COVER_IMAGE_DIR` / `COVER_IMAGE_URL_PREFIX` | `public/blog/covers` / `/blog/covers` | Where the picture is committed in the site's repository, and the path `coverImage` points at |
| `CMS_REQUEST_TIMEOUT` / `CMS_MAX_RETRIES` | `30` / `2` | Seconds per CMS request; retries of transient failures only |
| `WORDPRESS_BASE_URL` | *(empty)* | Your site (https; http only for localhost; no credentials in it) |
| `WORDPRESS_USERNAME` / `WORDPRESS_APPLICATION_PASSWORD` | *(empty)* | The WordPress user and an Application Password (secret; never logged or stored) |
| `PUBLISH_DEFAULT_STATUS` | `draft` | `draft` (GitHub: a pull request with a verified preview), `pending` or `publish` (`publish` needs `PUBLISH_ALLOW_DIRECT_PUBLISH`) |
| `WORDPRESS_DEFAULT_AUTHOR_ID` / `WORDPRESS_DEFAULT_CATEGORY_ID` | *(empty)* | Post author; category when the SEO package has none |
| `PUBLISH_ALLOW_DIRECT_PUBLISH` | `false` | Required to make a post public (GitHub: to merge the pull request) |
| `WORDPRESS_CREATE_MISSING_TERMS` | `false` | Create missing categories and tags (else a missing category blocks; missing tags are left out) |
| `PUBLISH_AUTO_APPROVE` | `false` | Approve ready articles automatically when publishing (never over a rejection) |
| `PUBLISH_DRAFT_FIRST` | `true` | Going public: a verified draft first |
| `SCHEDULER_ENABLED` | `false` | The worker fires schedules (manual runs work either way) |
| `SCHEDULER_TIMEZONE` | `Asia/Kolkata` | Schedules and daily limits use this calendar (IANA name) |
| `FULL_PIPELINE_SCHEDULE`, `SCAN_SCHEDULE`, `ANALYSIS_SCHEDULE`, `OPPORTUNITY_SCHEDULE`, `EDITORIAL_SCHEDULE`, `ARTICLE_GENERATION_SCHEDULE`, `QUALITY_SCHEDULE`, `PUBLISH_SCHEDULE` | *(empty)* | Cron (`0 6 * * *`) or `@hourly` / `@daily` / `@weekly`; empty: not scheduled |
| `SCHEDULER_CATCH_UP_HOURS` / `SCHEDULER_POLL_SECONDS` | `24` / `30` | How far back one catch-up run looks after an outage; how often the worker checks the queue |
| `AUTOMATED_PUBLISHING_ENABLED` | `false` | The kill switch: false keeps the pipeline away from the CMS |
| `MAX_ARTICLES_GENERATED_PER_DAY` / `MAX_ARTICLES_PER_DAY` | `3` / `1` | Articles created from competitor opportunities / public posts per local day (0 = none) |
| `MAX_EDITORIAL_ARTICLES_PER_DAY` | `0` | Articles created from editorial topics per local day, a separate allowance (0 = the editorial stage is off) |
| `MAX_CONCURRENT_PIPELINES` | `1` | Jobs that can spend Gemini tokens at the same time |
| `JOB_STALE_AFTER_MINUTES` | `60` | No heartbeat for this long (and no live process): continued from the checkpoint |
| `JOB_MAX_ATTEMPTS` / `JOB_RETRY_BASE_SECONDS` / `JOB_RETRY_MAX_SECONDS` | `3` / `300` / `3600` | Transient-failure retries with exponential backoff |
| `PIPELINE_APPROVE_OPPORTUNITIES` / `PIPELINE_MIN_OPPORTUNITY_SCORE` | `true` / `60` | The pipeline may approve the new or reviewed opportunities it selects, above this score |

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
  for Gemini. It reads prompts like the real model, and can fail, omit documents, return
  invalid output or invent numbers on demand.
- Opportunity scoring is unit-tested on synthetic facts (no database), then end to end on
  PostgreSQL through the CLI and API.
- Article generation is tested end to end with the fake Gemini and a controlled fake "web".
  It includes authoritative pages, a redirect, a made-up URL, a prompt-injection page, a
  competitor page and unsafe URLs. The tests cover failure and resume at every step, crash
  recovery, concurrency, budgets and prompt-version changes. `uv run pytest -m llm_live
  tests/live/test_live_article.py` writes one real article (about 40k tokens).
- Article validation is tested end to end with the fake Gemini. The tests cover each verdict,
  evidence that isn't in the source, re-reads and unsafe source URLs, uncited claims,
  competitor and company overlap, common phrases, SEO validation, every gate, the revision
  loop (fixing, worse, unchanged and unusable revisions, bounds, budgets), resume after
  failure, cancellation, dependency boundaries between steps, the API and the CLI.
  `uv run pytest -m llm_live tests/live/test_live_quality.py` validates one article with the
  real Gemini (about 20-30k tokens).
- Approval and publishing are tested end to end against `tests/fakewordpress.py`, an
  in-memory WordPress REST API. It behaves like WordPress for auth, slugs, terms and statuses,
  and can inject failures (`401`, `403`, `400`, `429`, `5xx`, timeouts, lost responses,
  malformed answers, redirects). The tests cover:
  - approval and each kind of invalidation;
  - rendering and escaping;
  - dry runs (no change), preflight, drafts, going public;
  - idempotency (one post after a lost response), reconciliation;
  - version safety, slug collisions, terms, locking, and credentials never leaking.

  `LIVE_WORDPRESS=1 uv run pytest -m cms_live tests/live/test_live_wordpress.py` writes,
  updates and trashes one test draft on a real site (never publishes).
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
    company.py                 versioned company profiles
    relevance.py               deterministic matching: stems, strategic fit, near-duplicates
    opportunity_signals.py     signals, gaps, score, suggestion, change reasons (no LLM)
    opportunities.py           generation runs: score → store → interpret; status changes
    editorial.py               editorial topics: ideas from the profile → checked → opportunities
    numbers.py                 drops model sentences whose numbers aren't in the evidence
    articles.py                article runs: checkpoints, resume, budgets, one live article
    article_brief.py           the deterministic brief
    research.py                search → URL screening → URL-context reading → sources and facts
    article_writing.py         outline, draft and editorial pass (Gemini)
    article_content.py         citations, completion checks, slugs, Markdown preview (no LLM)
    article_file.py            the import file format: frontmatter + Markdown → ArticleContent
    article_import.py          an article a person wrote → opportunity, article, authored report
    checkpoints.py             step fingerprints and checkpoint lookup (Phases 5 and 6)
    quality.py                 validation runs: steps, revision loop, best version, finish
    fact_check.py              claim verdicts, evidence checks, re-reads, uncited claims
    originality.py             n-gram similarity against stored pages (no LLM)
    seo.py                     keyword candidates, package validation, SEO checks
    quality_metrics.py         structure, length, readability, citation metrics (no LLM)
    quality_decision.py        score, gates, issue order, best-version choice (no LLM)
    quality_review.py          the Gemini judge and revisions
    approvals.py, approval_rules.py   approval decisions and when they stop applying
    article_render.py          target-neutral safe HTML: citations, sources, links, FAQ (no LLM)
    article_markdown.py        the same body as MDX-safe Markdown: escaping, inline citations
    publishing.py              publishing runs: preflight, idempotency, reconciliation, verify
    daily_limits.py            daily allowances per local day; the atomic publishing slot
    jobs.py                    jobs: queue, claim, locks, heartbeat, retries, recovery, cancel
    pipeline.py                the pipeline stages, checkpoints, selection, planning mode
    scheduler_state.py         pause / resume and the status dashboard
  scheduling/                  schedules and days (cron, time zones), retry policy, the
                               APScheduler worker, wiring
  cms/                         target-neutral interface (PublishingAdapter) and its adapters
    github/client.py           GitHub REST API: token, bounded read retries, error mapping, read-only
    github/mdx.py              the site's content contract: frontmatter, slug, category, safety
    github/publisher.py        branch, file, pull request, merge, Vercel deployments, live checks
    wordpress/                 the earlier adapter (client, publisher), kept until removed
  prompts/                     versioned prompts + their structured-output schemas
  db/                          models (by layer), sessions, advisory locks, queries, migrations
  llm/                         provider-agnostic LLM interface + Gemini provider
  api/                         HTTP routes and schemas
migrations/                    Alembic migrations
config/                        *.example.yaml: competitors, topics, company, scoring
tests/                         unit, integration (incl. PostgreSQL) and opt-in live tests
```

## Licensing

Small patterns are adapted from MIT-licensed projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
