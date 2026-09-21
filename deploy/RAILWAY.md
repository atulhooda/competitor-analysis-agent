# Running the agent on Railway

The agent then writes and publishes on its own schedule, whether or not your laptop is on.
One always-on worker process, one PostgreSQL database, no web service needed.

```
Railway project
├── Postgres                 (Railway's plugin)
└── content-agent            (this repository, Dockerfile)
       alembic upgrade head && python -m app worker
```

The image holds no configuration and no secrets. The company profile and the competitor
list live in the database, so nothing but environment variables has to be set.

---

## 1. Create the project

1. Railway → **New Project** → **Deploy from GitHub repo** → `atulhooda/competitor-analysis-agent`,
   branch `feat/phase-8-scheduling` (or `main` once it is merged).
2. Railway detects the `Dockerfile` at the repository root and builds it. No start command
   to set: the image already runs the migrations and then the worker.
3. In the same project: **New** → **Database** → **Add PostgreSQL**.

## 2. Variables

On the **content-agent** service → *Variables*. Copy the values from your local `.env`
(`GEMINI_API_KEY`, `GITHUB_TOKEN`, `PEXELS_API_KEY`, `VERCEL_PROTECTION_BYPASS_SECRET`);
never paste them into chat, a commit or a screenshot.

| Variable | Value |
|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` — Railway substitutes it; add `+psycopg` (see below) |
| `GEMINI_API_KEY` | your key |
| `GITHUB_TOKEN` | the fine-grained token with write access to `siddharthpathania/engageo-website` |
| `GITHUB_REPO` | `siddharthpathania/engageo-website` |
| `PUBLISH_SITE_URL` | `https://www.engageoagency.com` |
| `VERCEL_PROTECTION_BYPASS_SECRET` | the preview bypass secret |
| `PEXELS_API_KEY` | your Pexels key (post covers) |
| `SCHEDULER_ENABLED` | `true` |
| `SCHEDULER_TIMEZONE` | `Asia/Kolkata` |
| `FULL_PIPELINE_SCHEDULE` | `0 14 * * *` (14:00 IST daily) |
| `AUTOMATED_PUBLISHING_ENABLED` | `true` |
| `PUBLISH_AUTO_APPROVE` | `true` |
| `PUBLISH_ALLOW_DIRECT_PUBLISH` | `true` |
| `PIPELINE_APPROVE_OPPORTUNITIES` | `true` |
| `PIPELINE_MIN_OPPORTUNITY_SCORE` | `50` |
| `MAX_ARTICLES_GENERATED_PER_DAY` | `4` |
| `MAX_EDITORIAL_ARTICLES_PER_DAY` | `14` |
| `MAX_ARTICLES_PER_DAY` | `15` |
| `LLM_DAILY_TOKEN_BUDGET` | `8000000` |
| `PUBLISH_COVER_IMAGES` | `true` |
| `COVER_IMAGE_SOURCE` | `pexels` |
| `APP_ENV` | `production` |
| `LOG_JSON` | `true` |

**The one gotcha:** Railway's `DATABASE_URL` starts with `postgresql://`, and this app uses
the async psycopg driver, so it must read `postgresql+psycopg://`. Either set

```
DATABASE_URL=postgresql+psycopg://${{Postgres.PGUSER}}:${{Postgres.PGPASSWORD}}@${{Postgres.PGHOST}}:${{Postgres.PGPORT}}/${{Postgres.PGDATABASE}}
```

or paste Railway's URL with `+psycopg` inserted after `postgresql`.

## 3. Move the existing data across

Skip this and the agent starts from an empty database: it would re-propose topics it has
already published (it still checks your live sitemap, so it won't publish exact duplicates,
but it loses every score, decision and history). Moving the data keeps all of it.

From this Mac, with the project's local database running:

```bash
# 1. Dump everything (schema + rows)
/opt/homebrew/opt/postgresql@16/bin/pg_dump \
  "postgresql://postgres@127.0.0.1:5433/competitor_agent" \
  --no-owner --no-privileges -Fc -f ~/Desktop/agent-backup.dump

# 2. Restore into Railway (copy the connection string from the Postgres service →
#    "Connect" → "Postgres Connection URL"; keep it out of your shell history with a space
#    before the command)
 /opt/homebrew/opt/postgresql@16/bin/pg_restore \
  --no-owner --no-privileges --clean --if-exists \
  -d "postgresql://…railway-connection-url…" ~/Desktop/agent-backup.dump
```

Then redeploy the service so `alembic upgrade head` runs against it.

## 4. Check it works

- **Logs** (service → *Deployments* → *View logs*): `worker.started` with the schedule, then
  at 14:00 IST `job.started … full_pipeline`.
- **A run on demand**, without waiting for 14:00: Railway → service → *Settings* →
  **Run command**, or `railway run python -m app worker --help`. Simplest check:
  `railway run python -m app schedule status`.
- **What it published**: `railway run python -m app articles list` and your blog.

## 5. Cost

Railway bills the container and the database by usage: roughly $5–10/month for a worker
this small plus Postgres. Gemini is billed separately by Google and is the larger number at
10+ articles a day.

## Keeping the laptop copy

Once Railway runs the schedule, **turn the Mac worker off** so two workers don't race:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.engageo.content-agent.plist
```

They would not corrupt anything — every job is claimed atomically and the daily limits are
enforced under a lock — but they would compete for the same allowances against two different
databases, which makes the counts confusing.
