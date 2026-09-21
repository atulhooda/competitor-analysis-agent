# The scheduling worker, for an always-on host (Railway, Fly, a VPS). It fires the
# schedules in the environment and runs the jobs: scan → analyze → opportunities →
# editorial → generate → quality → approval → publish.
#
# The image carries no configuration and no secrets: everything comes from environment
# variables at run time (see deploy/RAILWAY.md). The company profile and the competitor
# list live in the database, not in the image, so the YAML files are not needed here.

FROM python:3.12-slim-bookworm

# uv, pinned: the lockfile is what makes a deploy reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first: they change far less often than the code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
RUN uv sync --frozen --no-dev

# Never run as root: this process talks to GitHub with a write token.
RUN useradd --create-home --uid 10001 agent && chown -R agent:agent /app
USER agent

ENV PATH="/app/.venv/bin:$PATH"

# The schema is brought up to date on every start (it is idempotent), then the worker runs
# in the foreground so the host restarts it if it ever exits.
CMD ["sh", "-c", "alembic upgrade head && exec python -m app worker"]
