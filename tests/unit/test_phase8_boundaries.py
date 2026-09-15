"""Static guards on the Phase 8 code: one authoritative publication path (Phase 7's
PublishingService), and no way to approve an article. Read from the source, so a future
shortcut fails here even if no behavior test covers it."""

import ast
from pathlib import Path

APP = Path(__file__).parents[2] / "app"
PHASE8 = [
    APP / "services" / "pipeline.py",
    APP / "services" / "jobs.py",
    APP / "services" / "daily_limits.py",
    APP / "services" / "scheduler_state.py",
    APP / "db" / "pipeline_queries.py",
    APP / "api" / "v1" / "jobs.py",
    *sorted((APP / "scheduling").glob("*.py")),
]
# What would write to a CMS, or decide an article's approval, without Phase 7's checks.
CMS_WRITES = {"create_post", "update_post", "resolve_terms", "find_posts", "build_payload", "delete"}  # fmt: skip
APPROVAL_WRITES = {"approve", "reject", "decide_in", "auto_approve_in", "invalidate_all", "invalidate_stale"}  # fmt: skip


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
    return names


def _called_attributes(tree: ast.Module) -> list[str]:
    return [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]  # fmt: skip


def test_the_phase8_files_are_the_ones_checked() -> None:
    assert all(path.exists() for path in PHASE8)
    assert {p.name for p in PHASE8} >= {"pipeline.py", "jobs.py", "worker.py", "runtime.py"}


def test_phase8_never_talks_to_a_cms_itself() -> None:
    for path in PHASE8:
        tree = _tree(path)
        imports = _imports(tree)
        assert not any(name.startswith("app.cms.wordpress") for name in imports), path
        assert not any(name.endswith(("WordPressClient", "WordPressPublisher", "CMSPublisher")) for name in imports), path  # fmt: skip
        assert not CMS_WRITES & set(_called_attributes(tree)), path


def test_the_pipeline_publishes_only_through_phase7_publish_now() -> None:
    tree = _tree(APP / "services" / "pipeline.py")
    publishing_calls = [
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "publishing"
    ]
    assert publishing_calls == ["publish_now"]
    # ...always with the daily limit for a public post.
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "publish_now")  # fmt: skip
    assert {k.arg for k in call.keywords} >= {"trigger", "target", "daily_limit"}


def test_phase8_never_decides_an_article_approval() -> None:
    for path in PHASE8:
        tree = _tree(path)
        assert not APPROVAL_WRITES & set(_called_attributes(tree)), path
        assert "app.services.approvals" not in _imports(tree), path
        constructed = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}  # fmt: skip
        assert "ArticleApproval" not in constructed, path


def test_the_only_status_the_pipeline_sets_is_an_opportunity_approval() -> None:
    tree = _tree(APP / "services" / "pipeline.py")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "set_status"]  # fmt: skip
    assert len(calls) == 1
    [call] = calls
    receiver = call.func.value  # type: ignore[attr-defined]
    assert isinstance(receiver, ast.Attribute)
    assert receiver.attr == "opportunities"  # self._s.opportunities.set_status(...)
    status = call.args[1]
    assert isinstance(status, ast.Attribute)
    assert (ast.unparse(status.value), status.attr) == ("OpportunityStatus", "APPROVED")
    actor = next(k.value for k in call.keywords if k.arg == "actor")
    assert isinstance(actor, ast.Constant)
    assert actor.value == "pipeline"
