"""Clinic compliance in the publishing workflow (rules R1-R3): every draft is checked and any
failure blocks it before the CMS is touched; a post becomes public only with the clinic
doctor's recorded approval; the pipeline only ever opens drafts. Real PostgreSQL, the fake
Gemini and the fake CMS. No network."""

from pathlib import Path
from typing import Any

import pytest

from app.domain.publishing import ApprovalChannel, PublicationStatus, TargetStatus
from tests.integration.test_publishing import Pub
from tests.integration.test_publishing import pub as pub
from tests.integration.test_quality import world as world

CLINIC = """\
clinic:
  clinic_id: test-clinic
  name: Test Clinic
  doctor:
    name: Dr. Asha Rao
    qualification: MBBS, MD (Dermatology)
    council: Maharashtra Medical Council
    registration_number: {registration}
  areas: {areas}
"""


def clinic_settings(directory: Path, *, registration: str = "'2012/03/1234'", areas: str = "[]", **overrides: Any) -> dict[str, Any]:  # fmt: skip
    (directory / "test-clinic.yaml").write_text(CLINIC.format(registration=registration, areas=areas), encoding="utf-8")  # fmt: skip
    return {"clinic_id": "test-clinic", "clinics_dir": directory, "publish_auto_approve": True, "publish_allow_direct_publish": True, "publish_cta_label": "Call the clinic", **overrides}  # fmt: skip


def checks(report: Any) -> dict[str, tuple[bool, str]]:
    return {c.name: (c.passed, c.detail) for c in report.checks}


async def test_every_draft_carries_the_compliance_report(pub: Pub, tmp_path: Path) -> None:
    report = await pub.service(**clinic_settings(tmp_path)).preflight(pub.article_id, target=TargetStatus.DRAFT)  # fmt: skip
    assert "compliance" in checks(report)
    assert report.compliance is not None
    assert [r["rule"] for r in report.compliance["rules"]] == [f"R{n}" for n in range(4, 17)]


async def test_a_compliance_failure_blocks_the_draft_before_the_cms_is_touched(pub: Pub, tmp_path: Path) -> None:  # fmt: skip
    # The post never names the clinic's area (R16).
    settings = clinic_settings(tmp_path, areas="[Kalyani Nagar]")
    report = await pub.service(**settings).preflight(pub.article_id, target=TargetStatus.DRAFT)
    passed, detail = checks(report)["compliance"]
    assert not passed
    assert "R16" in detail
    assert not report.ready
    await pub.approve()
    _, outcome = await pub.publish(TargetStatus.DRAFT, **settings)
    assert outcome.status is PublicationStatus.BLOCKED, outcome.error
    assert "compliance" in (outcome.error or "")
    assert not [c for c in pub.wp.calls if c[0] in ("POST", "PUT", "DELETE")]  # nothing was sent


async def test_making_a_post_public_needs_the_clinic_doctors_approval(pub: Pub, tmp_path: Path) -> None:  # fmt: skip
    settings = clinic_settings(tmp_path)
    service = pub.service(**settings)
    # No approval, or PUBLISH_AUTO_APPROVE's automatic one: not the doctor's.
    first = checks(await service.preflight(pub.article_id, target=TargetStatus.PUBLISH))
    assert first["doctor_approval"][0] is False
    assert "articles approve" in first["doctor_approval"][1]
    # Someone else's approval doesn't count either.
    await pub.approve()
    assert checks(await service.preflight(pub.article_id, target=TargetStatus.PUBLISH))["doctor_approval"][0] is False  # fmt: skip
    # The doctor's does, recorded with their name and the time.
    record, _ = await pub.approvals(**settings).approve(pub.article_id, channel=ApprovalChannel.CLI, approver="Dr. Asha Rao", note="Reviewed the preview.")  # fmt: skip
    passed, detail = checks(await service.preflight(pub.article_id, target=TargetStatus.PUBLISH))["doctor_approval"]  # fmt: skip
    assert passed
    assert detail.startswith("approved by Dr. Asha Rao on ")
    assert record.approver == "Dr. Asha Rao"


async def test_a_draft_needs_no_doctor_yet(pub: Pub, tmp_path: Path) -> None:
    report = await pub.service(**clinic_settings(tmp_path)).preflight(pub.article_id, target=TargetStatus.DRAFT)  # fmt: skip
    assert "doctor_approval" not in checks(report)


async def test_without_a_clinic_nothing_changes(pub: Pub) -> None:
    report = await pub.service().preflight(pub.article_id, target=TargetStatus.DRAFT)
    assert report.compliance is None
    assert "compliance" not in checks(report)
    assert "doctor_approval" not in checks(report)


@pytest.mark.parametrize("clinic", [True, False])
def test_the_pipeline_only_ever_opens_drafts_for_a_clinic(tmp_path: Path, clinic: bool) -> None:
    from app.services.pipeline import PipelineService
    from tests.fakesite import make_settings

    settings = make_settings(**(clinic_settings(tmp_path) if clinic else {"publish_allow_direct_publish": True}))  # fmt: skip
    pipeline = PipelineService.__new__(PipelineService)
    pipeline._settings = settings
    assert pipeline._publish_target() is (TargetStatus.DRAFT if clinic else TargetStatus.PUBLISH)


async def test_without_a_registration_number_a_draft_opens_but_nothing_goes_public(pub: Pub, tmp_path: Path) -> None:  # fmt: skip
    settings = clinic_settings(tmp_path, registration="")
    service = pub.service(**settings)
    draft = checks(await service.preflight(pub.article_id, target=TargetStatus.DRAFT))["compliance"]
    assert draft[0] is True, draft[1]
    assert "doesn't block a draft" in draft[1]
    public = checks(await service.preflight(pub.article_id, target=TargetStatus.PUBLISH))[
        "compliance"
    ]
    assert public[0] is False
    assert "R12" in public[1]


def test_a_missing_registration_number_never_blocks_a_draft_only_a_public_post() -> None:
    from app.compliance import ComplianceReport, RuleResult
    from app.services.publishing import compliance_blocking

    missing = ComplianceReport(clinic_id="c", passed=False, rules=[RuleResult(rule="R12", title="Author box", passed=False, detail="no registration number")])  # fmt: skip
    placeholder = ComplianceReport(clinic_id="c", passed=False, rules=[RuleResult(rule="R12", title="Author box", passed=False, findings=[{"rule": "R12", "term": "[PENDING]", "location": "author box", "excerpt": "", "suggestion": ""}])])  # fmt: skip
    assert compliance_blocking(missing, TargetStatus.DRAFT) == []
    assert [r.rule for r in compliance_blocking(missing, TargetStatus.PUBLISH)] == ["R12"]
    assert [r.rule for r in compliance_blocking(placeholder, TargetStatus.DRAFT)] == ["R12"]
