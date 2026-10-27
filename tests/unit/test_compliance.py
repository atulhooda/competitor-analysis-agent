"""Clinic compliance: every rule, on its own and together. Pure functions; no network."""

from pathlib import Path
from typing import Any

import pytest

from app.compliance import (
    CLOSING_NOTE,
    MEDICINE_ADVICE,
    ClinicConfig,
    ComplianceInput,
    check,
    content_problems,
    load_clinic,
    load_terms,
    writer_rules,
)
from app.compliance.site import from_site_post
from app.compliance.terms import parse_terms

TERMS = tuple(parse_terms("isotretinoin => a prescription oral acne medicine\nBotox => (describe the treatment)\nhydroquinone => a skin-lightening cream prescribed by your dermatologist\nsteroid\n"))  # fmt: skip
REPO_TERMS = Path(__file__).resolve().parents[2] / "config/compliance/group-a-terms.txt"


def clinic(**overrides: Any) -> ClinicConfig:
    values: dict[str, Any] = {
        "clinic_id": "test-clinic", "name": "Test Clinic",
        "doctor": {"name": "Dr. Asha Rao", "qualification": "MBBS, MD (Dermatology)", "council": "Maharashtra Medical Council", "registration_number": "2012/03/1234"},
        "treatments": ["Laser hair reduction", "Chemical peels", "Fat reduction"], "devices": ["Coolite BOLT diode laser"],
        "areas": ["Kalyani Nagar", "Pune"],
    }  # fmt: skip
    values.update(overrides)
    return ClinicConfig.model_validate(values)


GOOD_BODY = f"""Laser hair reduction in Kalyani Nagar, Pune is planned around your skin type.

## How sessions work

Hair grows in cycles, so a course takes several sessions spaced weeks apart.

## Before your first session

Shave the area the day before. To ask about it, call the clinic.

{CLOSING_NOTE}
"""


def post(body: str = GOOD_BODY, **overrides: Any) -> ComplianceInput:
    # fmt: off
    values: dict[str, Any] = {
        "fields": [("the title", "Laser Hair Reduction in Pune: What to Expect"), ("the slug", "laser-hair-reduction-pune"), ("the meta description", "How laser hair reduction works, how many sessions it takes and how to prepare, from a dermatology clinic in Pune.")],
        "body": body, "urls": [("the post URL", "/blog/laser-hair-reduction-pune")],
        "author_box": clinic().doctor.author_box, "ctas": ["Call the clinic"],
    }
    # fmt: on
    values.update(overrides)
    return ComplianceInput(**values)


def failing(report: Any) -> dict[str, list[str]]:
    return {r.rule: [f.term for f in r.findings] for r in report.rules if not r.passed}


def test_a_compliant_post_passes_every_rule() -> None:
    report = check(post(), clinic(), TERMS)
    assert failing(report) == {}
    assert report.passed
    assert [r.rule for r in report.rules] == [f"R{n}" for n in range(4, 17)]


@pytest.mark.parametrize(
    ("where", "value"),
    [
        ("fields", [("the title", "Isotretinoin for Acne in Pune")]),
        ("fields", [("the slug", "botox-vs-fillers-pune")]),
        ("fields", [("the cover image alt text", "A vial of BOTOX on a tray")]),
        ("urls", [("the cover image file name", "botox_vial_1.jpg")]),
        ("urls", [("citation 'A study'", "https://journal.example.org/isotretinoin-outcomes")]),
        ("body", "See [our steroid guide](/blog/x).\n\n" + GOOD_BODY),
    ],
)
def test_a_drug_name_is_found_wherever_it_is(where: str, value: Any) -> None:
    base = post()
    if where == "fields":
        base.fields = [*base.fields, *value]
    elif where == "urls":
        base.urls = [*base.urls, *value]
    else:
        base.body = value
    report = check(base, clinic(), TERMS)
    assert "R4" in failing(report)


def test_a_drug_finding_says_where_and_what_to_write_instead() -> None:
    body = "A prescription oral acne medicine such as isotretinoin may be advised. " + MEDICINE_ADVICE + "\n\n" + GOOD_BODY  # fmt: skip
    [finding] = next(r for r in check(post(body), clinic(), TERMS).rules if r.rule == "R4").findings
    assert finding.term == "isotretinoin"
    assert finding.location == "body, the introduction"
    assert finding.suggestion == 'say "a prescription oral acne medicine" instead'


def test_a_medicine_class_needs_the_dermatologists_advice_line() -> None:
    without = "Your dermatologist may prescribe a prescription retinoid cream.\n\n" + GOOD_BODY
    assert "R5" in failing(check(post(without), clinic(), TERMS))
    with_line = f"Your dermatologist may prescribe a prescription retinoid cream. {MEDICINE_ADVICE}\n\n" + GOOD_BODY  # fmt: skip
    assert "R5" not in failing(check(post(with_line), clinic(), TERMS))


@pytest.mark.parametrize("text", ["Take 20 mg daily.", "Apply a 4% cream.", "Use it twice a day.", "Apply a pea-sized amount.", "Here is how to use it."])  # fmt: skip
def test_doses_strengths_and_schedules_are_blocked(text: str) -> None:
    assert "R6" in failing(check(post(text + "\n\n" + GOOD_BODY), clinic(), TERMS))


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("Ask about our bridal packages.", "R8"),
        ("A 20% discount this month.", "R8"),
        ("Read our patients say it works.", "R9"),
        ("Over 3,00,000 patients treated.", "R9"),
        ("A 98% success rate.", "R9"),
        ("See the before and after photos.", "R9"),
        ("The best laser in town.", "R9"),
        ("It is painless and permanent.", "R9"),
        ("Guaranteed results.", "R9"),
        ("Led by an expert dermatologist.", "R10"),
        ("Book now, limited slots!", "R11"),
        ("Unlike CoolSculpting, this is gentle.", "R13"),
        ("This cures acne.", "R14"),
        ("In this case study, a woman of 34…", "R15"),
        ("It costs about $200.", "R16"),
    ],
)
def test_promotion_claims_and_scope_are_blocked(text: str, rule: str) -> None:
    assert rule in failing(check(post(text + "\n\n" + GOOD_BODY), clinic(), TERMS))


def test_a_studys_numbers_are_education_not_a_success_rate() -> None:
    text = "In one trial of 120 patients, 25% of patients saw fewer breakouts.\n\n"
    assert "R9" not in failing(check(post(text + GOOD_BODY), clinic(), TERMS))


def test_a_device_the_clinic_uses_may_be_named() -> None:
    text = "The clinic uses the Coolite BOLT diode laser.\n\n"
    assert "R13" not in failing(check(post(text + GOOD_BODY), clinic(), TERMS))


def test_a_pushy_or_unlisted_call_to_action_is_blocked() -> None:
    assert "R11" in failing(check(post(ctas=["Book your slot"]), clinic(), TERMS))
    assert "R11" in failing(check(post(ctas=["Get started"]), clinic(), TERMS))


def test_without_a_registration_number_every_post_is_blocked() -> None:
    missing = clinic(doctor={"name": "Dr. Asha Rao", "qualification": "MBBS, MD", "council": "Maharashtra Medical Council", "registration_number": None})  # fmt: skip
    result = next(r for r in check(post(author_box=None), missing, TERMS).rules if r.rule == "R12")
    assert not result.passed
    assert "registration number" in result.detail


@pytest.mark.parametrize("placeholder", ["[PENDING]", "pending", "TBD", " ", "[ TBC ]"])
def test_a_placeholder_registration_number_counts_as_missing(placeholder: str) -> None:
    c = clinic(doctor={"name": "Dr. Asha Rao", "qualification": "MBBS", "council": "MMC", "registration_number": placeholder})  # fmt: skip
    assert c.doctor.registration_number is None
    assert c.doctor.author_box is None


def test_a_placeholder_anywhere_in_a_post_is_blocked() -> None:
    report = check(post(author_box="Dr. Asha Rao, Maharashtra Medical Council Reg. No. [PENDING]"), clinic(), TERMS)  # fmt: skip
    assert failing(report)["R12"] == ["[PENDING]"]


def test_a_post_must_end_with_the_consultation_note() -> None:
    assert "R14" in failing(check(post(GOOD_BODY.replace(CLOSING_NOTE, "")), clinic(), TERMS))


def test_a_post_must_name_the_clinics_area() -> None:
    body = GOOD_BODY.replace("in Kalyani Nagar, Pune ", "")
    report = check(post(body, fields=[("the title", "Laser Hair Reduction: What to Expect")]), clinic(), TERMS)  # fmt: skip
    assert "R16" in failing(report)


def test_only_the_clinics_doctors_approve() -> None:
    c = clinic(approvers=["Dr. Asha Rao", "Dr. Vikram Shah"])
    assert c.approves(" dr.  asha rao ")
    assert c.approves("Dr. Vikram Shah")
    assert not c.approves("cli")
    assert clinic().allowed_approvers == ("Dr. Asha Rao",)


def test_what_the_text_breaks_goes_to_the_quality_gate_with_a_rewrite() -> None:
    problems = content_problems("Isotretinoin in Pune", ["The best option"], "It is painless.\n", clinic(), TERMS)  # fmt: skip
    assert any(p.startswith("R4") and "a prescription oral acne medicine" in p for p in problems)
    assert any(p.startswith("R9") and "'best'" in p for p in problems)
    assert not any(p.startswith(("R12", "R14")) for p in problems)  # assembled at publishing


def test_the_writer_is_given_the_rules() -> None:
    rules = " ".join(writer_rules(clinic(), TERMS))
    for needle in (MEDICINE_ADVICE, "Doses", "testimonials", "Book now", "CoolSculpting", "rupees", "Kalyani Nagar"):  # fmt: skip
        assert needle in rules


def test_the_shared_term_list_loads_and_covers_the_examples() -> None:
    terms = {t.term.casefold(): t for t in load_terms(REPO_TERMS)}
    for name in ("botox", "botulinum", "dysport", "xeomin", "isotretinoin", "accutane", "tretinoin", "adapalene", "hydroquinone", "minoxidil", "finasteride", "triamcinolone", "steroid"):  # fmt: skip
        assert name in terms, name
    assert terms["isotretinoin"].replacement == "a prescription oral acne medicine"
    assert terms["botox"].replacement is None


def test_the_clinic_files_in_the_repository_load() -> None:
    directory = Path(__file__).resolve().parents[2] / "config/clinics"
    for path in directory.glob("*.yaml"):
        assert load_clinic(directory, path.stem).clinic_id == path.stem


def test_a_post_as_the_site_stores_it_is_checked_the_same_way() -> None:
    fields = {"slug": "botox-vs-dermal-fillers-in-pune", "title": "Botox vs Fillers in Pune", "metaTitle": "x", "metaDescription": "y", "heroImage": {"src": "/images/blog/botox.jpg", "alt": "Injection"}}  # fmt: skip
    report = check(from_site_post(fields, GOOD_BODY), clinic(), TERMS)
    r4 = next(r for r in report.rules if r.rule == "R4")
    assert {f.location for f in r4.findings} >= {"the title", "the slug", "the post URL", "the hero image file name"}  # fmt: skip
    assert "R12" in failing(report)  # no reviewedBy author box


def test_the_websites_procedure_terms_are_blocked_too() -> None:
    terms = load_terms(REPO_TERMS)
    for text in ("Dermal fillers for smile lines.", "A liposuction alternative.", "Skin boosters explained.", "Mesotherapy for hair.", "An injection-free option."):  # fmt: skip
        assert "R4" in failing(check(post(text + "\n\n" + GOOD_BODY), clinic(), terms)), text
    assert "R4" not in failing(check(post("Your skin's lipid barrier matters.\n\n" + GOOD_BODY), clinic(), terms))  # fmt: skip


def test_superlatives_in_a_search_snippet_get_neutral_words() -> None:
    from app.compliance import neutral_wording

    assert neutral_wording("Permanent results with the safest, painless care") == "Long-lasting results with the safe, comfortable care"  # fmt: skip
    assert neutral_wording("A topical cream") == "A topical cream"  # whole words only
