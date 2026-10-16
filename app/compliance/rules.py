"""The rules: Google Ads restricted drug terms (R4-R7) and India's NMC rules on medical
advertising (R8-R12), accuracy and scope (R13-R16). Deterministic: no model decides
whether a post passes.

Every rule reports pass or fail with each flagged term, where it is and a compliant
rewrite. Rules R1-R3 (draft-only, the doctor's approval, blocking) are the publishing
workflow's; ``app.services.publishing`` enforces them with this report.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.compliance.config import ClinicConfig
from app.compliance.terms import GENERIC_CLASS, Term

CLOSING_NOTE = "This information is general and not a substitute for a consultation. Results vary from person to person."  # fmt: skip
MEDICINE_ADVICE = "Your dermatologist will advise whether this is suitable for you."
PLACEHOLDER = re.compile(r"\[\s*(?:PENDING|TBD|TBC|TO BE CONFIRMED|XXX)\s*\]", re.IGNORECASE)


def _words(*phrases: str) -> list[re.Pattern[str]]:
    """Whole words or phrases, any case; hyphens, underscores and spaces are alike."""
    out = []
    for phrase in phrases:
        parts = [re.escape(w) for w in re.split(r"[\s\-_]+", phrase) if w]
        out.append(re.compile(r"(?<![a-z0-9])" + r"[\s_\-]*".join(parts) + r"(?![a-z0-9])", re.IGNORECASE))  # fmt: skip
    return out


# (pattern, what it is, the rewrite)
PROMOTION = [(p, "promotional offer", "leave offers, prices and packages out: the post explains, it doesn't sell") for p in _words("special offer", "offer price", "discount", "discounted", "% off", "package", "packages", "limited time", "limited-time", "limited period", "special deal", "great deal", "best deal", "deals", "free consultation", "lowest price", "cashback", "EMI")]  # fmt: skip
PROMOTION += [(p, "testimonial or patient story", "remove testimonials and patient stories") for p in _words("testimonial", "testimonials", "patient story", "patient stories", "our patients say", "5-star", "five-star")]  # fmt: skip
# The clinic's own numbers (not a study's: "25% of patients in a trial" is education):
# counts with "+", "over" or "more than", counts of a thousand or more, "our N patients",
# and success or satisfaction rates.
PROMOTION += [(re.compile(r"(?:\b(?:over|more than|nearly|almost)\s+\d[\d,]*|\b\d[\d,]*\s*\+|\b\d{1,3}(?:,\d{2,3})+|\b\d{4,}|\bour\s+\d[\d,]*)\s*(?:happy\s+|satisfied\s+)?(?:patients|clients|customers|procedures|treatments)\b", re.IGNORECASE), "patient or treatment count", "remove the count")]  # fmt: skip
PROMOTION += [(re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\s*(?:success|satisfaction|of our patients)\b|\bsuccess rates?\b|\bsatisfaction rates?\b", re.IGNORECASE), "success rate", "remove the success rate")]  # fmt: skip
PROMOTION += [(p, "before/after image or claim", "remove before/after material") for p in _words("before and after", "before-and-after", "before/after")]  # fmt: skip
SUPERLATIVES = [(p, "superlative or guarantee", "drop the word, or say plainly what the treatment involves") for p in _words("best", "top", "leading", "No. 1", "number one", "#1", "painless", "permanent", "permanently", "guaranteed", "guarantee", "guarantees", "100%", "miracle", "instant results", "world-class", "state-of-the-art", "revolutionary", "breakthrough", "risk-free", "safest", "most advanced", "unmatched", "no side effects", "zero side effects", "flawless")]  # fmt: skip
PORTRAYAL = [(p, "promotional portrayal of the doctor", "describe the doctor only by name and qualification, in the author box") for p in _words("expert", "experts", "renowned", "eminent", "celebrated", "famous", "highly experienced", "award-winning", "pioneer", "pioneering")]  # fmt: skip
CTAS = [(p, "pushy call to action", 'use a neutral call to action: "Call the clinic", "WhatsApp the clinic" or "Send an enquiry"') for p in _words("book now", "book today", "book your", "hurry", "limited slots", "limited seats", "act now", "call now", "don't wait", "do not wait", "reserve your slot", "few slots left", "grab")]  # fmt: skip
DOSES = [(re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|iu|units?)\b", re.IGNORECASE), "dose", "remove the dose: the dermatologist decides it")]  # fmt: skip
DOSES += [(re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(?:cream|gel|lotion|solution|serum|ointment|strength|concentration)\b", re.IGNORECASE), "strength", "remove the strength: the dermatologist decides it")]  # fmt: skip
DOSES += [(re.compile(r"\b(?:once|twice|thrice|three times|\d+\s+times)\s+(?:a|per|every)\s+(?:day|night|week|morning)\b|\bat bedtime\b|\bpea-sized\b|\bdosage\b|\bdoses?\b|\bhow to (?:use|apply|take)\b", re.IGNORECASE), "dosing schedule or instructions", "remove the schedule: say the dermatologist explains how to use any prescribed medicine")]  # fmt: skip
CLAIMS = [(p, "treatment claim", "say what the treatment aims to do and that results vary") for p in _words("cure", "cures", "cured", "curing", "guaranteed to")]  # fmt: skip
PATIENTS = [(p, "patient photo, video or case detail", "remove anything about an identifiable patient") for p in _words("our patient", "our patients", "a patient of ours", "real patient", "case study", "patient photo", "patient photos", "patient video", "patient videos")]  # fmt: skip
CURRENCY = [(re.compile(r"\$|£|€|\bUSD\b|\bdollars?\b|\bAED\b|\bGBP\b|\beuros?\b", re.IGNORECASE), "foreign currency", "use rupees (₹), or leave the price out")]  # fmt: skip
CLASS_WORDING = re.compile(r"\bprescription\b[^.]{0,60}\b(?:medicine|medication|cream|retinoid|tablet|gel|treatment)\b|\bprescribed by your dermatologist\b", re.IGNORECASE)  # fmt: skip
# Trademarked treatments and devices: allowed only when the clinic offers them, by that name.
TRADEMARKS: dict[str, str] = {
    "CoolSculpting": "fat-freezing (cryolipolysis)", "SculpSure": "laser fat reduction", "Emsculpt": "electromagnetic muscle stimulation",
    "HydraFacial": "a hydradermabrasion facial", "Ultherapy": "focused ultrasound skin tightening", "Thermage": "radiofrequency skin tightening",
    "Morpheus8": "radiofrequency microneedling", "Fraxel": "fractional laser resurfacing", "PicoSure": "a picosecond laser", "PicoWay": "a picosecond laser",
    "Profhilo": "an injectable skin booster", "Dermapen": "microneedling", "Venus Legacy": "radiofrequency body contouring", "Vaser": "ultrasound-assisted liposuction",
    "Exilis": "radiofrequency skin tightening", "Clear + Brilliant": "gentle fractional laser", "Juvederm": "a dermal filler", "Restylane": "a dermal filler",
}  # fmt: skip

RULE_TITLES = {
    "R4": "No prescription drug names (brand or generic) anywhere",
    "R5": "Drug classes come with the dermatologist's advice line",
    "R6": "No doses, strengths, schedules or how-to-use instructions",
    "R7": "Group A term list loaded from the shared file",
    "R8": "Educational, not promotional",
    "R9": "No offers, testimonials, counts, success rates, before/after or superlatives",
    "R10": "No promotional portrayal of the doctor",
    "R11": "Neutral calls to action only",
    "R12": "Author box: doctor, qualification, council and registration number",
    "R13": "Only treatments and devices the clinic offers; generic names otherwise",
    "R14": "General education only, ending with the consultation note",
    "R15": "No patient photos, videos or identifiable case details",
    "R16": "Indian context: rupees and the clinic's area",
}


class ComplianceFinding(BaseModel):
    rule: str
    term: str
    location: str
    excerpt: str
    suggestion: str


class RuleResult(BaseModel):
    rule: str
    title: str
    passed: bool
    detail: str = ""
    findings: list[ComplianceFinding] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    clinic_id: str
    passed: bool
    rules: list[RuleResult]

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.rules if not r.passed]

    def summary(self) -> str:
        if self.passed:
            return f"every rule passes for clinic {self.clinic_id} ({len(self.rules)} rules)"
        parts = []
        for r in self.failed:
            terms = ", ".join(dict.fromkeys(f"'{f.term}' ({f.location})" for f in r.findings[:4]))
            parts.append(f"{r.rule} {r.title}: {terms or r.detail}")
        return "; ".join(parts)


@dataclass
class ComplianceInput:
    """Everything of a post that a reader, an ad reviewer or a crawler can see. ``fields``
    are (location, text): title, slug, meta and social tags, headings, FAQ, image alt and
    caption. ``body`` is the Markdown body; ``urls`` are (location, URL) of every link,
    image file and citation."""

    fields: list[tuple[str, str]] = field(default_factory=list)
    body: str = ""
    urls: list[tuple[str, str]] = field(default_factory=list)
    author_box: str | None = None
    ctas: list[str] = field(default_factory=list)


def _body_fields(body: str) -> list[tuple[str, str]]:
    """The body, one entry per line, labelled with the section it sits under."""
    section = "the introduction"
    out = []
    for line in body.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            section = f"the section '{text.lstrip('#').strip()[:60]}'"
            out.append((f"heading '{text.lstrip('#').strip()[:60]}'", text.lstrip("#").strip()))
            continue
        out.append((f"body, {section}", text))
    return out


def _link_texts(body: str) -> list[tuple[str, str]]:
    return [(f"link text '{m.group(1)[:40]}'", m.group(1)) for m in re.finditer(r"\[([^\]]+)\]\([^)]+\)", body)]  # fmt: skip


def _excerpt(text: str, match: re.Match[str]) -> str:
    start, end = max(match.start() - 50, 0), min(match.end() + 50, len(text))
    return (
        ("…" if start else "")
        + " ".join(text[start:end].split())
        + ("…" if end < len(text) else "")
    )


def _scan(rule: str, places: Iterable[tuple[str, str]], patterns: Sequence[tuple[re.Pattern[str], str, str]]) -> list[ComplianceFinding]:  # fmt: skip
    """Every match, once: where terms overlap ("botulinum toxin" and "botulinum"), the
    longest is reported."""
    found = []
    for location, text in places:
        hits = sorted(((m, suggestion) for pattern, _, suggestion in patterns for m in pattern.finditer(text)), key=lambda h: (h[0].start(), -(h[0].end() - h[0].start())))  # fmt: skip
        taken: list[tuple[int, int]] = []
        for match, suggestion in hits:
            if any(s <= match.start() and match.end() <= e for s, e in taken):
                continue
            taken.append((match.start(), match.end()))
            found.append(ComplianceFinding(rule=rule, term=match.group(0), location=location, excerpt=_excerpt(text, match), suggestion=suggestion))  # fmt: skip
    return found


def _term_patterns(terms: Sequence[Term]) -> list[tuple[re.Pattern[str], str, str]]:
    return [(t.pattern, t.term, t.suggestion) for t in terms]


def _offered(name: str, clinic: ClinicConfig) -> bool:
    key = name.casefold()
    return any(key in o.casefold() for o in clinic.offered)


def _result(rule: str, findings: list[ComplianceFinding], detail_ok: str = "", detail_fail: str = "") -> RuleResult:  # fmt: skip
    passed = not findings and not detail_fail
    return RuleResult(rule=rule, title=RULE_TITLES[rule], passed=passed, findings=findings, detail=detail_ok if passed else (detail_fail or f"{len(findings)} finding(s)"))  # fmt: skip


def check(post: ComplianceInput, clinic: ClinicConfig, terms: Sequence[Term]) -> ComplianceReport:
    """Every rule for one post."""
    body_places = _body_fields(post.body)
    texts = [*post.fields, *body_places, *_link_texts(post.body)]
    everything = [*texts, *post.urls, *([("author box", post.author_box)] if post.author_box else []), *[("call to action", c) for c in post.ctas]]  # fmt: skip
    rules: list[RuleResult] = []
    rules.append(_result("R4", _scan("R4", everything, _term_patterns(terms)), "no Group A term anywhere"))  # fmt: skip
    class_used = bool(CLASS_WORDING.search(post.body))
    advice_given = MEDICINE_ADVICE.casefold() in " ".join(post.body.split()).casefold()
    rules.append(_result("R5", [], "no prescription medicine is mentioned" if not class_used else "the advice line follows the medicine class", "" if not class_used or advice_given else f'a prescription medicine is mentioned without "{MEDICINE_ADVICE}"'))  # fmt: skip
    rules.append(_result("R6", _scan("R6", texts, DOSES), "no dose, strength or schedule"))
    rules.append(_result("R7", [], f"{len(terms)} Group A term(s) loaded", "" if terms else "the Group A term list is empty"))  # fmt: skip
    rules.append(_result("R8", _scan("R8", [*texts, *[("call to action", c) for c in post.ctas]], [p for p in PROMOTION if p[1] == "promotional offer"]), "no promotional framing"))  # fmt: skip
    rules.append(_result("R9", _scan("R9", [*texts, *post.urls], [p for p in PROMOTION if p[1] != "promotional offer"] + SUPERLATIVES), "no testimonials, counts, success rates, before/after or superlatives"))  # fmt: skip
    rules.append(_result("R10", _scan("R10", [*texts, *([("author box", post.author_box)] if post.author_box else [])], PORTRAYAL), "the doctor is named only in the author box"))  # fmt: skip
    cta_findings = _scan("R11", [*texts, *[("call to action", c) for c in post.ctas]], CTAS)
    allowed = {c.casefold() for c in clinic.ctas}
    cta_findings += [ComplianceFinding(rule="R11", term=c, location="call to action", excerpt=c, suggestion=f'use one of: {", ".join(clinic.ctas)}') for c in post.ctas if c.casefold() not in allowed]  # fmt: skip
    rules.append(_result("R11", cta_findings, "neutral calls to action only"))
    placeholders = [ComplianceFinding(rule="R12", term=m.group(0), location=loc, excerpt=_excerpt(text, m), suggestion="never publish a placeholder: add the real value to the clinic's config") for loc, text in everything for m in PLACEHOLDER.finditer(text)]  # fmt: skip
    if not clinic.doctor.registration_number:
        missing = f"the registration number of {clinic.doctor.name} ({clinic.doctor.council}) isn't in config/clinics/{clinic.clinic_id}.yaml"  # fmt: skip
    elif not post.author_box or clinic.doctor.registration_number not in post.author_box:
        missing = "the post has no author box with the doctor's registration number"
    else:
        missing = ""
    rules.append(_result("R12", placeholders, f"author box: {post.author_box}", missing))
    marks = [(re.compile(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", re.IGNORECASE), name, f'say "{generic}" (the clinic doesn\'t offer {name})') for name, generic in TRADEMARKS.items() if not _offered(name, clinic)]  # fmt: skip
    rules.append(_result("R13", _scan("R13", everything, marks), "no treatment the clinic doesn't offer is named"))  # fmt: skip
    closing = " ".join(post.body.split()).rstrip().endswith(CLOSING_NOTE)
    rules.append(_result("R14", _scan("R14", texts, CLAIMS), "general education, ending with the consultation note", "" if closing else f'the post doesn\'t end with "{CLOSING_NOTE}"'))  # fmt: skip
    rules.append(
        _result("R15", _scan("R15", [*texts, *post.urls], PATIENTS), "no patient material")
    )
    currency = _scan("R16", texts, CURRENCY)
    named = not clinic.areas or any(re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", " ".join(t for _, t in texts), re.IGNORECASE) for a in clinic.areas)  # fmt: skip
    rules.append(_result("R16", currency, "rupees only, and the clinic's area is named", "" if named else f"none of the clinic's areas is named ({', '.join(clinic.areas)})"))  # fmt: skip
    return ComplianceReport(clinic_id=clinic.clinic_id, passed=all(r.passed for r in rules), rules=rules)  # fmt: skip


# Rules an article's own text can break (the rest is assembled at publishing: the author box,
# the closing note, the call to action).
CONTENT_RULES = ("R4", "R5", "R6", "R8", "R9", "R10", "R11", "R13", "R15", "R16")


def content_problems(title: str, headings: Sequence[str], body: str, clinic: ClinicConfig, terms: Sequence[Term]) -> list[str]:  # fmt: skip
    """What in an article's own text breaks the rules, with the rewrite: the quality gate
    fails on these, so the revision fixes them before the post reaches publishing."""
    post = ComplianceInput(fields=[("the title", title), *[(f"heading '{h[:60]}'", h) for h in headings]], body=body)  # fmt: skip
    report = check(post, clinic, terms)
    problems = []
    for result in report.rules:
        if result.passed or result.rule not in CONTENT_RULES:
            continue
        if result.rule == "R16" and not result.findings and not clinic.areas:
            continue
        for f in result.findings[:6]:
            problems.append(f"{f.rule} {RULE_TITLES[f.rule].lower()}: '{f.term}' in {f.location}: {f.suggestion}")  # fmt: skip
        if not result.findings:
            problems.append(f"{result.rule} {RULE_TITLES[result.rule].lower()}: {result.detail}")
    return problems


def writer_rules(clinic: ClinicConfig, terms: Sequence[Term]) -> list[str]:
    """The rules as the writer reads them (the brief's things to avoid)."""
    classes = sorted({t.replacement for t in terms if t.replacement})[:6]
    offered = ", ".join(clinic.treatments[:30]) or "the clinic's listed treatments"
    # fmt: off
    return [
        f"Any prescription drug name, brand or generic (e.g. {', '.join(t.term for t in terms[:8])}): where safety needs it, use the class ({'; '.join(classes) or GENERIC_CLASS}) and add: \"{MEDICINE_ADVICE}\"",
        "Doses, strengths, schedules or how-to-use instructions for any medicine.",
        "Promotion of any kind: offers, discounts, packages, deals, testimonials, patient stories, patient counts, success rates, before/after material.",
        "Superlatives and guarantees: best, top, leading, No. 1, painless, permanent, guaranteed, 100%, miracle, instant results.",
        "Describing the doctor as expert, renowned or the like; the doctor appears only in the author box.",
        'Calls to action other than "Call the clinic", "WhatsApp the clinic" or "Send an enquiry" (never "Book now", "Hurry", "Limited slots").',
        f"Treatments, services or devices the clinic doesn't offer, and trademarked treatment names it doesn't provide (say \"fat-freezing (cryolipolysis)\", not CoolSculpting). It offers: {offered}.",
        "Diagnosing, or claiming to cure or treat a specific case: general education only.",
        "Patient photos, videos or identifiable case details.",
        f"Prices in any currency but rupees (₹); name the clinic's area where natural ({', '.join(clinic.areas) or 'its city'}).",
    ]
    # fmt: on


__all__ = [
    "CLOSING_NOTE",
    "CONTENT_RULES",
    "MEDICINE_ADVICE",
    "RULE_TITLES",
    "TRADEMARKS",
    "ComplianceFinding",
    "ComplianceInput",
    "ComplianceReport",
    "RuleResult",
    "check",
    "content_problems",
    "writer_rules",
]
