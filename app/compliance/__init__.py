"""Content compliance for clinics: Google Ads restricted drug terms and India's NMC rules on
medical advertising, checked on every post of every clinic before it can become a pull
request.

- **Per clinic.** ``config/clinics/<clinic_id>.yaml`` (``ClinicConfig``) holds what differs:
  the doctor and their registration, who may approve, what the clinic actually offers, its
  areas. ``CLINIC_ID`` picks the one a deployment writes for; empty, the rules are off.
- **One term list.** The Group A list (prescription drug names, brand and generic, with the
  class wording to use instead) is a file shared with the website's own check
  (``COMPLIANCE_TERMS_FILE``).
- **Blocking.** ``check`` returns a ``ComplianceReport``: pass/fail per rule, every flagged
  term with where it is, and a compliant rewrite. Any failure blocks the pull request; there
  are no warnings.
"""

from app.compliance.config import ClinicConfig, Doctor, load_clinic
from app.compliance.rules import (
    CLOSING_NOTE,
    MEDICINE_ADVICE,
    ComplianceFinding,
    ComplianceInput,
    ComplianceReport,
    RuleResult,
    check,
    content_problems,
    neutral_wording,
    writer_rules,
)
from app.compliance.terms import Term, load_terms

__all__ = [
    "CLOSING_NOTE",
    "MEDICINE_ADVICE",
    "ClinicConfig",
    "ComplianceFinding",
    "ComplianceInput",
    "ComplianceReport",
    "Doctor",
    "RuleResult",
    "Term",
    "check",
    "content_problems",
    "load_clinic",
    "load_terms",
    "neutral_wording",
    "writer_rules",
]
