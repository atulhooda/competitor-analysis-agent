"""A clinic's compliance configuration: ``config/clinics/<clinic_id>.yaml``.

Only facts the clinic has confirmed go here. A missing registration number stays missing:
the author box can't be completed, and every post is blocked until it is added. Nothing is
ever filled in with a placeholder."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.errors import ConfigurationError

DEFAULT_CTAS = ("Call the clinic", "WhatsApp the clinic", "Send an enquiry")


class Doctor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    qualification: str = Field(min_length=1, max_length=200)
    council: str = Field(
        min_length=1, max_length=120, description="e.g. Maharashtra Medical Council"
    )
    registration_number: str | None = Field(default=None, max_length=60)

    @field_validator("registration_number")
    @classmethod
    def _no_placeholder(cls, value: str | None) -> str | None:
        """An unknown number is absent, never a placeholder like [PENDING] or TBD."""
        if value is None:
            return None
        text = value.strip()
        if not text or text.strip("[]()").strip().upper() in {"PENDING", "TBD", "TBC", "NA", "N/A", "XXX", "TO BE CONFIRMED"}:  # fmt: skip
            return None
        return text

    @property
    def author_box(self) -> str | None:
        """Name, qualification, council and registration, or None while any is missing."""
        if not self.registration_number:
            return None
        return (
            f"{self.name}, {self.qualification}, {self.council} Reg. No. {self.registration_number}"
        )


class ClinicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinic_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=120)
    doctor: Doctor
    approvers: list[str] = Field(default_factory=list, description="Who may approve publishing; the doctor when empty")  # fmt: skip
    treatments: list[str] = Field(default_factory=list, description="Treatments the clinic actually offers")  # fmt: skip
    services: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list, description="Devices the clinic actually uses (brand names allowed only here)")  # fmt: skip
    areas: list[str] = Field(default_factory=list, description="Local area names a post names (at least one)")  # fmt: skip
    ctas: list[str] = Field(default_factory=lambda: list(DEFAULT_CTAS), description="The only call-to-action wording allowed")  # fmt: skip

    @property
    def allowed_approvers(self) -> tuple[str, ...]:
        return tuple(self.approvers or [self.doctor.name])

    def approves(self, approver: str) -> bool:
        """Whether ``approver`` is one of the clinic's doctors (case and spacing aside)."""
        key = " ".join(approver.split()).casefold()
        return any(" ".join(a.split()).casefold() == key for a in self.allowed_approvers)

    @property
    def offered(self) -> tuple[str, ...]:
        """Everything the clinic offers, by the names it uses."""
        return (*self.treatments, *self.services, *self.devices)


class _ClinicFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinic: ClinicConfig


def load_clinic(directory: Path, clinic_id: str) -> ClinicConfig:
    path = directory / f"{clinic_id}.yaml"
    if not path.is_file():
        raise ConfigurationError(f"No clinic configuration {path}: CLINIC_ID={clinic_id} needs one (see config/clinics/)")  # fmt: skip
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        clinic = _ClinicFile.model_validate(data).clinic
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid clinic configuration {path}:\n{exc}") from exc
    if clinic.clinic_id != clinic_id:
        raise ConfigurationError(f"{path}: clinic_id is {clinic.clinic_id!r}, not {clinic_id!r}")
    return clinic


__all__ = ["DEFAULT_CTAS", "ClinicConfig", "Doctor", "load_clinic"]
