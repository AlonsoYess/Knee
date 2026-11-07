from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

class Login(StrictModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value):
        value = value.strip().casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Invalid email")
        return value

class ClinicalDraft(StrictModel):
    age: StrictInt | None = Field(gt=0)
    sex: Literal["F", "M"] | None
    bmi: StrictFloat | None = Field(gt=0)
    side: Literal["right", "left"] | None

class ClinicalPatch(StrictModel):
    age: StrictInt | None = Field(default=None, gt=0)
    sex: Literal["F", "M"] | None = None
    bmi: StrictFloat | None = Field(default=None, gt=0)
    side: Literal["right", "left"] | None = None

    @model_validator(mode="after")
    def at_least_one_field(self):
        if not self.model_fields_set:
            raise ValueError("At least one clinical field is required")
        return self

class CreateEvaluation(StrictModel):
    clinical: ClinicalDraft

class UpdateEvaluation(StrictModel):
    clinical: ClinicalPatch
