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

class SourceReferenceSubmission(StrictModel):
    preparationRevisionId: str = Field(min_length=1, max_length=128)
    confirmedSide: Literal['right', 'left']
    centerResolved: bool
    centerXY: list[StrictInt] | None = Field(min_length=2, max_length=2)
    originalAndHalfInspected: bool
    outcomeBlinded: bool
    note: str = Field(min_length=35, max_length=2000)

class QualityReviewSubmission(StrictModel):
    preparationRevisionId: str = Field(min_length=1, max_length=128)
    confirmedSide: Literal['right', 'left']
    decision: Literal['accepted', 'rejected']
    confirmedCheckIds: list[str]
    reason: str | None = Field(min_length=1, max_length=2000)
    observation: dict | None

    @model_validator(mode='after')
    def valid_decision(self):
        if len(set(self.confirmedCheckIds)) != len(self.confirmedCheckIds):
            raise ValueError('Repeated checks')
        if self.decision == 'rejected' and (self.reason is None or not self.reason.strip()):
            raise ValueError('A rejection reason is required')
        if self.decision == 'accepted' and self.observation is None:
            raise ValueError('A complete quality observation is required')
        return self

class PreparationRetry(StrictModel):
    pass
