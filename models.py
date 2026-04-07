"""Shared data models for ``dataops-gym``.

This module is responsible for defining typed request, response, and domain
schemas used across task execution, inference, grading, and server layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator


class Action(BaseModel):
    """Represents a single environment action issued by an agent or client."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    action_type: Literal[
        "detect_issue",
        "fix_value",
        "cannot_determine",
        "skip",
    ] = Field(..., description="Step action type for semantic cleaning evaluation.")
    record_id: str = Field(..., description="Target record identifier.")
    field: str = Field(..., description="Target field associated with the action.")
    value: Optional[str] = Field(
        default=None,
        description="Replacement value. Required only for fix_value actions.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Action confidence score between 0.0 and 1.0.",
    )

    @root_validator(skip_on_failure=True)
    def validate_action_requirements(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce conditional field requirements for specific action types."""
        action_type = values.get("action_type")
        value = values.get("value")
        record_id = values.get("record_id")
        field = values.get("field")

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(field, str) or not field.strip():
            raise ValueError("field must be a non-empty string")
        if action_type == "fix_value" and value is None:
            raise ValueError("value must not be None when action_type is 'fix_value'")
        if action_type != "fix_value" and value is not None:
            raise ValueError("value is only allowed when action_type is 'fix_value'")

        return values


class Observation(BaseModel):
    """Represents the current observable state returned by the environment."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    dataset: Dict[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Original and current modified dataset snapshots.",
    )
    action_history: List[Dict[str, Any]] = Field(
        ...,
        description="Ordered list of actions taken in strict action schema.",
    )
    per_record_scores: Dict[str, float] = Field(
        ...,
        description="Raw per-record scores (not clamped).",
    )
    current_iteration_score: float = Field(
        ...,
        description="Current iteration aggregate score before final normalization.",
    )
    previous_iteration_score: float = Field(
        ...,
        description="Previous iteration aggregate score used for improvement rewards.",
    )
    steps_remaining: int = Field(..., description="Remaining allowed interaction steps.")


class Reward(BaseModel):
    """Represents the reward outcome associated with an environment step."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    reward: float = Field(
        ...,
        description="Numeric reward assigned to the most recent action or transition.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation for why the reward was assigned.",
    )
    components: Dict[str, float] = Field(
        ...,
        description="Breakdown of reward contributions (e.g., duplicate_removal: 0.3, penalty: -0.1)",
    )
