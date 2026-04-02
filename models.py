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

    action_id: Optional[str] = Field(
        default=None,
        description="A unique identifier for the action instance, useful for tracking repeated actions and mistake patterns.",
    )
    action_type: Literal[
        "remove_duplicate",
        "fill_missing",
        "normalize_column",
        "delete_row",
        "validate",
        "noop",
    ] = Field(
        ...,
        description="The type of data-cleaning action to apply in the environment.",
    )
    column: Optional[str] = Field(
        default=None,
        description="Optional target column name associated with the action.",
    )
    row_id: Optional[int] = Field(
        default=None,
        description="Optional target row identifier associated with the action.",
    )
    value: Optional[str] = Field(
        default=None,
        description="Optional value payload used by the action when needed.",
    )

    @root_validator(skip_on_failure=True)
    def validate_action_requirements(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce conditional field requirements for specific action types."""
        action_type = values.get("action_type")
        column = values.get("column")
        row_id = values.get("row_id")
        value = values.get("value")

        if action_type == "delete_row" and row_id is None:
            raise ValueError("row_id must not be None when action_type is 'delete_row'")
        if action_type == "normalize_column" and column is None:
            raise ValueError("column must not be None when action_type is 'normalize_column'")
        if action_type == "fill_missing" and (column is None or value is None):
            raise ValueError(
                "column and value must not be None when action_type is 'fill_missing'"
            )

        return values


class Observation(BaseModel):
    """Represents the current observable state returned by the environment."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    goal: str = Field(
        ...,
        description="A natural language description of the task objective the agent must achieve.",
    )
    table: List[Dict[str, Any]] = Field(
        ...,
        description="JSON-serializable table snapshot represented as a list of row dictionaries.",
    )
    issues: List[str] = Field(
        ...,
        description="Detected data-quality issues currently present in the table.",
    )
    history: List[str] = Field(
        ...,
        description="Ordered list of previously applied actions or events.",
    )
    mistakes: Dict[str, int] = Field(
        ...,
        description="Counts of mistake categories accumulated during the episode.",
    )
    hints: List[str] = Field(
        ...,
        description="Optional guidance hints available to the agent or client.",
    )
    progress: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A normalized estimate (0.0–1.0) of how much of the task is completed.",
    )
    steps_remaining: int = Field(
        ...,
        description="Number of steps remaining before the episode terminates.",
    )


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
