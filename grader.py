"""Evaluation and grading interfaces for ``dataops-gym``.

This module is responsible for validating outputs, scoring task results, and
capturing assessment metadata independently from task execution logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple


# Dense reward values are intentionally small and additive so the agent receives
# feedback for intermediate progress without requiring full task completion.
CORRECT_DUPLICATE_REMOVAL_REWARD = 0.3
CORRECT_NORMALIZATION_REWARD = 0.2
FIX_MISSING_VALUE_REWARD = 0.2
VALIDATION_SUCCESS_REWARD = 0.2
EFFICIENCY_BONUS = 0.2
RECOVERY_BONUS = 0.25
STEP_PENALTY = -0.02
PROGRESS_REWARD_SCALE = 0.3

# Penalties are split into:
# 1. a direct penalty for the current bad action, and
# 2. an escalating repetition penalty if the same mistake keeps happening.
WRONG_DELETION_PENALTY = -0.3
UNNECESSARY_ACTION_PENALTY = -0.1
NOOP_PENALTY = -0.05
DESTRUCTIVE_ACTION_PENALTY = -0.4

FIRST_REPEAT_PENALTY = -0.1
SECOND_REPEAT_PENALTY = -0.2
THIRD_OR_MORE_REPEAT_PENALTY = -0.4
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def detect_repeated_mistake(mistakes: Mapping[str, int], mistake_key: str) -> int:
    """Return how many times a mistake has already occurred before this step."""

    return int(mistakes.get(mistake_key, 0))


def track_mistake(state: MutableMapping[str, Any], mistake_key: str) -> int:
    """Update the mistake counter in state and return the new occurrence count."""

    mistakes = state.setdefault("mistakes", {})
    if not isinstance(mistakes, dict):
        raise ValueError("state['mistakes'] must be a dictionary for mistake tracking")

    current_count = int(mistakes.get(mistake_key, 0))
    new_count = current_count + 1
    mistakes[mistake_key] = new_count
    return new_count


def repeated_mistake_penalty(occurrence_count: int) -> float:
    """Return the escalating penalty for repeated mistakes."""

    if occurrence_count <= 1:
        return FIRST_REPEAT_PENALTY
    if occurrence_count == 2:
        return SECOND_REPEAT_PENALTY
    return THIRD_OR_MORE_REPEAT_PENALTY


def _to_bool(mapping: Mapping[str, Any], key: str) -> bool:
    """Normalize truthy result flags into deterministic boolean checks."""

    return bool(mapping.get(key, False))


def _mistake_key(
    action: Mapping[str, Any],
    result: Mapping[str, Any],
    fallback_key: str,
) -> str:
    """Build an action-specific mistake key with a safe fallback."""

    action_type = action.get("action_type")
    error_type = result.get("error_type", "general")

    if action_type:
        return f"{action_type}:{error_type}"
    return fallback_key


def _clamp_reward(value: float) -> float:
    """Keep rewards in the required [-1.0, 1.0] range."""

    return max(-1.0, min(1.0, round(value, 4)))


def _clamp_score(value: float) -> float:
    """Keep task-level scores in the required [0.0, 1.0] range."""

    return max(0.0, min(1.0, round(value, 4)))


def _is_missing_value(value: Any) -> bool:
    """Return whether a cell should be considered missing."""

    return value is None or value == ""


def _is_valid_email(value: str) -> bool:
    """Validate email formatting used by task graders."""

    return bool(EMAIL_PATTERN.match(value.strip()))


def _is_valid_phone(value: str) -> bool:
    """Validate phone formatting used by task graders."""

    digits = re.sub(r"\D", "", value)
    return len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))


def _needs_title_case(value: str) -> bool:
    """Return whether text still violates title-case normalization."""

    cleaned = value.strip()
    return bool(cleaned) and cleaned != cleaned.title()


def _has_duplicates(table: Iterable[Dict[str, Any]], column: str) -> bool:
    """Check whether a column contains duplicate non-empty values."""

    values = [row.get(column) for row in table if row.get(column) not in (None, "")]
    return len(values) != len(set(values))


def _table_by_row_id(table: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index a table by ``row_id`` for deterministic issue evaluation."""

    return {
        int(row["row_id"]): dict(row)
        for row in table
        if row.get("row_id") is not None
    }


def _is_issue_resolved(issue: Mapping[str, Any], table_by_row_id: Dict[int, Dict[str, Any]]) -> bool:
    """Return whether a structured hidden issue has been resolved."""

    issue_type = issue.get("type")

    if issue_type == "valid_trap":
        return True

    if issue_type in {"duplicate", "conflict"}:
        rows = issue.get("rows", [])
        return not all(row_id in table_by_row_id for row_id in rows)

    if issue_type == "missing_value":
        row = table_by_row_id.get(issue.get("row"))
        column = issue.get("column")
        return row is None or column is None or not _is_missing_value(row.get(column))

    if issue_type == "inconsistent_casing":
        column = issue.get("column")
        rows = issue.get("rows", [])
        return not any(
            row_id in table_by_row_id
            and isinstance(table_by_row_id[row_id].get(column), str)
            and _needs_title_case(str(table_by_row_id[row_id].get(column)))
            for row_id in rows
        )

    if issue_type == "invalid_format":
        row = table_by_row_id.get(issue.get("row"))
        column = issue.get("column")
        if row is None or column is None:
            return True
        value = row.get(column)
        if column == "email":
            return _is_valid_email(str(value))
        if column == "phone":
            return _is_valid_phone(str(value))
        return True

    if issue_type == "constraint_violation" and issue.get("constraint") == "unique_email":
        rows = issue.get("rows", [])
        emails = [
            table_by_row_id[row_id].get("email")
            for row_id in rows
            if row_id in table_by_row_id
        ]
        return len(emails) == len(set(emails))

    return True


def _task_check_results(
    task_definition: Mapping[str, Any],
    table: Iterable[Dict[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Build explicit pass/fail checks for final grading and validation."""

    rows = [dict(row) for row in table]
    table_by_row_id = _table_by_row_id(rows)
    expected_outcome = dict(task_definition.get("expected_outcome", {}))
    checks: list[Dict[str, Any]] = []

    expected_row_count = expected_outcome.get("expected_row_count")
    if expected_row_count is not None:
        checks.append(
            {
                "name": "expected_row_count",
                "passed": len(rows) == expected_row_count,
                "message": f"Expected exactly {expected_row_count} rows in the cleaned table.",
            }
        )

    expected_row_range = expected_outcome.get("expected_row_count_range")
    if expected_row_range is not None:
        checks.append(
            {
                "name": "expected_row_count_range",
                "passed": expected_row_range["min"] <= len(rows) <= expected_row_range["max"],
                "message": (
                    "Expected the cleaned table to contain between "
                    f"{expected_row_range['min']} and {expected_row_range['max']} rows."
                ),
            }
        )

    required_columns = expected_outcome.get(
        "required_non_null_columns", task_definition.get("required_columns", [])
    )
    if required_columns:
        checks.append(
            {
                "name": "required_non_null_columns",
                "passed": not any(
                    _is_missing_value(row.get(column))
                    for row in rows
                    for column in required_columns
                ),
                "message": "Required columns must be populated for all remaining rows.",
            }
        )

    for unique_column in expected_outcome.get("unique_by", []):
        checks.append(
            {
                "name": f"unique_by:{unique_column}",
                "passed": not _has_duplicates(rows, unique_column),
                "message": f"Values in '{unique_column}' must remain unique.",
            }
        )

    for column, rule in expected_outcome.get("normalized_columns", {}).items():
        if rule == "title_case":
            checks.append(
                {
                    "name": f"normalized_column:{column}",
                    "passed": not any(
                        isinstance(row.get(column), str)
                        and _needs_title_case(str(row.get(column)))
                        for row in rows
                    ),
                    "message": f"Column '{column}' should use a consistent title-case style.",
                }
            )

    for column, rule in expected_outcome.get("format_rules", {}).items():
        if rule == "valid_email":
            checks.append(
                {
                    "name": f"valid_email:{column}",
                    "passed": not any(
                        row.get(column) not in (None, "")
                        and not _is_valid_email(str(row.get(column)))
                        for row in rows
                    ),
                    "message": "All remaining email values must use a valid email format.",
                }
            )
        if rule == "normalized_phone":
            checks.append(
                {
                    "name": f"normalized_phone:{column}",
                    "passed": not any(
                        row.get(column) not in (None, "")
                        and not _is_valid_phone(str(row.get(column)))
                        for row in rows
                    ),
                    "message": "All remaining phone values must use a consistent valid format.",
                }
            )

    initial_rows = {}
    if state is not None:
        initial_rows = dict(state.get("initial_table_by_row_id", {}))

    for row_id in expected_outcome.get("must_preserve_valid_rows", []):
        current_row = table_by_row_id.get(row_id)
        checks.append(
            {
                "name": f"preserve_valid_row:{row_id}",
                "passed": current_row is not None and current_row == initial_rows.get(row_id),
                "message": f"Valid row {row_id} should remain logically unchanged.",
            }
        )

    for row_group in expected_outcome.get("exactly_one_of_rows", []):
        surviving = [row_id for row_id in row_group if row_id in table_by_row_id]
        checks.append(
            {
                "name": f"exactly_one_of_rows:{','.join(str(row_id) for row_id in row_group)}",
                "passed": len(surviving) == 1,
                "message": f"Exactly one of rows {row_group} should remain in the cleaned table.",
            }
        )

    for row_id in expected_outcome.get("rows_must_survive", []):
        checks.append(
            {
                "name": f"rows_must_survive:{row_id}",
                "passed": row_id in table_by_row_id,
                "message": f"Row {row_id} must still be present in the cleaned table.",
            }
        )

    for row_id in expected_outcome.get("rows_must_be_removed", []):
        checks.append(
            {
                "name": f"rows_must_be_removed:{row_id}",
                "passed": row_id not in table_by_row_id,
                "message": f"Row {row_id} should not remain in the cleaned table.",
            }
        )

    for issue in task_definition.get("hidden_issues", []):
        if issue.get("type") == "valid_trap":
            continue
        message = issue.get("description") or f"Issue '{issue.get('type')}' must be resolved."
        checks.append(
            {
                "name": f"hidden_issue:{issue.get('type')}",
                "passed": _is_issue_resolved(issue, table_by_row_id),
                "message": message,
            }
        )

    return checks


def _calculate_reward(
    state: MutableMapping[str, Any],
    action: Mapping[str, Any],
    result: MutableMapping[str, Any],
) -> float:
    """Compute the deterministic scalar reward for a single environment step."""

    reward = 0.0

    # Every step incurs a small cost so the agent is encouraged to solve the
    # task quickly instead of exploring indefinitely.
    reward += STEP_PENALTY

    # Intermediate rewards encourage the agent to make progress even when the
    # dataset is not fully clean yet.
    if _to_bool(result, "correct_duplicate_removal"):
        reward += CORRECT_DUPLICATE_REMOVAL_REWARD

    if _to_bool(result, "correct_normalization"):
        reward += CORRECT_NORMALIZATION_REWARD

    if _to_bool(result, "fixed_missing_value") or _to_bool(
        result, "fixing_missing_values"
    ):
        reward += FIX_MISSING_VALUE_REWARD

    if _to_bool(result, "validation_success"):
        reward += VALIDATION_SUCCESS_REWARD

    if _to_bool(result, "corrected_previous_mistake"):
        reward += RECOVERY_BONUS

    if _to_bool(result, "noop"):
        reward += NOOP_PENALTY

    if _to_bool(result, "destructive_action"):
        reward += DESTRUCTIVE_ACTION_PENALTY

    # Progress-based shaping provides a smoother learning signal for partial
    # improvement, even when a step does not fully resolve a visible issue.
    progress_delta = float(result.get("progress_delta", 0.0))
    progress_delta = max(0.0, min(1.0, progress_delta))
    reward += progress_delta * PROGRESS_REWARD_SCALE

    # Explicitly penalize steps that fail to improve task progress so agents do
    # not learn that random but harmless actions are equivalent to useful ones.
    if progress_delta == 0.0:
        reward -= 0.05

    # Direct penalties handle obviously harmful moves. Repetition is tracked
    # separately so the same bad behavior becomes more expensive over time.
    if _to_bool(result, "wrong_deletion"):
        reward += WRONG_DELETION_PENALTY
        mistake_key = _mistake_key(action, result, "wrong_deletion")
        occurrence_count = track_mistake(state, mistake_key)
        reward += repeated_mistake_penalty(occurrence_count)

    if _to_bool(result, "unnecessary_action"):
        reward += UNNECESSARY_ACTION_PENALTY
        mistake_key = _mistake_key(action, result, "unnecessary_action")
        occurrence_count = track_mistake(state, mistake_key)
        reward += repeated_mistake_penalty(occurrence_count)

    # Support arbitrary custom mistake keys in addition to the built-in ones.
    for mistake_key in result.get("mistake_keys", []):
        if mistake_key not in {"wrong_deletion", "unnecessary_action"}:
            occurrence_count = track_mistake(state, str(mistake_key))
            reward += repeated_mistake_penalty(occurrence_count)

    # Reward early completion only when the task finishes with steps still
    # available. This creates a simple deterministic efficiency incentive.
    if _to_bool(result, "task_completed") and int(state.get("steps_remaining", 0)) > 0:
        reward += EFFICIENCY_BONUS

    return _clamp_reward(reward)


def grade_step(
    state: MutableMapping[str, Any],
    action: Mapping[str, Any],
    result: MutableMapping[str, Any],
) -> float:
    """Compute a deterministic dense reward for a single environment step."""

    return _calculate_reward(state, action, result)


def grade_step_details(
    state: MutableMapping[str, Any],
    action: Mapping[str, Any],
    result: MutableMapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward plus a structured component breakdown for debugging."""

    previous_mistakes = {
        key: int(value)
        for key, value in state.get("mistakes", {}).items()
    }
    reward = grade_step(state, action, result)

    wrong_deletion_repeat_penalty = 0.0
    if result.get("wrong_deletion"):
        mistake_key = _mistake_key(action, result, "wrong_deletion")
        occurrence_count = int(state.get("mistakes", {}).get(mistake_key, 0))
        if occurrence_count > int(previous_mistakes.get(mistake_key, 0)):
            wrong_deletion_repeat_penalty = repeated_mistake_penalty(occurrence_count)

    unnecessary_repeat_penalty = 0.0
    if result.get("unnecessary_action"):
        mistake_key = _mistake_key(action, result, "unnecessary_action")
        occurrence_count = int(state.get("mistakes", {}).get(mistake_key, 0))
        if occurrence_count > int(previous_mistakes.get(mistake_key, 0)):
            unnecessary_repeat_penalty = repeated_mistake_penalty(occurrence_count)

    components: Dict[str, Any] = {
        "step_penalty": STEP_PENALTY,
        "duplicate_reward": (
            CORRECT_DUPLICATE_REMOVAL_REWARD
            if result.get("correct_duplicate_removal")
            else 0.0
        ),
        "normalization_reward": (
            CORRECT_NORMALIZATION_REWARD
            if result.get("correct_normalization")
            else 0.0
        ),
        "missing_value_reward": (
            FIX_MISSING_VALUE_REWARD if result.get("fixed_missing_value") else 0.0
        ),
        "validation_reward": (
            VALIDATION_SUCCESS_REWARD if result.get("validation_success") else 0.0
        ),
        "penalties": {
            "wrong_deletion": (
                WRONG_DELETION_PENALTY if result.get("wrong_deletion") else 0.0
            ),
            "unnecessary_action": (
                UNNECESSARY_ACTION_PENALTY if result.get("unnecessary_action") else 0.0
            ),
            "wrong_deletion_repeat": wrong_deletion_repeat_penalty,
            "unnecessary_action_repeat": unnecessary_repeat_penalty,
            "noop": NOOP_PENALTY if result.get("noop") else 0.0,
            "destructive_action": (
                DESTRUCTIVE_ACTION_PENALTY
                if result.get("destructive_action")
                else 0.0
            ),
        },
        "progress_reward": round(
            max(0.0, min(1.0, float(result.get("progress_delta", 0.0))))
            * PROGRESS_REWARD_SCALE,
            4,
        ),
        "recovery_bonus": (
            RECOVERY_BONUS if result.get("corrected_previous_mistake") else 0.0
        ),
        "efficiency_bonus": (
            EFFICIENCY_BONUS
            if result.get("task_completed") and int(state.get("steps_remaining", 0)) > 0
            else 0.0
        ),
    }

    if float(result.get("progress_delta", 0.0)) == 0.0:
        components["no_progress_penalty"] = -0.05

    result["reward_components"] = components
    result["reward_total"] = reward
    return reward, components


def grade_task_result(
    task_definition: Mapping[str, Any],
    table: Iterable[Dict[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
) -> float:
    """Compute a deterministic final task score between 0.0 and 1.0."""

    checks = _task_check_results(task_definition, table, state)
    if not checks:
        return 0.0
    return _clamp_score(
        sum(1.0 for check in checks if check["passed"]) / len(checks)
    )


def task_failure_messages(
    task_definition: Mapping[str, Any],
    table: Iterable[Dict[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return explicit failure messages for unresolved outcome checks."""

    return [
        str(check["message"])
        for check in _task_check_results(task_definition, table, state)
        if not bool(check["passed"])
    ]


def grade_easy_cleaning_task(
    task_definition: Mapping[str, Any],
    table: Iterable[Dict[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
) -> float:
    """Grade the easy cleaning task on a 0.0–1.0 scale."""

    return grade_task_result(task_definition, table, state)


def grade_medium_normalization_task(
    task_definition: Mapping[str, Any],
    table: Iterable[Dict[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
) -> float:
    """Grade the medium normalization task on a 0.0–1.0 scale."""

    return grade_task_result(task_definition, table, state)


def grade_hard_conflict_resolution_task(
    task_definition: Mapping[str, Any],
    table: Iterable[Dict[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
) -> float:
    """Grade the hard conflict-resolution task on a 0.0–1.0 scale."""

    return grade_task_result(task_definition, table, state)


__all__ = [
    "detect_repeated_mistake",
    "grade_step",
    "grade_step_details",
    "grade_task_result",
    "task_failure_messages",
    "grade_easy_cleaning_task",
    "grade_medium_normalization_task",
    "grade_hard_conflict_resolution_task",
    "repeated_mistake_penalty",
    "track_mistake",
]
