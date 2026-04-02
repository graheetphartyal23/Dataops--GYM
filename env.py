"""OpenEnv environment entrypoint for ``dataops-gym``.

This module is responsible for declaring top-level environment metadata,
configuration wiring, and lifecycle integration points for the OpenEnv runtime.
"""

from __future__ import annotations

from copy import deepcopy
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from grader import grade_step_details, grade_task_result, task_failure_messages
from models import Action, Observation
from task import (
    HiddenIssue,
    TaskDefinition,
    easy_cleaning_task,
    hard_conflict_resolution_task,
    medium_normalization_task,
)


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


class DataOpsEnv:
    """Deterministic multi-step data-cleaning environment for OpenEnv."""

    def __init__(self, seed: int = 0, task_name: Optional[str] = None) -> None:
        """Initialize the environment with deterministic task sampling."""

        self._seed = seed
        self._rng = random.Random(seed)
        self._task_registry: List[Tuple[str, Any]] = [
            ("easy", easy_cleaning_task),
            ("medium", medium_normalization_task),
            ("hard", hard_conflict_resolution_task),
        ]
        self._fixed_task_name = task_name
        self._global_mistake_memory: Dict[str, int] = {}
        self._state_data: Dict[str, Any] = {}

    def reset(self) -> Observation:
        """Load a random task, initialize episode state, and return an observation."""

        task_name, task_factory = self._select_task_factory()
        variant_count = max(1, int(getattr(task_factory, "variant_count", 1)))
        variant_index = self._rng.randrange(variant_count)
        task_definition = deepcopy(task_factory(variant=variant_index))
        initial_table = deepcopy(task_definition["initial_table"])
        initial_table_by_row_id = self._table_by_row_id(initial_table)

        self._state_data = {
            "seed": self._seed,
            "task_name": task_name,
            "task_variant": task_definition.get("variant_id", f"{task_name}_variant_{variant_index}"),
            "task": task_definition,
            "table": initial_table,
            "history": [],
            "mistakes": {},
            "mistake_memory": [],
            "hints": [],
            "steps_taken": 0,
            "steps_remaining": task_definition["max_steps"],
            "done": False,
            "last_reward_components": {},
            "last_info": {},
            "last_task_score": 0.0,
            "initial_issue_count": 1,
            "initial_table_by_row_id": initial_table_by_row_id,
        }
        initial_issue_count = len(self._current_issue_messages(initial_table, task_definition))
        self._state_data["initial_issue_count"] = max(1, initial_issue_count)
        return self._build_observation()

    def step(
        self, action: Action | Mapping[str, Any]
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Apply one action, score it, update state, and return a gym-style step tuple."""

        if not self._state_data:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._state_data.get("done", False):
            raise RuntimeError("Episode is finished. Call reset() before stepping again.")

        parsed_action, action_error = self._coerce_action(action)
        task_definition: TaskDefinition = self._state_data["task"]
        table_before = deepcopy(self._state_data["table"])
        issues_before = self._current_issue_messages(table_before, task_definition)

        result: Dict[str, Any] = {
            "mistake_keys": [],
            "error_type": "general",
        }

        if action_error is not None:
            parsed_action = Action(action_type="noop")
            result["noop"] = True
            result["unnecessary_action"] = True
            result["error_type"] = "invalid_action"
            result["mistake_keys"].append("invalid_action:general")
            history_entry = f"invalid_action({action_error})"
        else:
            history_entry = self._apply_action(parsed_action, result)

        self._state_data["history"].append(history_entry)
        self._state_data["steps_taken"] += 1
        self._state_data["steps_remaining"] = max(
            0, task_definition["max_steps"] - self._state_data["steps_taken"]
        )

        table_after = deepcopy(self._state_data["table"])
        issues_after = self._current_issue_messages(table_after, task_definition)
        self._populate_result_signals(
            parsed_action,
            table_before,
            table_after,
            issues_before,
            issues_after,
            result,
        )

        reward, components = grade_step_details(
            self._state_data, parsed_action.model_dump(), result
        )
        self._record_mistake_memory(parsed_action, result)
        self._update_hints(result, issues_after)

        done = not issues_after or self._state_data["steps_remaining"] <= 0
        self._state_data["done"] = done
        task_score = grade_task_result(
            task_definition, self._state_data["table"], self._state_data
        )
        self._state_data["last_task_score"] = task_score

        observation = self._build_observation()
        info = {
            "task_name": self._state_data["task_name"],
            "task_variant": self._state_data["task_variant"],
            "difficulty": task_definition["difficulty"],
            "reward_components": components,
            "mistakes": deepcopy(self._state_data["mistakes"]),
            "hints": list(self._state_data["hints"]),
            "issues_remaining": len(issues_after),
            "done_reason": "resolved" if not issues_after else "max_steps" if done else None,
            "task_score": task_score,
            "result": deepcopy(result),
        }
        self._state_data["last_reward_components"] = deepcopy(components)
        self._state_data["last_info"] = deepcopy(info)
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return a deep copy of the internal environment state."""

        return deepcopy(self._state_data)

    def close(self) -> None:
        """Release environment state for callers using explicit lifecycle cleanup."""

        self._state_data = {}

    def _select_task_factory(self) -> Tuple[str, Any]:
        """Pick the configured task factory deterministically."""

        if self._fixed_task_name is None:
            return self._rng.choice(self._task_registry)

        for task_name, task_factory in self._task_registry:
            if self._fixed_task_name in {task_name, task_factory.__name__}:
                return task_name, task_factory

        raise ValueError(f"Unknown task_name: {self._fixed_task_name}")

    def _coerce_action(
        self, action: Action | Mapping[str, Any]
    ) -> Tuple[Optional[Action], Optional[str]]:
        """Convert user input into an ``Action`` model without raising outward."""

        if isinstance(action, Action):
            return action, None

        try:
            return Action(**dict(action)), None
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return None, str(exc)

    def _apply_action(self, action: Action, result: MutableMapping[str, Any]) -> str:
        """Apply a single action to the current table and capture side effects."""

        if action.action_type == "noop":
            result["noop"] = True
            result["mistake_keys"].append(f"{action.action_type}:noop")
            return self._format_history(action)

        if action.action_type == "remove_duplicate":
            self._remove_duplicate(action, result)
            return self._format_history(action)

        if action.action_type == "delete_row":
            self._delete_row(action, result)
            return self._format_history(action)

        if action.action_type == "fill_missing":
            self._fill_missing(action, result)
            return self._format_history(action)

        if action.action_type == "normalize_column":
            self._normalize_column(action, result)
            return self._format_history(action)

        if action.action_type == "validate":
            return self._format_history(action)

        result["unnecessary_action"] = True
        result["error_type"] = "unsupported_action"
        result["mistake_keys"].append(f"{action.action_type}:unsupported_action")
        return self._format_history(action)

    def _remove_duplicate(
        self, action: Action, result: MutableMapping[str, Any]
    ) -> None:
        """Remove a duplicate row when the target belongs to a duplicate issue."""

        duplicate_groups = [
            issue
            for issue in self._state_data["task"]["hidden_issues"]
            if issue["type"] == "duplicate" and self._is_issue_unresolved(issue, self._state_data["table"])
        ]
        if not duplicate_groups:
            result["unnecessary_action"] = True
            result["error_type"] = "no_duplicate_available"
            return

        candidate_rows = set(duplicate_groups[0].get("rows", []))
        target_row_id = action.row_id or max(candidate_rows)

        if target_row_id not in candidate_rows:
            result["unnecessary_action"] = True
            result["error_type"] = "invalid_duplicate_target"
            return

        removed = self._remove_row_by_id(target_row_id)
        if not removed:
            result["unnecessary_action"] = True
            result["error_type"] = "missing_row"

    def _delete_row(self, action: Action, result: MutableMapping[str, Any]) -> None:
        """Delete a row and mark destructive behavior when the target is unsafe."""

        target_row = self._get_row_by_id(action.row_id)
        if target_row is None:
            result["unnecessary_action"] = True
            result["error_type"] = "missing_row"
            return

        if self._row_is_protected(action.row_id):
            result["wrong_deletion"] = True
            result["destructive_action"] = True
            result["error_type"] = "protected_row"
            result["mistake_keys"].append(f"{action.action_type}:protected_row")
        elif not self._row_belongs_to_removable_issue(action.row_id):
            result["wrong_deletion"] = True
            result["destructive_action"] = True
            result["error_type"] = "wrong_deletion"
            result["mistake_keys"].append(f"{action.action_type}:wrong_deletion")

        self._remove_row_by_id(action.row_id)

    def _fill_missing(self, action: Action, result: MutableMapping[str, Any]) -> None:
        """Fill a missing field on the target row or the first matching missing cell."""

        target_row = self._resolve_missing_target_row(action.row_id, action.column)
        if target_row is None or action.column is None:
            result["unnecessary_action"] = True
            result["error_type"] = "missing_target"
            return

        if not self._is_missing_value(target_row.get(action.column)):
            result["unnecessary_action"] = True
            result["error_type"] = "cell_not_missing"
            return

        target_row[action.column] = action.value

    def _normalize_column(self, action: Action, result: MutableMapping[str, Any]) -> None:
        """Normalize a supported column using deterministic, minimal edits."""

        if action.column is None:
            result["unnecessary_action"] = True
            result["error_type"] = "missing_column"
            return

        changed_rows = 0
        for row in self._state_data["table"]:
            original = row.get(action.column)
            normalized = self._normalized_value(action.column, original)
            if normalized is None or normalized == original:
                continue

            # Keep trap rows stable unless the value is actually invalid.
            if self._row_is_protected(row.get("row_id")) and self._value_is_valid(
                action.column, original
            ):
                continue

            row[action.column] = normalized
            changed_rows += 1

        if changed_rows == 0:
            result["unnecessary_action"] = True
            result["error_type"] = "no_normalization_needed"

    def _populate_result_signals(
        self,
        action: Action,
        table_before: List[Dict[str, Any]],
        table_after: List[Dict[str, Any]],
        issues_before: List[str],
        issues_after: List[str],
        result: MutableMapping[str, Any],
    ) -> None:
        """Derive reward signals from before/after state transitions."""

        task_definition: TaskDefinition = self._state_data["task"]
        hidden_before = self._issue_type_counts(table_before, task_definition)
        hidden_after = self._issue_type_counts(table_after, task_definition)

        if hidden_after.get("duplicate", 0) < hidden_before.get("duplicate", 0):
            result["correct_duplicate_removal"] = True

        if hidden_after.get("missing_value", 0) < hidden_before.get("missing_value", 0):
            result["fixed_missing_value"] = True

        normalization_before = hidden_before.get("inconsistent_casing", 0) + hidden_before.get(
            "invalid_format", 0
        )
        normalization_after = hidden_after.get("inconsistent_casing", 0) + hidden_after.get(
            "invalid_format", 0
        )
        if (
            action.action_type == "normalize_column"
            and normalization_after < normalization_before
        ):
            result["correct_normalization"] = True

        if action.action_type == "validate" and not issues_after:
            result["validation_success"] = True
            result["task_completed"] = True

        if not issues_after:
            result["task_completed"] = True

        issue_delta = max(0, len(issues_before) - len(issues_after))
        result["progress_delta"] = round(
            issue_delta / float(self._state_data["initial_issue_count"]),
            4,
        )

        if issue_delta > 0 and any(self._state_data["mistakes"].values()):
            result["corrected_previous_mistake"] = True

        if action.action_type == "noop" and issues_after:
            result["unnecessary_action"] = True
            result["error_type"] = result.get("error_type", "noop")

    def _build_observation(self) -> Observation:
        """Construct the typed observation returned to callers."""

        task_definition: TaskDefinition = self._state_data["task"]
        issue_messages = self._current_issue_messages(self._state_data["table"], task_definition)
        progress = self._compute_progress(issue_messages)
        return Observation(
            goal=task_definition["goal"],
            table=deepcopy(self._state_data["table"]),
            issues=issue_messages,
            history=list(self._state_data["history"]),
            mistakes=deepcopy(self._state_data["mistakes"]),
            hints=list(self._state_data["hints"]),
            progress=progress,
            steps_remaining=int(self._state_data["steps_remaining"]),
        )

    def _compute_progress(self, issue_messages: List[str]) -> float:
        """Estimate progress from the current unresolved issue count."""

        baseline = float(self._state_data["initial_issue_count"])
        remaining = min(len(issue_messages), self._state_data["initial_issue_count"])
        resolved_fraction = 1.0 - (remaining / baseline)
        return round(max(0.0, min(1.0, resolved_fraction)), 4)

    def _current_issue_messages(
        self, table: List[Dict[str, Any]], task_definition: TaskDefinition
    ) -> List[str]:
        """Return unresolved issue descriptions plus validation-rule failures."""

        messages: List[str] = []
        for issue in task_definition["hidden_issues"]:
            if self._is_issue_unresolved(issue, table):
                description = issue.get("description")
                if description:
                    messages.append(description)

        messages.extend(self._validation_failures(table, task_definition))
        return messages

    def _validation_failures(
        self, table: List[Dict[str, Any]], task_definition: TaskDefinition
    ) -> List[str]:
        """Evaluate rule-based outcome constraints beyond the hidden issue list."""

        return task_failure_messages(task_definition, table, self._state_data)

    def _issue_type_counts(
        self, table: List[Dict[str, Any]], task_definition: TaskDefinition
    ) -> Dict[str, int]:
        """Count unresolved hidden issues by type."""

        counts: Dict[str, int] = {}
        for issue in task_definition["hidden_issues"]:
            if self._is_issue_unresolved(issue, table):
                issue_type = issue["type"]
                counts[issue_type] = counts.get(issue_type, 0) + 1
        return counts

    def _is_issue_unresolved(self, issue: HiddenIssue, table: List[Dict[str, Any]]) -> bool:
        """Determine whether a hidden issue is still unresolved."""

        issue_type = issue["type"]
        table_by_row_id = self._table_by_row_id(table)

        if issue_type == "valid_trap":
            return False

        if issue_type in {"duplicate", "conflict"}:
            rows = issue.get("rows", [])
            return all(row_id in table_by_row_id for row_id in rows)

        if issue_type == "missing_value":
            row = table_by_row_id.get(issue.get("row"))
            column = issue.get("column")
            return row is not None and column is not None and self._is_missing_value(row.get(column))

        if issue_type == "inconsistent_casing":
            column = issue.get("column")
            return any(
                row_id in table_by_row_id
                and self._needs_title_case(str(table_by_row_id[row_id].get(column, "")))
                for row_id in issue.get("rows", [])
            )

        if issue_type == "invalid_format":
            row = table_by_row_id.get(issue.get("row"))
            column = issue.get("column")
            return row is not None and column is not None and not self._value_is_valid(
                column, row.get(column)
            )

        if issue_type == "constraint_violation" and issue.get("constraint") == "unique_email":
            rows = issue.get("rows", [])
            emails = [
                table_by_row_id[row_id].get("email")
                for row_id in rows
                if row_id in table_by_row_id
            ]
            return len(emails) != len(set(emails))

        return False

    def _update_hints(self, result: Mapping[str, Any], issues_after: List[str]) -> None:
        """Add deterministic hints when the agent stalls or accumulates mistakes."""

        if not issues_after:
            return

        global_wrong_deletion_count = sum(
            count
            for key, count in self._global_mistake_memory.items()
            if key == "wrong_deletion" or key.endswith(":wrong_deletion")
        )
        if global_wrong_deletion_count >= 3:
            hint = (
                "You are repeatedly deleting valid rows. Try resolving issues "
                "instead of deleting."
            )
            if hint not in self._state_data["hints"]:
                self._state_data["hints"].append(hint)

        total_mistakes = sum(self._state_data["mistakes"].values())
        should_hint = bool(result.get("unnecessary_action")) or bool(
            result.get("wrong_deletion")
        ) or total_mistakes >= 2 or float(result.get("progress_delta", 0.0)) == 0.0

        if not should_hint:
            return

        next_hint = self._build_hint(issues_after[0])
        if next_hint not in self._state_data["hints"]:
            self._state_data["hints"].append(next_hint)

    def _build_hint(self, issue_message: str) -> str:
        """Map unresolved issue descriptions to small, actionable hints."""

        lowered = issue_message.lower()
        if "duplicate" in lowered:
            return "Look for rows that describe the same entity and keep only one representative record."
        if "missing" in lowered:
            return "A required field is still empty. Fill the missing value instead of deleting the row."
        if "email" in lowered and "format" in lowered:
            return "Normalize only the invalid email values; valid addresses should be preserved."
        if "phone" in lowered:
            return "Repair only phone values that are actually malformed."
        if "title-case" in lowered or "casing" in lowered:
            return "Normalize text columns to a consistent title-case style."
        if "unchanged" in lowered:
            return "Some unusual-looking rows are valid traps and should be preserved."
        return "Focus on the first unresolved issue and prefer minimal corrective actions."

    def _record_mistake_memory(
        self, action: Action, result: Mapping[str, Any]
    ) -> None:
        """Persist mistake events so hinting can look at prior failures."""

        for key, count in self._state_data["mistakes"].items():
            if count <= 0:
                continue
            if action.action_id:
                memory_entry = f"{action.action_id}:{key}:{count}"
            else:
                memory_entry = f"{action.action_type}:{key}:{count}"
            if memory_entry not in self._state_data["mistake_memory"]:
                self._state_data["mistake_memory"].append(memory_entry)

            self._global_mistake_memory[key] = (
                self._global_mistake_memory.get(key, 0) + 1
            )
            category_key = key.split(":")[-1]
            self._global_mistake_memory[category_key] = (
                self._global_mistake_memory.get(category_key, 0) + 1
            )

        if result.get("destructive_action"):
            entry = f"{action.action_type}:destructive_action"
            if entry not in self._state_data["mistake_memory"]:
                self._state_data["mistake_memory"].append(entry)

    def _resolve_missing_target_row(
        self, row_id: Optional[int], column: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Choose the requested row or the first matching missing-value row."""

        if row_id is not None:
            return self._get_row_by_id(row_id)

        if column is None:
            return None

        for row in self._state_data["table"]:
            if self._is_missing_value(row.get(column)):
                return row
        return None

    def _normalized_value(self, column: str, value: Any) -> Any:
        """Return a normalized value for supported columns."""

        if not isinstance(value, str):
            return value

        if column in {"name", "city"}:
            return value.title()

        if column == "email" and not self._is_valid_email(value):
            normalized = value.strip().lower()
            normalized = normalized.replace("[at]", "@").replace(" at ", "@")
            if "@" not in normalized and normalized.endswith(".example.com"):
                normalized = normalized.replace(".example.com", "@example.com", 1)
            if "@" in normalized and "." not in normalized.split("@", 1)[1]:
                normalized = normalized + ".com"
            return normalized

        if column == "phone" and not self._is_valid_phone(value):
            digits = re.sub(r"\D", "", value)
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            if len(digits) == 10:
                return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
        return value

    def _value_is_valid(self, column: str, value: Any) -> bool:
        """Validate known column types used by the tasks."""

        if value is None:
            return False
        if column == "email":
            return self._is_valid_email(str(value))
        if column == "phone":
            return self._is_valid_phone(str(value))
        if column in {"name", "city"}:
            return not self._needs_title_case(str(value))
        return True

    def _is_valid_email(self, value: str) -> bool:
        """Return whether the supplied email string looks valid."""

        return bool(EMAIL_PATTERN.match(value.strip()))

    def _is_valid_phone(self, value: str) -> bool:
        """Return whether the supplied phone value is valid for this environment."""

        digits = re.sub(r"\D", "", value)
        return len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))

    def _needs_title_case(self, value: str) -> bool:
        """Detect whether a string still needs title-case normalization."""

        cleaned = value.strip()
        return bool(cleaned) and cleaned != cleaned.title()

    def _has_missing_required_values(
        self, table: Iterable[Dict[str, Any]], required_columns: Iterable[str]
    ) -> bool:
        """Check whether any required field remains missing."""

        for row in table:
            for column in required_columns:
                if self._is_missing_value(row.get(column)):
                    return True
        return False

    def _has_duplicates(self, table: Iterable[Dict[str, Any]], column: str) -> bool:
        """Check whether a column contains duplicate non-empty values."""

        values = [row.get(column) for row in table if row.get(column) not in (None, "")]
        return len(values) != len(set(values))

    def _column_has_invalid_email(
        self, table: Iterable[Dict[str, Any]], column: str
    ) -> bool:
        """Check whether any remaining email value is invalid."""

        return any(
            row.get(column) not in (None, "") and not self._is_valid_email(str(row.get(column)))
            for row in table
        )

    def _column_has_invalid_phone(
        self, table: Iterable[Dict[str, Any]], column: str
    ) -> bool:
        """Check whether any remaining phone value is invalid."""

        return any(
            row.get(column) not in (None, "") and not self._is_valid_phone(str(row.get(column)))
            for row in table
        )

    def _column_needs_title_case(
        self, table: Iterable[Dict[str, Any]], column: str
    ) -> bool:
        """Check whether any remaining value still violates title-case normalization."""

        return any(
            isinstance(row.get(column), str) and self._needs_title_case(str(row.get(column)))
            for row in table
        )

    def _row_has_changed_from_initial(
        self, row_id: int, current_table: List[Dict[str, Any]]
    ) -> bool:
        """Check whether a protected row has changed relative to the task start."""

        current_row = self._table_by_row_id(current_table).get(row_id)
        initial_row = self._state_data["initial_table_by_row_id"].get(row_id)
        if current_row is None or initial_row is None:
            return True
        return current_row != initial_row

    def _row_is_protected(self, row_id: Optional[int]) -> bool:
        """Return whether a row is marked as a valid trap in the current task."""

        if row_id is None:
            return False
        for issue in self._state_data["task"]["hidden_issues"]:
            if issue["type"] == "valid_trap" and issue.get("row") == row_id:
                return True
        return False

    def _row_belongs_to_removable_issue(self, row_id: Optional[int]) -> bool:
        """Return whether deleting a row could plausibly resolve a structural issue."""

        if row_id is None:
            return False
        for issue in self._state_data["task"]["hidden_issues"]:
            if issue["type"] in {"duplicate", "conflict", "constraint_violation"} and row_id in issue.get(
                "rows", []
            ):
                return True
        return False

    def _remove_row_by_id(self, row_id: Optional[int]) -> bool:
        """Remove a row by id and report whether a row was deleted."""

        if row_id is None:
            return False
        table = self._state_data["table"]
        for index, row in enumerate(table):
            if row.get("row_id") == row_id:
                del table[index]
                return True
        return False

    def _get_row_by_id(self, row_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """Return a mutable row reference by id."""

        if row_id is None:
            return None
        for row in self._state_data["table"]:
            if row.get("row_id") == row_id:
                return row
        return None

    def _table_by_row_id(self, table: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Index a table by row id."""

        return {
            int(row["row_id"]): deepcopy(row)
            for row in table
            if row.get("row_id") is not None
        }

    def _is_missing_value(self, value: Any) -> bool:
        """Return whether a cell should be treated as missing."""

        return value is None or value == ""

    def _format_history(self, action: Action) -> str:
        """Return a compact history entry for the applied action."""

        details = []
        if action.row_id is not None:
            details.append(f"row_id={action.row_id}")
        if action.column is not None:
            details.append(f"column={action.column}")
        if action.value is not None:
            details.append(f"value={action.value}")
        detail_text = ", ".join(details)
        return f"{action.action_type}({detail_text})" if detail_text else action.action_type


class DataOpsGymEnv(DataOpsEnv):
    """Compatibility wrapper matching the configured OpenEnv entrypoint."""

    pass

__all__ = ["DataOpsEnv", "DataOpsGymEnv"]
