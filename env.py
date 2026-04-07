"""Semantic data-cleaning evaluation environment."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from grader import grade_step_details, grade_task_result
from models import Action, Observation
from task import easy_cleaning_task, hard_conflict_resolution_task, medium_normalization_task


class DataOpsEnv:
    """Step-based semantic evaluator with strict action protocol."""

    def __init__(self, seed: int = 0, task_name: Optional[str] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_registry: List[Tuple[str, Any]] = [
            ("easy", easy_cleaning_task),
            ("medium", medium_normalization_task),
            ("hard", hard_conflict_resolution_task),
        ]
        self._fixed_task_name = task_name
        self._state_data: Dict[str, Any] = {}

    def reset(self) -> Observation:
        task_name, task_factory = self._select_task_factory()
        variant_count = max(1, int(getattr(task_factory, "variant_count", 1)))
        task_definition = deepcopy(task_factory(variant=self._rng.randrange(variant_count)))
        initial_table = deepcopy(task_definition["initial_table"])
        self._state_data = {
            "seed": self._seed,
            "task_name": task_name,
            "task_variant": task_definition.get("variant_id", task_name),
            "task": task_definition,
            "dataset_original": initial_table,
            "dataset_modified": deepcopy(initial_table),
            "action_history": [],
            "per_record_scores": {},
            "current_iteration_score": 0.0,
            "previous_iteration_score": 0.0,
            "failure_logs": [],
            "steps_taken": 0,
            "steps_remaining": task_definition["max_steps"],
            "done": False,
            "totals": {
                "total_fixes": 0,
                "hallucinated_fixes": 0,
                "total_cannot_determine": 0,
                "correct_cannot_determine": 0,
                "total_related_cases": 0,
                "consistent_decisions": 0,
            },
            "related_decisions": {},
            "detected_unresolved_issues": {},
            "detected_issues": {},
            "hallucination_rate": 0.0,
            "uncertainty_accuracy": 0.0,
            "consistency_score": 1.0,
        }
        return self._build_observation()

    def step(self, action: Action | Mapping[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if not self._state_data:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._state_data["done"]:
            raise RuntimeError("Episode is finished. Call reset() before stepping again.")

        parsed_action = action if isinstance(action, Action) else Action(**dict(action))
        result = self._evaluate_action(parsed_action)

        self._state_data["action_history"].append(parsed_action.model_dump())
        self._state_data["steps_taken"] += 1
        self._state_data["steps_remaining"] = max(
            0, self._state_data["task"]["max_steps"] - self._state_data["steps_taken"]
        )

        self._state_data["previous_iteration_score"] = float(
            self._state_data["current_iteration_score"]
        )
        reward, reward_components = grade_step_details(
            self._state_data, parsed_action.model_dump(), result
        )
        rid = parsed_action.record_id
        self._state_data["per_record_scores"][rid] = float(
            self._state_data["per_record_scores"].get(rid, 0.0)
        ) + reward
        self._state_data["current_iteration_score"] = sum(
            float(v) for v in self._state_data["per_record_scores"].values()
        )
        prev = self._state_data["previous_iteration_score"]
        curr = self._state_data["current_iteration_score"]
        if curr > prev:
            reward += 0.1
            reward_components["iteration_improvement"] = 0.1
        elif curr < prev:
            reward -= 0.1
            reward_components["iteration_improvement"] = -0.1

        self._update_metrics()
        task_score = grade_task_result(
            self._state_data["task"], self._state_data["dataset_modified"], self._state_data
        )

        done = self._state_data["steps_remaining"] <= 0
        self._state_data["done"] = done
        info = {
            "actions_taken": deepcopy(self._state_data["action_history"]),
            "updated_dataset": deepcopy(self._state_data["dataset_modified"]),
            "per_record_scores": deepcopy(self._state_data["per_record_scores"]),
            "final_task_score": task_score,
            "metrics": {
                "hallucination_rate": self._state_data["hallucination_rate"],
                "uncertainty_accuracy": self._state_data["uncertainty_accuracy"],
                "consistency_score": self._state_data["consistency_score"],
            },
            "failure_logs": deepcopy(self._state_data["failure_logs"]),
            "reward_components": reward_components,
            "result": result,
        }
        return self._build_observation(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return deepcopy(self._state_data)

    def close(self) -> None:
        self._state_data = {}

    def _select_task_factory(self) -> Tuple[str, Any]:
        """Pick the configured task factory deterministically."""

        if self._fixed_task_name is None:
            return self._rng.choice(self._task_registry)

        for task_name, task_factory in self._task_registry:
            if self._fixed_task_name in {task_name, task_factory.__name__}:
                return task_name, task_factory

        raise ValueError(f"Unknown task_name: {self._fixed_task_name}")

    def _evaluate_action(self, action: Action) -> Dict[str, Any]:
        table = self._state_data["dataset_modified"]
        issue = self._matching_issue(action.record_id, action.field)
        issue_key = self._issue_key(issue)
        result: Dict[str, Any] = {"extra_fields_modified": 0}
        self._apply_related_consistency(action, issue, result)
        self._apply_follow_up_requirement(action, issue_key, result)

        if action.action_type == "skip":
            if issue is not None:
                result["missed_issue"] = True
                result["passive_penalty"] = True
                if issue_key is not None:
                    self._state_data["detected_unresolved_issues"][issue_key] = True
                self._append_failure(action, "missed_issue", "Issue exists but action was skip.")
            return result

        if action.action_type == "detect_issue":
            if issue is not None:
                result["classification_correct"] = True
                result["correct_issue_detected"] = True
                result["passive_penalty"] = True
                if issue_key is not None:
                    if issue_key in self._state_data["detected_issues"]:
                        result["repeated_detection"] = True
                    self._state_data["detected_issues"][issue_key] = True
                    self._state_data["detected_unresolved_issues"][issue_key] = True
            else:
                result["classification_incorrect"] = True
                result["false_issue"] = True
            return result

        if action.action_type == "cannot_determine":
            self._state_data["totals"]["total_cannot_determine"] += 1
            if issue is None:
                result["wrong_cannot_determine"] = True
                self._append_failure(
                    action, "wrong_fix", "cannot_determine used without any supporting issue."
                )
            elif issue.get("fixable", True) is False:
                result["correct_cannot_determine"] = True
                self._state_data["totals"]["correct_cannot_determine"] += 1
                if issue_key is not None:
                    self._state_data["detected_unresolved_issues"].pop(issue_key, None)
                    if issue_key in self._state_data["detected_issues"]:
                        result["resolved_detected_issue"] = True
            else:
                result["wrong_cannot_determine"] = True
                self._append_failure(
                    action, "wrong_fix", "cannot_determine used when evidence was sufficient."
                )
            return result

        # fix_value
        self._state_data["totals"]["total_fixes"] += 1
        if issue is None:
            related_issue_count = self._count_issues_for_record(action.record_id)
            if related_issue_count > 0:
                result["extra_fields_modified"] += 1

        row = self._find_record(action.record_id, table)
        if row is None or action.field not in row:
            result["hallucinated_fix"] = True
            self._state_data["totals"]["hallucinated_fixes"] += 1
            self._append_failure(action, "hallucination", "Attempted fix with no evidence.")
            return result

        if issue is None:
            result["hallucinated_fix"] = True
            self._state_data["totals"]["hallucinated_fixes"] += 1
            self._append_failure(action, "hallucination", "Field had no target issue.")
            return result

        if self._issue_resolved(issue, table):
            result["hallucinated_fix"] = True
            self._state_data["totals"]["hallucinated_fixes"] += 1
            self._append_failure(action, "hallucination", "Field is already correct.")
            return result

        old_value = row.get(action.field)
        before_row = deepcopy(row)
        row[action.field] = action.value
        if self._introduces_inconsistency(row, action.field, table):
            result["hallucinated_fix"] = True
            self._state_data["totals"]["hallucinated_fixes"] += 1
            row[action.field] = old_value
            self._append_failure(
                action, "hallucination", "Fix introduces cross-record or temporal inconsistency."
            )
            return result

        if self.validate_fix(issue, before_row, row, table):
            result["correct_fix"] = True
            result["classification_correct"] = True
            if issue_key is not None:
                if issue_key in self._state_data["detected_issues"]:
                    result["resolved_detected_issue"] = True
                self._state_data["detected_unresolved_issues"].pop(issue_key, None)
        else:
            row[action.field] = old_value
            result["wrong_fix"] = True
            self._append_failure(action, "wrong_fix", "Fix does not resolve the identified issue.")
        return result

    def _apply_follow_up_requirement(
        self, action: Action, issue_key: Optional[str], result: Dict[str, Any]
    ) -> None:
        unresolved = self._state_data.get("detected_unresolved_issues", {})
        if not unresolved:
            return

        # Follow-up action types are fix/cannot_determine against a detected issue.
        is_follow_up = (
            action.action_type in {"fix_value", "cannot_determine"}
            and issue_key is not None
            and issue_key in unresolved
        )
        if not is_follow_up:
            result["passive_penalty"] = True

    def _apply_related_consistency(
        self, action: Action, issue: Optional[Dict[str, Any]], result: Dict[str, Any]
    ) -> None:
        if issue is None:
            return
        issue_type = issue.get("type")
        if issue_type not in {"duplicate", "conflict"}:
            return

        rows = issue.get("rows", [])
        if not rows:
            return
        key = f"{issue_type}:{','.join(str(v) for v in sorted(rows))}"
        self._state_data["totals"]["total_related_cases"] += 1
        seen = self._state_data["related_decisions"]
        decision = action.action_type
        if key not in seen:
            seen[key] = decision
            result["consistent_handling"] = True
            self._state_data["totals"]["consistent_decisions"] += 1
            return
        if seen[key] == decision:
            result["consistent_handling"] = True
            self._state_data["totals"]["consistent_decisions"] += 1
        else:
            result["inconsistent_handling"] = True
            self._append_failure(
                action, "inconsistency", "Related records were handled inconsistently."
            )

    def _matching_issue(self, record_id: str, field: str) -> Optional[Dict[str, Any]]:
        rid = self._parse_record_id(record_id)
        for issue in self._state_data["task"]["hidden_issues"]:
            issue_type = issue.get("type")
            if issue_type == "missing_value" and issue.get("row") == rid and issue.get("column") == field:
                return issue
            if issue_type == "invalid_format" and issue.get("row") == rid and issue.get("column") == field:
                return issue
            if issue_type == "inconsistent_casing" and field == issue.get("column") and rid in issue.get("rows", []):
                return issue
            if (
                issue_type in {"duplicate", "conflict", "constraint_violation"}
                and (field in {"row", "record"} or field == issue.get("field"))
                and rid in issue.get("rows", [])
            ):
                ambiguous = issue_type in {"conflict", "constraint_violation"}
                c = dict(issue)
                c["ambiguous"] = ambiguous
                return c
        return None

    def _issue_resolved(self, issue: Mapping[str, Any], table: List[Dict[str, Any]]) -> bool:
        if issue.get("type") in {"duplicate", "conflict", "constraint_violation"}:
            return False
        rid = int(issue.get("row", -1))
        field = issue.get("column")
        row = self._find_record(str(rid), table)
        if row is None:
            return True
        if issue.get("type") == "missing_value":
            return row.get(field) not in (None, "", "unknown", "9999")
        if issue.get("type") == "invalid_format":
            value = str(row.get(field, ""))
            if field == "email":
                return "@" in value and "." in value.split("@")[-1]
            if field == "phone":
                digits = "".join(ch for ch in value if ch.isdigit())
                return len(digits) in {10, 11}
            if field in {"start_date", "end_date"}:
                start = row.get("start_date")
                end = row.get("end_date")
                return not (start and end and str(end) < str(start))
        return row.get(field) not in (None, "", "unknown", "9999")

    def validate_fix(
        self,
        issue: Mapping[str, Any],
        before_row: Mapping[str, Any],
        after_row: Mapping[str, Any],
        table: List[Dict[str, Any]],
    ) -> bool:
        """Ground-truth validator for semantic fixes."""

        issue_type = str(issue.get("type", ""))
        field = str(issue.get("column") or issue.get("field") or "")

        if field and before_row.get(field) == after_row.get(field):
            return False

        if field == "age":
            try:
                age = int(after_row.get("age"))
            except Exception:
                return False
            if age < 0 or age > 120:
                return False

        if issue_type == "missing_value":
            return after_row.get(field) not in (None, "", "unknown", "9999")

        if issue_type == "invalid_format":
            value = str(after_row.get(field, ""))
            if field == "email":
                return "@" in value and "." in value.split("@")[-1]
            if field == "phone":
                digits = "".join(ch for ch in value if ch.isdigit())
                return len(digits) in {10, 11}
            if field in {"start_date", "end_date"}:
                start = after_row.get("start_date")
                end = after_row.get("end_date")
                return not (start and end and str(end) < str(start))
            return value not in ("", "unknown", "9999")

        if issue_type == "inconsistent_casing":
            value = after_row.get(field)
            return isinstance(value, str) and value == value.title()

        if issue_type in {"duplicate", "conflict", "constraint_violation"}:
            return False

        return not self._introduces_inconsistency(dict(after_row), field, table) and self._issue_resolved(
            issue, table
        )

    def _count_issues_for_record(self, record_id: str) -> int:
        rid = self._parse_record_id(record_id)
        count = 0
        for issue in self._state_data["task"]["hidden_issues"]:
            if issue.get("row") == rid:
                count += 1
                continue
            if rid in issue.get("rows", []):
                count += 1
        return count

    def _issue_key(self, issue: Optional[Dict[str, Any]]) -> Optional[str]:
        if issue is None:
            return None
        issue_type = issue.get("type", "unknown")
        if "row" in issue and "column" in issue:
            return f"{issue_type}:row={issue.get('row')}:col={issue.get('column')}"
        if "rows" in issue:
            rows = ",".join(str(v) for v in sorted(issue.get("rows", [])))
            field = issue.get("field", "record")
            return f"{issue_type}:rows={rows}:field={field}"
        return f"{issue_type}:generic"

    def _introduces_inconsistency(
        self, row: Dict[str, Any], field: str, table: List[Dict[str, Any]]
    ) -> bool:
        # Unique email consistency check across records.
        if field == "email":
            email = row.get("email")
            if email not in (None, ""):
                duplicates = [
                    r for r in table
                    if r is not row and str(r.get("email", "")).strip() == str(email).strip()
                ]
                if duplicates:
                    return True

        # Temporal consistency check where both fields are present.
        if field in {"start_date", "end_date"}:
            start = row.get("start_date")
            end = row.get("end_date")
            if start and end and str(end) < str(start):
                return True

        return False

    def _build_observation(self) -> Observation:
        return Observation(
            dataset={
                "original": deepcopy(self._state_data["dataset_original"]),
                "modified": deepcopy(self._state_data["dataset_modified"]),
            },
            action_history=deepcopy(self._state_data["action_history"]),
            per_record_scores=deepcopy(self._state_data["per_record_scores"]),
            current_iteration_score=float(self._state_data["current_iteration_score"]),
            previous_iteration_score=float(self._state_data["previous_iteration_score"]),
            steps_remaining=int(self._state_data["steps_remaining"]),
        )

    def _update_metrics(self) -> None:
        totals = self._state_data["totals"]
        total_fixes = int(totals["total_fixes"])
        self._state_data["hallucination_rate"] = (
            0.0 if total_fixes == 0 else float(totals["hallucinated_fixes"]) / total_fixes
        )
        total_cd = int(totals["total_cannot_determine"])
        self._state_data["uncertainty_accuracy"] = (
            0.0 if total_cd == 0 else float(totals["correct_cannot_determine"]) / total_cd
        )
        total_related = int(totals["total_related_cases"])
        self._state_data["consistency_score"] = (
            1.0 if total_related == 0 else float(totals["consistent_decisions"]) / total_related
        )

    def _parse_record_id(self, record_id: str) -> int:
        digits = "".join(ch for ch in str(record_id) if ch.isdigit())
        return int(digits) if digits else -1

    def _find_record(self, record_id: str, table: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rid = self._parse_record_id(record_id)
        for row in table:
            if int(row.get("row_id", -1)) == rid:
                return row
        return None

    def _append_failure(self, action: Action, error_type: str, details: str) -> None:
        mapped = error_type
        if error_type == "wrong_fix":
            mapped = "wrong_fix"
        self._state_data["failure_logs"].append(
            {
                "record_id": action.record_id,
                "error_type": mapped,
                "details": details,
                "confidence": float(action.confidence),
            }
        )


class DataOpsGymEnv(DataOpsEnv):
    """Compatibility wrapper matching the configured OpenEnv entrypoint."""

    pass

__all__ = ["DataOpsEnv", "DataOpsGymEnv"]
