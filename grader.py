"""Strict semantic evaluation math for ``dataops-gym``."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


def grade_step_details(
    state: MutableMapping[str, Any],
    action: Mapping[str, Any],
    result: MutableMapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Apply the exact per-step reward rules with no score clamping."""

    score = 0.0
    components: Dict[str, float] = {}
    confidence = float(action.get("confidence", 0.0))

    action_type = str(action.get("action_type", ""))
    if result.get("classification_correct"):
        # Detect is intentionally lower value than fix/cannot_determine.
        if action_type == "detect_issue":
            score += 0.1
            components["classification"] = 0.1
        else:
            score += 0.2
            components["classification"] = 0.2
    elif result.get("classification_incorrect"):
        score -= 0.2
        components["classification"] = -0.2

    if result.get("correct_issue_detected"):
        if action_type == "detect_issue":
            score += 0.05
            components["issue_detection"] = 0.05
        else:
            score += 0.15
            components["issue_detection"] = 0.15
    elif result.get("missed_issue"):
        score -= 0.15
        components["issue_detection"] = -0.15
    elif result.get("false_issue"):
        score -= 0.05
        components["issue_detection"] = -0.05

    if result.get("correct_fix"):
        score += 0.25
        components["decision"] = 0.25
    elif result.get("correct_cannot_determine"):
        score += 0.25
        components["decision"] = 0.25
    elif result.get("hallucinated_fix"):
        score -= 0.5
        components["decision"] = -0.5
    elif result.get("wrong_fix"):
        score -= 0.4
        components["decision"] = -0.4
    elif result.get("wrong_cannot_determine"):
        score -= 0.2
        components["decision"] = -0.2

    if result.get("passive_penalty"):
        score -= 0.05
        components["passive_penalty"] = -0.05

    if result.get("repeated_detection"):
        score -= 0.1
        components["repeated_detection_penalty"] = -0.1

    extra_mods = int(result.get("extra_fields_modified", 0))
    if extra_mods > 0:
        over = -0.05 * extra_mods
        score += over
        components["overcorrection"] = over

    if result.get("consistent_handling"):
        score += 0.2
        components["cross_record_consistency"] = 0.2
    elif result.get("inconsistent_handling"):
        score -= 0.3
        components["cross_record_consistency"] = -0.3

    is_correct = bool(
        result.get("classification_correct")
        or result.get("correct_fix")
        or result.get("correct_cannot_determine")
        or result.get("correct_issue_detected")
    )
    is_wrong = bool(
        result.get("classification_incorrect")
        or result.get("wrong_fix")
        or result.get("hallucinated_fix")
        or result.get("wrong_cannot_determine")
        or result.get("false_issue")
    )
    if confidence > 0.7 and is_correct:
        score += 0.05
        components["confidence"] = 0.05
    elif confidence > 0.7 and is_wrong:
        score -= 0.1
        components["confidence"] = -0.1

    if result.get("hallucinated_fix") and confidence > 0.8:
        score -= 0.2
        components["confident_hallucination_amplification"] = -0.2

    if result.get("resolved_detected_issue"):
        score += 0.15
        components["resolution_reward"] = 0.15

    return score, components


def grade_task_result(
    task_definition: Mapping[str, Any],
    table: Any,
    state: Optional[Mapping[str, Any]] = None,
) -> float:
    """Compute final task score in [0, 1] using required formula."""

    _ = task_definition
    _ = table
    state = state or {}
    per_record_scores = dict(state.get("per_record_scores", {}))
    n = max(1, len(per_record_scores))
    avg_record_score = sum(float(v) for v in per_record_scores.values()) / n
    normalized_record_score = (avg_record_score + 1.0) / 2.0
    normalized_record_score = max(0.0, min(1.0, normalized_record_score))

    hallucination_rate = float(state.get("hallucination_rate", 0.0))
    uncertainty_accuracy = float(state.get("uncertainty_accuracy", 0.0))
    consistency_score = float(state.get("consistency_score", 1.0))

    task_score = (
        0.5 * normalized_record_score
        + 0.2 * (1.0 - hallucination_rate)
        + 0.15 * uncertainty_accuracy
        + 0.15 * consistency_score
    )
    return max(0.0, min(1.0, task_score))


def task_failure_messages(
    task_definition: Mapping[str, Any],
    table: Any,
    state: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return lightweight failure reasons collected during stepping."""

    _ = task_definition
    _ = table
    state = state or {}
    failures = state.get("failure_logs", [])
    return [str(f.get("details", "")) for f in failures if f.get("details")]


__all__ = ["grade_step_details", "grade_task_result", "task_failure_messages"]
