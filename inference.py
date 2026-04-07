"""Inference entrypoints for ``dataops-gym``.

This runner keeps the hackathon-required OpenAI-compatible model interface, but
adds a stronger local planner so baseline behavior is still competitive and
reproducible when the model is weak, unavailable, or partially aligned.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
import re
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

from env import DataOpsEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BROWSERGYM_BENCHMARK", "dataops-env")
TASK_NAME = os.getenv("BROWSERGYM_TASK_NAME", "all")
TASK_ORDER = ["easy", "medium", "hard"]
MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 160
MODEL_RETRIES = 2
FALLBACK_ACTION = "skip(record_id='0', field='record', confidence=0.0)"
ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:\-]\s*", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
IDENTIFIER_COLUMNS = ("customer_id", "vendor_id", "partner_id")
POLICY_CACHE_PATH = os.getenv("POLICY_CACHE_PATH", ".dataops_policy_cache.json")
POLICY_CACHE_VERSION = 1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a data-cleaning environment.
    Reply with exactly one action string and nothing else.

    Only choose from the candidate actions provided by the user prompt.
    Favor actions that remove visible issues quickly and avoid actions that were
    already blocked because they caused errors or no progress.
    Use single quotes for string arguments.
    """
).strip()


class PolicyMemory:
    """Persistent lightweight experience cache used across episodes and runs."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "version": POLICY_CACHE_VERSION,
            "states": {},
            "patterns": {},
        }
        self._load()

    def _load(self) -> None:
        """Load cache from disk if it exists and is compatible."""

        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if int(payload.get("version", 0)) != POLICY_CACHE_VERSION:
            return
        self.data = payload

    def save(self) -> None:
        """Persist the current cache contents to disk."""

        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
        os.replace(temp_path, self.path)

    def _bucket(self, bucket_name: str, key: str) -> Dict[str, Any]:
        """Return the mutable bucket for an exact state or a problem pattern."""

        return self.data.setdefault(bucket_name, {}).setdefault(key, {"actions": {}})

    def _action_stats(self, bucket_name: str, key: str, action_text: str) -> Dict[str, Any]:
        """Return mutable stats for an action within a memory bucket."""

        actions = self._bucket(bucket_name, key).setdefault("actions", {})
        return actions.setdefault(
            action_text,
            {
                "attempts": 0,
                "successes": 0,
                "progresses": 0,
                "failures": 0,
                "cumulative_reward": 0.0,
                "last_error": None,
            },
        )

    def update(
        self,
        *,
        state_key: str,
        pattern_key: str,
        action_text: str,
        reward: float,
        progress_delta: float,
        error: Optional[str],
        done: bool,
        task_score: float,
    ) -> None:
        """Record one step outcome for exact-state and problem-pattern memory."""

        was_success = bool(done and task_score >= 0.95 and error is None)
        made_progress = bool(progress_delta > 0.0 or reward > 0.0)
        was_failure = bool(error is not None or (progress_delta == 0.0 and reward <= 0.0))

        for bucket_name, key in (("states", state_key), ("patterns", pattern_key)):
            stats = self._action_stats(bucket_name, key, action_text)
            stats["attempts"] += 1
            stats["cumulative_reward"] = round(
                float(stats["cumulative_reward"]) + float(reward),
                4,
            )
            stats["last_error"] = error
            if was_success:
                stats["successes"] += 1
            elif made_progress:
                stats["progresses"] += 1
            if was_failure:
                stats["failures"] += 1

    def _combined_stats(self, state_key: str, pattern_key: str, action_text: str) -> Dict[str, float]:
        """Merge exact-state and pattern-level stats into one weighted view."""

        combined = {
            "attempts": 0.0,
            "successes": 0.0,
            "progresses": 0.0,
            "failures": 0.0,
            "cumulative_reward": 0.0,
        }
        for bucket_name, key, weight in (
            ("states", state_key, 1.0),
            ("patterns", pattern_key, 0.5),
        ):
            stats = self.data.get(bucket_name, {}).get(key, {}).get("actions", {}).get(action_text)
            if not isinstance(stats, dict):
                continue
            for field in combined:
                combined[field] += float(stats.get(field, 0.0)) * weight
        return combined

    def score_action(self, state_key: str, pattern_key: str, action_text: str) -> float:
        """Score a candidate action using remembered prior outcomes."""

        stats = self._combined_stats(state_key, pattern_key, action_text)
        attempts = max(1.0, stats["attempts"])
        average_reward = stats["cumulative_reward"] / attempts
        return round(
            (stats["successes"] * 3.0)
            + (stats["progresses"] * 1.25)
            + average_reward
            - (stats["failures"] * 2.0),
            4,
        )

    def blocked_actions(self, state_key: str, pattern_key: str) -> set[str]:
        """Return actions that repeatedly failed for the same state or pattern."""

        blocked: set[str] = set()
        for bucket_name, key in (("states", state_key), ("patterns", pattern_key)):
            actions = self.data.get(bucket_name, {}).get(key, {}).get("actions", {})
            for action_text, stats in actions.items():
                attempts = int(stats.get("attempts", 0))
                failures = int(stats.get("failures", 0))
                successes = int(stats.get("successes", 0))
                progresses = int(stats.get("progresses", 0))
                if attempts >= 2 and failures >= attempts and successes == 0 and progresses == 0:
                    blocked.add(action_text)
        return blocked


def log_start(task: str, env: str, model: str) -> None:
    """Emit the required episode start line."""

    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Emit the required per-step line."""

    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], final_score: float) -> None:
    """Emit the required episode end line."""

    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_text} final_score={final_score:.4f}",
        flush=True,
    )


def build_history_lines(history: Sequence[str]) -> str:
    """Render the last few steps for the model prompt."""

    if not history:
        return "None"
    return "\n".join(history[-5:])


def _stable_json(value: Any) -> str:
    """Serialize a value deterministically for memory key generation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash_key(payload: Mapping[str, Any]) -> str:
    """Build a compact deterministic memory key."""

    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def _normalize_issue_text(issue: str) -> str:
    """Remove row-specific numbers so pattern memory generalizes better."""

    lowered = issue.lower().strip()
    return re.sub(r"\d+", "#", lowered)


def _table_summary(table: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract a compact problem summary from the visible table state."""

    present_columns = sorted({key for row in table for key in row.keys() if key != "row_id"})
    missing_counts: Dict[str, int] = {}
    for column in present_columns:
        count = sum(1 for row in table if _is_missing(row.get(column)))
        if count > 0:
            missing_counts[column] = count

    duplicate_counts: Dict[str, int] = {}
    for column in list(IDENTIFIER_COLUMNS) + ["email"]:
        values = [row.get(column) for row in table if row.get(column) not in (None, "")]
        if values and len(values) != len(set(values)):
            duplicate_counts[column] = len(values) - len(set(values))

    return {
        "row_count": len(table),
        "present_columns": present_columns,
        "missing_counts": missing_counts,
        "duplicate_counts": duplicate_counts,
        "invalid_email_count": sum(
            1
            for row in table
            if row.get("email") not in (None, "") and not _is_valid_email(row.get("email"))
        ),
        "invalid_phone_count": sum(
            1
            for row in table
            if row.get("phone") not in (None, "") and not _is_valid_phone(row.get("phone"))
        ),
        "title_case_columns": sorted(
            column
            for column in ("name", "city")
            if any(_needs_title_case(row.get(column)) for row in table)
        ),
    }


def build_memory_keys(
    task_name: str,
    task_variant: str,
    goal: str,
    observation: Mapping[str, Any],
) -> Tuple[str, str]:
    """Build exact-state and generalized problem-pattern keys."""

    dataset = observation.get("dataset", {}) if isinstance(observation, dict) else {}
    table = list(dataset.get("modified", []))
    normalized_issues = [
        f"rows={len(table)}",
        f"history={len(observation.get('action_history', []))}",
        f"iter={observation.get('current_iteration_score', 0.0)}",
    ]
    state_key = _hash_key(
        {
            "task_name": task_name,
            "task_variant": task_variant,
            "goal": goal,
            "table": [
                {key: row.get(key) for key in sorted(row.keys())}
                for row in sorted(table, key=lambda row: int(row.get("row_id", 0)))
            ],
            "issues": sorted(normalized_issues),
        }
    )
    pattern_key = _hash_key(
        {
            "task_name": task_name,
            "goal": goal,
            "summary": _table_summary(table),
            "issues": sorted(normalized_issues),
        }
    )
    return state_key, pattern_key


def _is_missing(value: Any) -> bool:
    """Return whether a value is missing."""

    return value is None or value == ""


def _needs_title_case(value: Any) -> bool:
    """Return whether a string still needs title-case normalization."""

    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    return bool(cleaned) and cleaned != cleaned.title()


def _is_valid_email(value: Any) -> bool:
    """Return whether an email-like string is valid."""

    return isinstance(value, str) and bool(EMAIL_PATTERN.match(value.strip()))


def _is_valid_phone(value: Any) -> bool:
    """Return whether a phone-like string is valid."""

    if not isinstance(value, str):
        return False
    digits = re.sub(r"\D", "", value)
    return len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))


def _slugify_text(value: str) -> str:
    """Convert free text into a stable email-local-part fragment."""

    lowered = re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".")
    return lowered or "record"


def _infer_email(row: Mapping[str, Any]) -> str:
    """Infer a safe placeholder email from row context."""

    if isinstance(row.get("name"), str) and row["name"].strip():
        return f"{_slugify_text(row['name'])}@example.com"
    for key in IDENTIFIER_COLUMNS:
        if row.get(key):
            return f"{str(row[key]).lower()}@example.com"
    return f"row{row.get('row_id', 'unknown')}@example.com"


def _infer_name(row: Mapping[str, Any]) -> str:
    """Infer a readable name when a name field is missing."""

    email = row.get("email")
    if isinstance(email, str) and "@" in email:
        return email.split("@", 1)[0].replace(".", " ").title()
    for key in IDENTIFIER_COLUMNS:
        if row.get(key):
            return str(row[key]).replace("_", " ").title()
    return "Unknown Record"


def _infer_city(table: Sequence[Mapping[str, Any]]) -> str:
    """Infer a plausible city using the mode of visible values."""

    candidates = [
        str(row.get("city")).title()
        for row in table
        if isinstance(row.get("city"), str) and row.get("city")
    ]
    if not candidates:
        return "Seattle"
    return Counter(candidates).most_common(1)[0][0]


def _infer_fill_value(
    row: Mapping[str, Any],
    column: str,
    table: Sequence[Mapping[str, Any]],
) -> str:
    """Infer a deterministic fill value from local table context."""

    for key in IDENTIFIER_COLUMNS:
        identifier = row.get(key)
        if not identifier:
            continue
        for candidate in table:
            if candidate.get("row_id") == row.get("row_id"):
                continue
            if candidate.get(key) == identifier and not _is_missing(candidate.get(column)):
                return str(candidate[column])

    if column == "email":
        return _infer_email(row)
    if column == "city":
        return _infer_city(table)
    if column == "phone":
        return "555-555-0100"
    if column == "status":
        return "active"
    if column == "name":
        return _infer_name(row)
    return "resolved"


def _row_signature(row: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Create a comparable row signature excluding runtime row identifiers."""

    return tuple(sorted((key, value) for key, value in row.items() if key != "row_id"))


def _build_action_string(payload: Mapping[str, Any]) -> str:
    """Reconstruct a normalized action string for logging and filtering."""

    action_type = str(payload["action_type"])
    args: List[str] = []
    for key in ("record_id", "field", "value", "confidence"):
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if isinstance(value, str):
            args.append(f"{key}='{value}'")
        else:
            args.append(f"{key}={value}")
    return f"{action_type}({', '.join(args)})" if args else f"{action_type}()"


def build_action_string(payload: Dict[str, Any]) -> str:
    """Backward-compatible public wrapper around action string generation."""

    return _build_action_string(payload)


def parse_model_action(response_text: str) -> str:
    """Extract a single action string from model output."""

    if not response_text:
        return FALLBACK_ACTION

    for raw_line in response_text.splitlines():
        line = ACTION_PREFIX_RE.sub("", raw_line.strip())
        if "(" in line and line.endswith(")"):
            return re.sub(r"\s+", " ", line)

    compact = ACTION_PREFIX_RE.sub("", response_text.strip())
    match = re.search(r"[a-zA-Z_]+\s*\(.*\)", compact)
    if match:
        return re.sub(r"\s+", " ", match.group(0))

    return FALLBACK_ACTION


def action_string_to_payload(action_str: str, step_number: int) -> Tuple[str, Dict[str, Any]]:
    """Convert a model action string into an environment action payload."""

    try:
        expression = ast.parse(action_str, mode="eval").body
    except SyntaxError:
        return FALLBACK_ACTION, {"action_type": "skip", "record_id": "0", "field": "record", "confidence": 0.0}

    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return FALLBACK_ACTION, {"action_type": "skip", "record_id": "0", "field": "record", "confidence": 0.0}

    allowed_actions = {
        "detect_issue",
        "fix_value",
        "cannot_determine",
        "skip",
    }
    action_type = expression.func.id
    if action_type not in allowed_actions:
        return FALLBACK_ACTION, {"action_type": "skip", "record_id": "0", "field": "record", "confidence": 0.0}

    payload: Dict[str, Any] = {
        "action_type": action_type,
    }
    try:
        for keyword in expression.keywords:
            if keyword.arg is None:
                continue
            payload[keyword.arg] = ast.literal_eval(keyword.value)
    except (SyntaxError, ValueError, TypeError):
        return FALLBACK_ACTION, {"action_type": "skip", "record_id": "0", "field": "record", "confidence": 0.0}

    payload.setdefault("record_id", "0")
    payload.setdefault("field", "record")
    payload.setdefault("confidence", 0.6 if action_type != "skip" else 0.0)

    return _build_action_string(payload), payload


def create_client() -> Optional[OpenAI]:
    """Create an OpenAI-compatible client when credentials look real."""

    if HF_TOKEN in {None, "", "local-test", "test", "dummy"}:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _extract_response_text(content: Any) -> str:
    """Normalize OpenAI response content into plain text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def _table_preview(table: Sequence[Mapping[str, Any]], limit: int = 6) -> str:
    """Render a compact table preview for prompting."""

    preview_lines: List[str] = []
    for row in table[:limit]:
        summary = ", ".join(
            f"{key}={value}"
            for key, value in row.items()
            if key in {"row_id", "name", "city", "email", "phone", "status", "customer_id", "vendor_id", "partner_id", "age", "start_date", "end_date"}
        )
        preview_lines.append(f"- {summary}")
    return "\n".join(preview_lines) if preview_lines else "- None"


def build_user_prompt(
    step: int,
    goal: str,
    observation: Dict[str, Any],
    history: Sequence[str],
    last_error: Optional[str],
    candidate_actions: Sequence[str],
    blocked_actions: Sequence[str],
) -> str:
    """Construct a compact prompt that constrains the model to useful actions."""

    dataset = observation.get("dataset", {})
    modified = dataset.get("modified", [])
    candidates_text = "\n".join(f"- {action}" for action in candidate_actions)
    blocked_text = "\n".join(f"- {action}" for action in blocked_actions[:5]) if blocked_actions else "- None"

    return textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Steps remaining: {observation.get("steps_remaining")}
        Current iteration score: {observation.get("current_iteration_score")}
        Previous iteration score: {observation.get("previous_iteration_score")}
        Per-record scores: {observation.get("per_record_scores")}
        Table preview:
        {_table_preview(modified)}
        Recent history:
        {build_history_lines(history)}
        Last action error: {last_error or "null"}
        Blocked actions:
        {blocked_text}

        Choose exactly one action from this candidate list:
        {candidates_text}
        """
    ).strip()


def _prefer_action(
    candidates: Sequence[Dict[str, Any]],
    blocked_actions: set[str],
) -> Dict[str, Any]:
    """Return the first candidate action that is not blocked."""

    for candidate in candidates:
        action_text = _build_action_string(candidate)
        if action_text not in blocked_actions:
            return dict(candidate)
    return {"action_type": "skip", "record_id": "0", "field": "record", "confidence": 0.0}


def _record_id(row: Mapping[str, Any]) -> str:
    rid = row.get("row_id")
    return str(rid) if rid is not None else "0"


def _issue_like_candidates(table: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Generate issue detection/fix candidates for new semantic action schema."""

    actions: List[Dict[str, Any]] = []
    for row in table:
        rid = _record_id(row)
        for field, value in row.items():
            if field == "row_id":
                continue
            if _is_missing(value) or str(value).strip().lower() in {"unknown", "9999"}:
                actions.append(
                    {"action_type": "detect_issue", "record_id": rid, "field": field, "confidence": 0.85}
                )
                actions.append(
                    {
                        "action_type": "fix_value",
                        "record_id": rid,
                        "field": field,
                        "value": _infer_fill_value(row, field, table),
                        "confidence": 0.75,
                    }
                )
            elif field == "email" and not _is_valid_email(value):
                fixed = str(value).replace("[at]", "@").replace(" at ", "@").replace(" ", "")
                if "@" in fixed and "." not in fixed.split("@")[-1]:
                    fixed += ".com"
                actions.append({"action_type": "detect_issue", "record_id": rid, "field": field, "confidence": 0.85})
                actions.append({"action_type": "fix_value", "record_id": rid, "field": field, "value": fixed, "confidence": 0.8})
            elif field == "phone" and not _is_valid_phone(value):
                digits = re.sub(r"\D", "", str(value))
                if len(digits) == 10:
                    fixed = f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
                    actions.append({"action_type": "detect_issue", "record_id": rid, "field": field, "confidence": 0.8})
                    actions.append({"action_type": "fix_value", "record_id": rid, "field": field, "value": fixed, "confidence": 0.75})
            elif field in {"start_date", "end_date"}:
                start = row.get("start_date")
                end = row.get("end_date")
                if start and end and str(end) < str(start):
                    actions.append({"action_type": "detect_issue", "record_id": rid, "field": field, "confidence": 0.8})
                    actions.append({"action_type": "cannot_determine", "record_id": rid, "field": field, "confidence": 0.7})
            elif field == "age":
                try:
                    age = int(value)
                except Exception:
                    age = -1
                if age < 0 or age > 120:
                    actions.append({"action_type": "detect_issue", "record_id": rid, "field": field, "confidence": 0.9})
                    actions.append({"action_type": "cannot_determine", "record_id": rid, "field": field, "confidence": 0.8})
    return actions


def _detected_keys_from_history(action_history: Sequence[Mapping[str, Any]]) -> set[str]:
    """Extract previously detected issue keys from observation history."""

    keys: set[str] = set()
    for action in action_history:
        if action.get("action_type") != "detect_issue":
            continue
        keys.add(f"{action.get('record_id')}::{action.get('field')}")
    return keys


def propose_candidate_actions(
    observation: Mapping[str, Any],
    blocked_actions: set[str],
) -> List[Dict[str, Any]]:
    """Generate ranked candidate actions from visible table state."""

    dataset = observation.get("dataset", {}) if isinstance(observation, dict) else {}
    table = list(dataset.get("modified", []))
    detected_keys = _detected_keys_from_history(observation.get("action_history", []))
    raw_candidates = _issue_like_candidates(table)
    candidates: List[Dict[str, Any]] = []
    for candidate in raw_candidates:
        if candidate.get("action_type") == "detect_issue":
            key = f"{candidate.get('record_id')}::{candidate.get('field')}"
            # Detect once; then prefer follow-up actions.
            if key in detected_keys:
                continue
        candidates.append(candidate)
    candidates += [
        {"action_type": "skip", "record_id": "0", "field": "record", "confidence": 0.0}
    ]

    unique_candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        action_text = _build_action_string(candidate)
        if action_text in seen:
            continue
        seen.add(action_text)
        unique_candidates.append(candidate)

    preferred = _prefer_action(unique_candidates, blocked_actions)
    preferred_text = _build_action_string(preferred)
    ordered = [preferred] + [
        candidate
        for candidate in unique_candidates
        if _build_action_string(candidate) != preferred_text
    ]
    return ordered[:12]


def _order_candidates_with_memory(
    candidates: Sequence[Dict[str, Any]],
    memory: PolicyMemory,
    state_key: str,
    pattern_key: str,
    recent_history: Sequence[str],
) -> List[Dict[str, Any]]:
    """Re-rank candidates using persistent cross-episode memory."""

    scored = []
    recent_action_counts = Counter()
    for item in recent_history[-5:]:
        try:
            parsed = item.split(" action=", 1)[1].split(" reward=", 1)[0].strip()
            if parsed:
                recent_action_counts[parsed] += 1
        except Exception:
            continue

    for index, candidate in enumerate(candidates):
        action_text = _build_action_string(candidate)
        repeat_penalty = recent_action_counts.get(action_text, 0) * 2.0
        scored.append(
            (
                -memory.score_action(state_key, pattern_key, action_text) + repeat_penalty,
                index,
                dict(candidate),
            )
        )
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored]


def model_action(
    client: Optional[OpenAI],
    model_name: str,
    step: int,
    goal: str,
    observation: Dict[str, Any],
    history: Sequence[str],
    last_error: Optional[str],
    candidate_actions: Sequence[str],
    blocked_actions: Sequence[str],
) -> Optional[str]:
    """Ask the model to choose among pre-computed candidate actions."""

    if client is None:
        return None

    prompt = build_user_prompt(
        step=step,
        goal=goal,
        observation=observation,
        history=history,
        last_error=last_error,
        candidate_actions=candidate_actions,
        blocked_actions=blocked_actions,
    )
    current_prompt = prompt
    candidate_set = set(candidate_actions)
    for _ in range(MODEL_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": current_prompt}],
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = _extract_response_text(completion.choices[0].message.content)
            action_text = parse_model_action(response_text)
            if action_text in candidate_set and action_text not in set(blocked_actions):
                return action_text
            current_prompt = (
                prompt
                + "\n\nYour previous answer was invalid or blocked. Choose exactly one action from the candidate list."
            )
        except Exception:  # noqa: BLE001
            return None
    return None


def choose_action(
    client: Optional[OpenAI],
    memory: PolicyMemory,
    task_name: str,
    task_variant: str,
    observation: Dict[str, Any],
    goal: str,
    step_number: int,
    history: Sequence[str],
    last_error: Optional[str],
    blocked_actions: set[str],
) -> Tuple[str, Dict[str, Any], str, str, str]:
    """Choose the next action using a heuristic planner with optional model arbitration."""

    state_key, pattern_key = build_memory_keys(task_name, task_variant, goal, observation)
    memory_blocked = memory.blocked_actions(state_key, pattern_key)
    combined_blocked = set(blocked_actions) | set(memory_blocked)
    candidates = propose_candidate_actions(observation, combined_blocked)
    candidates = _order_candidates_with_memory(
        candidates, memory, state_key, pattern_key, history
    )
    heuristic_candidate = candidates[0]
    heuristic_text = _build_action_string(heuristic_candidate)
    candidate_texts = [_build_action_string(candidate) for candidate in candidates]

    model_text = model_action(
        client=client,
        model_name=MODEL_NAME,
        step=step_number,
        goal=goal,
        observation=observation,
        history=history,
        last_error=last_error,
        candidate_actions=candidate_texts,
        blocked_actions=sorted(combined_blocked),
    )

    if model_text not in candidate_texts:
        model_text = None
    chosen_text = model_text or heuristic_text
    normalized_text, payload = action_string_to_payload(chosen_text, step_number)
    if normalized_text in combined_blocked:
        normalized_text, payload = action_string_to_payload(heuristic_text, step_number)
        return normalized_text, payload, "heuristic", state_key, pattern_key
    return normalized_text, payload, "model" if model_text else "heuristic", state_key, pattern_key


def run_episode(
    client: Optional[OpenAI],
    memory: PolicyMemory,
    task_name: str,
    seed: int,
) -> float:
    """Run one deterministic task episode and return its final task score."""

    env = DataOpsEnv(seed=seed, task_name=task_name)
    rewards: List[float] = []
    history: List[str] = []
    blocked_actions: set[str] = set()
    steps_taken = 0
    success = False
    last_error: Optional[str] = None
    final_score = 0.0
    task_variant = "unknown"
    action_repeat_counts: Dict[str, int] = defaultdict(int)
    no_change_counts: Dict[str, int] = defaultdict(int)

    try:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        observation_model = env.reset()
        observation = observation_model.model_dump()
        task_variant = str(env.state().get("task_variant", "unknown"))

        for step_number in range(1, MAX_STEPS + 1):
            action_text, action_payload, action_source, state_key, pattern_key = choose_action(
                client=client,
                memory=memory,
                task_name=task_name,
                task_variant=task_variant,
                observation=observation,
                goal=str(env.state().get("task", {}).get("goal", "")),
                step_number=step_number,
                history=history,
                last_error=last_error,
                blocked_actions=blocked_actions,
            )

            try:
                before_dataset = observation.get("dataset", {}) if isinstance(observation, dict) else {}
                before_modified = before_dataset.get("modified", [])
                observation_model, reward, done, info = env.step(action_payload)
                observation = observation_model.model_dump()
                after_dataset = observation.get("dataset", {}) if isinstance(observation, dict) else {}
                after_modified = after_dataset.get("modified", [])
                result = info.get("result", {})
                curr_iter = float(observation.get("current_iteration_score", 0.0))
                prev_iter = float(observation.get("previous_iteration_score", 0.0))
                progress_delta = max(0.0, curr_iter - prev_iter)
                error_value = "step_error" if (
                    result.get("wrong_fix")
                    or result.get("hallucinated_fix")
                    or result.get("wrong_cannot_determine")
                    or result.get("classification_incorrect")
                ) else None
                final_score = float(info.get("final_task_score", 0.0))
                if error_value == "general":
                    error_value = None
                memory.update(
                    state_key=state_key,
                    pattern_key=pattern_key,
                    action_text=action_text,
                    reward=reward,
                    progress_delta=progress_delta,
                    error=error_value,
                    done=done,
                    task_score=final_score,
                )
                if error_value or progress_delta == 0.0 or reward <= 0.0:
                    blocked_actions.add(action_text)
                action_repeat_counts[action_text] += 1
                if action_repeat_counts[action_text] > 2:
                    blocked_actions.add(action_text)
                if _stable_json(before_modified) == _stable_json(after_modified):
                    no_change_counts[action_text] += 1
                    if no_change_counts[action_text] >= 2:
                        blocked_actions.add(action_text)
                else:
                    no_change_counts[action_text] = 0
            except Exception as exc:  # noqa: BLE001
                reward = 0.0
                done = True
                info = {}
                error_value = str(exc)
                blocked_actions.add(action_text)
                memory.update(
                    state_key=state_key,
                    pattern_key=pattern_key,
                    action_text=action_text,
                    reward=reward,
                    progress_delta=0.0,
                    error=error_value,
                    done=done,
                    task_score=final_score,
                )

            rewards.append(reward)
            steps_taken = step_number
            last_error = error_value
            log_step(
                step=step_number,
                action=action_text,
                reward=reward,
                done=done,
                error=error_value,
            )

            history.append(
                f"step={step_number} source={action_source} action={action_text} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_value or 'null'}"
            )

            if done:
                success = bool(final_score > 0.0)
                break
    finally:
        memory.save()
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()
        log_end(success=success, steps=steps_taken, rewards=rewards, final_score=final_score)
    return final_score


def main() -> None:
    """Run one configured task or all tasks in deterministic order."""

    client = create_client()
    memory = PolicyMemory(POLICY_CACHE_PATH)
    task_name = str(TASK_NAME).strip().lower()
    if task_name in {"all", "*"}:
        for task_index, ordered_task in enumerate(TASK_ORDER):
            run_episode(client=client, memory=memory, task_name=ordered_task, seed=task_index)
        return
    run_episode(client=client, memory=memory, task_name=task_name, seed=0)


if __name__ == "__main__":
    main()
