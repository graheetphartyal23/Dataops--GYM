"""Task definitions for ``dataops-gym``.

This module defines the benchmark scenarios used by the OpenEnv environment.
Each public task family keeps the hackathon-facing `easy` / `medium` / `hard`
shape while internally supporting deterministic variants so the benchmark is
broader and less gameable.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


TableRow = Dict[str, Any]


class TaskDefinition(TypedDict):
    """Typed structure returned by task factory functions."""

    initial_table: List[TableRow]
    hidden_issues: List[Dict[str, Any]]
    constraints: List[str]
    max_steps: int
    goal: str
    difficulty: str
    required_columns: List[str]
    expected_outcome: Dict[str, Any]
    variant_id: str


class HiddenIssue(TypedDict, total=False):
    """Structured hidden issue description for a task table."""

    type: str
    rows: List[int]
    row: int
    column: str
    constraint: str
    description: str


def _pick_variant(variant: int | None, variants: List[TaskDefinition]) -> TaskDefinition:
    """Select a deterministic task variant with a stable default."""

    index = 0 if variant is None else max(0, min(len(variants) - 1, int(variant)))
    return variants[index]


def easy_cleaning_task(variant: int | None = None) -> TaskDefinition:
    """Create an easy multi-step cleaning task with duplicates and missing data."""

    variants: List[TaskDefinition] = [
        {
            "goal": "Clean the dataset by removing duplicates and filling missing values.",
            "difficulty": "easy",
            "variant_id": "easy_customer_master",
            "required_columns": ["name", "city", "email"],
            "expected_outcome": {
                "expected_row_count": 4,
                "required_non_null_columns": ["name", "city", "email"],
                "unique_by": ["customer_id"],
                "exactly_one_of_rows": [[2, 3]],
                "validation_rules": [
                    "Exactly one of rows 2 or 3 should remain after deduplication.",
                    "No remaining row should have null values in name, city, or email.",
                    "All remaining customer_id values should be unique.",
                ],
            },
            "initial_table": [
                {"row_id": 1, "customer_id": "C001", "name": "Alice Wong", "city": "Seattle", "email": "alice@example.com"},
                {"row_id": 2, "customer_id": "C002", "name": "Ben Ortiz", "city": None, "email": "ben@example.com"},
                {"row_id": 3, "customer_id": "C002", "name": "Ben Ortiz", "city": None, "email": "ben@example.com"},
                {"row_id": 4, "customer_id": "C003", "name": "Carla Singh", "city": "Austin", "email": None},
                {"row_id": 5, "customer_id": "C004", "name": "Drew Park", "city": "Boston", "email": "drew@example.com"},
            ],
            "hidden_issues": [
                {
                    "type": "duplicate",
                    "rows": [2, 3],
                    "description": "Rows 2 and 3 are duplicates and only one should remain.",
                },
                {
                    "type": "missing_value",
                    "row": 2,
                    "column": "city",
                    "description": "Row 2 is missing a required city value.",
                },
                {
                    "type": "missing_value",
                    "row": 4,
                    "column": "email",
                    "description": "Row 4 is missing a required email value.",
                },
            ],
            "constraints": [
                "Keep one representative row for each real customer.",
                "Do not delete rows solely because they contain missing values.",
                "Name, city, and email must be populated for every remaining row.",
            ],
            "max_steps": 7,
        },
        {
            "goal": "Clean the dataset by removing duplicates and filling missing values.",
            "difficulty": "easy",
            "variant_id": "easy_vendor_onboarding",
            "required_columns": ["name", "city", "email"],
            "expected_outcome": {
                "expected_row_count": 4,
                "required_non_null_columns": ["name", "city", "email"],
                "unique_by": ["vendor_id"],
                "exactly_one_of_rows": [[32, 33]],
                "validation_rules": [
                    "Exactly one of rows 32 or 33 should remain after deduplication.",
                    "No remaining row should have null values in name, city, or email.",
                    "All remaining vendor_id values should be unique.",
                ],
            },
            "initial_table": [
                {"row_id": 31, "vendor_id": "V001", "name": "Northwind Foods", "city": "Denver", "email": "ops@northwind.example.com"},
                {"row_id": 32, "vendor_id": "V002", "name": "Blue Harbor Ltd", "city": "Miami", "email": "contact@blueharbor.example.com"},
                {"row_id": 33, "vendor_id": "V002", "name": "Blue Harbor Ltd", "city": "Miami", "email": "contact@blueharbor.example.com"},
                {"row_id": 34, "vendor_id": "V003", "name": "Atlas Office Supply", "city": None, "email": "service@atlas.example.com"},
                {"row_id": 35, "vendor_id": "V004", "name": "Peak Systems", "city": "Portland", "email": None},
            ],
            "hidden_issues": [
                {
                    "type": "duplicate",
                    "rows": [32, 33],
                    "description": "Rows 32 and 33 are duplicates and only one should remain.",
                },
                {
                    "type": "missing_value",
                    "row": 34,
                    "column": "city",
                    "description": "Row 34 is missing a required city value.",
                },
                {
                    "type": "missing_value",
                    "row": 35,
                    "column": "email",
                    "description": "Row 35 is missing a required email value.",
                },
            ],
            "constraints": [
                "Keep one representative row for each real vendor.",
                "Do not delete rows solely because they contain missing values.",
                "Name, city, and email must be populated for every remaining row.",
            ],
            "max_steps": 7,
        },
    ]
    return _pick_variant(variant, variants)


def medium_normalization_task(variant: int | None = None) -> TaskDefinition:
    """Create a medium multi-step normalization task with several issue types."""

    variants: List[TaskDefinition] = [
        {
            "goal": "Normalize the dataset by fixing casing, removing duplicates, and correcting invalid email formats.",
            "difficulty": "medium",
            "variant_id": "medium_customer_normalization",
            "required_columns": ["name", "city", "email"],
            "expected_outcome": {
                "expected_row_count": 5,
                "required_non_null_columns": ["name", "city", "email"],
                "unique_by": ["customer_id"],
                "normalized_columns": {"name": "title_case", "city": "title_case"},
                "format_rules": {"email": "valid_email"},
                "exactly_one_of_rows": [[11, 13]],
                "validation_rules": [
                    "Exactly one of rows 11 or 13 should remain after deduplication.",
                    "All remaining emails should satisfy a valid email format.",
                    "Names and cities should follow a consistent human-readable casing convention.",
                    "All remaining customer_id values should be unique.",
                ],
            },
            "initial_table": [
                {"row_id": 10, "customer_id": "C100", "name": "jane miller", "city": "new york", "email": "jane.miller@example.com"},
                {"row_id": 11, "customer_id": "C101", "name": "OMAR HASSAN", "city": "CHICAGO", "email": "omar.hassan[at]example.com"},
                {"row_id": 12, "customer_id": "C102", "name": "Priya Nair", "city": "San Jose", "email": "priya.nair@example.com"},
                {"row_id": 13, "customer_id": "C101", "name": "OMAR HASSAN", "city": "CHICAGO", "email": "omar.hassan[at]example.com"},
                {"row_id": 14, "customer_id": "C103", "name": "li wei", "city": "seattle", "email": "li.wei.example.com"},
                {"row_id": 15, "customer_id": "C104", "name": "Maria Gomez", "city": "Austin", "email": "maria.gomez@example.com"},
            ],
            "hidden_issues": [
                {
                    "type": "duplicate",
                    "rows": [11, 13],
                    "description": "Rows 11 and 13 are duplicates and only one should remain.",
                },
                {
                    "type": "inconsistent_casing",
                    "rows": [10, 11, 14],
                    "column": "name",
                    "description": "Rows 10, 11, and 14 contain inconsistent casing in names.",
                },
                {
                    "type": "inconsistent_casing",
                    "rows": [10, 11, 14],
                    "column": "city",
                    "description": "Rows 10, 11, and 14 contain inconsistent casing in cities.",
                },
                {
                    "type": "invalid_format",
                    "row": 11,
                    "column": "email",
                    "description": "Row 11 contains an invalid email format.",
                },
                {
                    "type": "invalid_format",
                    "row": 14,
                    "column": "email",
                    "description": "Row 14 contains an invalid email format.",
                },
            ],
            "constraints": [
                "Preserve the original entity identity of each remaining row.",
                "Normalize names and cities to a consistent human-readable casing style.",
                "Only repair emails that are actually invalid.",
                "Do not introduce new duplicates while normalizing values.",
            ],
            "max_steps": 9,
        },
        {
            "goal": "Normalize the dataset by fixing casing, removing duplicates, and correcting invalid email formats.",
            "difficulty": "medium",
            "variant_id": "medium_partner_directory",
            "required_columns": ["name", "city", "email"],
            "expected_outcome": {
                "expected_row_count": 5,
                "required_non_null_columns": ["name", "city", "email"],
                "unique_by": ["partner_id"],
                "normalized_columns": {"name": "title_case", "city": "title_case"},
                "format_rules": {"email": "valid_email"},
                "exactly_one_of_rows": [[41, 43]],
                "validation_rules": [
                    "Exactly one of rows 41 or 43 should remain after deduplication.",
                    "All remaining emails should satisfy a valid email format.",
                    "Names and cities should use consistent title case.",
                    "All remaining partner_id values should be unique.",
                ],
            },
            "initial_table": [
                {"row_id": 40, "partner_id": "P100", "name": "delta analytics", "city": "san francisco", "email": "hello@delta.example.com"},
                {"row_id": 41, "partner_id": "P101", "name": "LUCIA ROMERO", "city": "MADRID", "email": "lucia.romero at example.com"},
                {"row_id": 42, "partner_id": "P102", "name": "Ken Ito", "city": "Tokyo", "email": "ken.ito@example.com"},
                {"row_id": 43, "partner_id": "P101", "name": "LUCIA ROMERO", "city": "MADRID", "email": "lucia.romero at example.com"},
                {"row_id": 44, "partner_id": "P103", "name": "amina ali", "city": "dubai", "email": "amina.ali.example.com"},
                {"row_id": 45, "partner_id": "P104", "name": "Sofia Hart", "city": "London", "email": "sofia.hart@example.com"},
            ],
            "hidden_issues": [
                {
                    "type": "duplicate",
                    "rows": [41, 43],
                    "description": "Rows 41 and 43 are duplicates and only one should remain.",
                },
                {
                    "type": "inconsistent_casing",
                    "rows": [40, 41, 44],
                    "column": "name",
                    "description": "Rows 40, 41, and 44 contain inconsistent casing in names.",
                },
                {
                    "type": "inconsistent_casing",
                    "rows": [40, 41, 44],
                    "column": "city",
                    "description": "Rows 40, 41, and 44 contain inconsistent casing in cities.",
                },
                {
                    "type": "invalid_format",
                    "row": 41,
                    "column": "email",
                    "description": "Row 41 contains an invalid email format.",
                },
                {
                    "type": "invalid_format",
                    "row": 44,
                    "column": "email",
                    "description": "Row 44 contains an invalid email format.",
                },
            ],
            "constraints": [
                "Preserve the original entity identity of each remaining row.",
                "Normalize names and cities to a consistent human-readable casing style.",
                "Only repair emails that are actually invalid.",
                "Do not introduce new duplicates while normalizing values.",
            ],
            "max_steps": 9,
        },
    ]
    return _pick_variant(variant, variants)


def hard_conflict_resolution_task(variant: int | None = None) -> TaskDefinition:
    """Create a hard multi-step conflict-resolution task with deceptive records."""

    variants: List[TaskDefinition] = [
        {
            "goal": "Resolve conflicting records, enforce unique email constraints, fix invalid formats, and preserve valid but unusual data.",
            "difficulty": "hard",
            "variant_id": "hard_customer_conflicts",
            "required_columns": ["name", "email", "phone", "status"],
            "expected_outcome": {
                "expected_row_count_range": {"min": 5, "max": 6},
                "unique_by": ["email"],
                "format_rules": {"email": "valid_email", "phone": "normalized_phone"},
                "exactly_one_of_rows": [[21, 22], [23, 24], [26, 27]],
                "must_preserve_valid_rows": [25, 28],
                "validation_rules": [
                    "Exactly one of rows 21 or 22 should remain after deduplication.",
                    "Exactly one of rows 23 or 24 should remain after conflict resolution.",
                    "Exactly one of rows 26 or 27 should remain after enforcing email uniqueness.",
                    "No two remaining rows should share the same email address.",
                    "All remaining emails should satisfy a valid email format.",
                    "All remaining phone values should be normalized to a consistent valid format.",
                    "Rows 25 and 28 should remain logically unchanged because they are valid trap rows.",
                ],
            },
            "initial_table": [
                {"row_id": 21, "customer_id": "C200", "name": "Nina Patel", "email": "nina.patel@example.com", "phone": "206-555-0101", "status": "active"},
                {"row_id": 22, "customer_id": "C200", "name": "Nina Patel", "email": "nina.patel@example.com", "phone": "206-555-0101", "status": "active"},
                {"row_id": 23, "customer_id": "C201", "name": "Evan Cole", "email": "evan.cole@example", "phone": "4155550102", "status": "active"},
                {"row_id": 24, "customer_id": "C201", "name": "Evan Cole", "email": "evan.cole@example.com", "phone": "(415) 555-0102", "status": "inactive"},
                {"row_id": 25, "customer_id": "C202", "name": "A. J. Brown", "email": "aj.brown@example.com", "phone": "+1-312-555-0103", "status": "active"},
                {"row_id": 26, "customer_id": "C203", "name": "Marta Silva", "email": "shared@example.com", "phone": "646-555-0104", "status": "active"},
                {"row_id": 27, "customer_id": "C204", "name": "Martin Silva", "email": "shared@example.com", "phone": "646-555-0105", "status": "active"},
                {"row_id": 28, "customer_id": "C205", "name": "Q Xu", "email": "q.xu+vip@example.com", "phone": "917-555-0106", "status": "active"},
            ],
            "hidden_issues": [
                {
                    "type": "duplicate",
                    "rows": [21, 22],
                    "description": "Rows 21 and 22 are exact duplicates and only one should remain.",
                },
                {
                    "type": "conflict",
                    "rows": [23, 24],
                    "description": "Rows 23 and 24 conflict for the same customer and must be reconciled into one trustworthy record.",
                },
                {
                    "type": "invalid_format",
                    "row": 23,
                    "column": "email",
                    "description": "Row 23 contains an invalid email format.",
                },
                {
                    "type": "invalid_format",
                    "row": 23,
                    "column": "phone",
                    "description": "Row 23 contains an invalid phone format.",
                },
                {
                    "type": "constraint_violation",
                    "constraint": "unique_email",
                    "rows": [26, 27],
                    "description": "Rows 26 and 27 violate the unique email constraint.",
                },
                {
                    "type": "valid_trap",
                    "row": 28,
                    "description": "Row 28 is valid even though the plus-address format may look suspicious.",
                },
                {
                    "type": "valid_trap",
                    "row": 25,
                    "description": "Row 25 is valid even though the name abbreviation may look inconsistent.",
                },
            ],
            "constraints": [
                "Email values must be unique across the final table.",
                "Every remaining row must represent a single coherent customer record.",
                "Do not modify valid rows just because they look unusual.",
                "Prefer correction and conflict resolution over unnecessary deletion.",
            ],
            "max_steps": 14,
        },
        {
            "goal": "Resolve conflicting records, enforce unique email constraints, fix invalid formats, and preserve valid but unusual data.",
            "difficulty": "hard",
            "variant_id": "hard_account_merges",
            "required_columns": ["name", "email", "phone", "status"],
            "expected_outcome": {
                "expected_row_count_range": {"min": 5, "max": 6},
                "unique_by": ["email"],
                "format_rules": {"email": "valid_email", "phone": "normalized_phone"},
                "exactly_one_of_rows": [[51, 52], [53, 54], [56, 57]],
                "must_preserve_valid_rows": [55, 58],
                "validation_rules": [
                    "Exactly one of rows 51 or 52 should remain after deduplication.",
                    "Exactly one of rows 53 or 54 should remain after conflict resolution.",
                    "Exactly one of rows 56 or 57 should remain after enforcing email uniqueness.",
                    "No two remaining rows should share the same email address.",
                    "All remaining emails should satisfy a valid email format.",
                    "All remaining phone values should be normalized to a consistent valid format.",
                    "Rows 55 and 58 should remain logically unchanged because they are valid trap rows.",
                ],
            },
            "initial_table": [
                {"row_id": 51, "customer_id": "A900", "name": "Lena Brooks", "email": "lena.brooks@example.com", "phone": "212-555-0111", "status": "active"},
                {"row_id": 52, "customer_id": "A900", "name": "Lena Brooks", "email": "lena.brooks@example.com", "phone": "212-555-0111", "status": "active"},
                {"row_id": 53, "customer_id": "A901", "name": "Ravi Shah", "email": "ravi.shah example.com", "phone": "6465550112", "status": "active"},
                {"row_id": 54, "customer_id": "A901", "name": "Ravi Shah", "email": "ravi.shah@example.com", "phone": "646-555-0112", "status": "inactive"},
                {"row_id": 55, "customer_id": "A902", "name": "M. E. Klein", "email": "mek@example.com", "phone": "+1-303-555-0113", "status": "active"},
                {"row_id": 56, "customer_id": "A903", "name": "Sana Noor", "email": "ops@example.com", "phone": "718-555-0114", "status": "active"},
                {"row_id": 57, "customer_id": "A904", "name": "Sana N.", "email": "ops@example.com", "phone": "718-555-0115", "status": "active"},
                {"row_id": 58, "customer_id": "A905", "name": "Bo Li", "email": "bo.li+archive@example.com", "phone": "415-555-0116", "status": "active"},
            ],
            "hidden_issues": [
                {
                    "type": "duplicate",
                    "rows": [51, 52],
                    "description": "Rows 51 and 52 are exact duplicates and only one should remain.",
                },
                {
                    "type": "conflict",
                    "rows": [53, 54],
                    "description": "Rows 53 and 54 conflict for the same customer and must be reconciled into one trustworthy record.",
                },
                {
                    "type": "invalid_format",
                    "row": 53,
                    "column": "email",
                    "description": "Row 53 contains an invalid email format.",
                },
                {
                    "type": "invalid_format",
                    "row": 53,
                    "column": "phone",
                    "description": "Row 53 contains an invalid phone format.",
                },
                {
                    "type": "constraint_violation",
                    "constraint": "unique_email",
                    "rows": [56, 57],
                    "description": "Rows 56 and 57 violate the unique email constraint.",
                },
                {
                    "type": "valid_trap",
                    "row": 55,
                    "description": "Row 55 is valid even though the abbreviated name may look unusual.",
                },
                {
                    "type": "valid_trap",
                    "row": 58,
                    "description": "Row 58 is valid even though the plus-address format may look suspicious.",
                },
            ],
            "constraints": [
                "Email values must be unique across the final table.",
                "Every remaining row must represent a single coherent customer record.",
                "Do not modify valid rows just because they look unusual.",
                "Prefer correction and conflict resolution over unnecessary deletion.",
            ],
            "max_steps": 14,
        },
    ]
    return _pick_variant(variant, variants)


easy_cleaning_task.variant_count = 2
medium_normalization_task.variant_count = 2
hard_conflict_resolution_task.variant_count = 2
