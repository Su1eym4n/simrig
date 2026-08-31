"""Stable machine-readable outcome and failure taxonomy."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class FailureCategory(str, Enum):
    """Backend-neutral terminal categories used across evaluation reports."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    FORBIDDEN_CONTACT = "forbidden_contact"
    SAFETY_VIOLATION = "safety_violation"
    INVALID_STATE = "invalid_state"
    TASK_FAILURE = "task_failure"
    INCOMPLETE = "incomplete"
    EVALUATOR_ERROR = "evaluator_error"
    UNKNOWN = "unknown"


FAILURE_CATEGORIES = frozenset(item.value for item in FailureCategory)
SAFETY_FAILURE_CATEGORIES = frozenset(
    {
        FailureCategory.FORBIDDEN_CONTACT.value,
        FailureCategory.SAFETY_VIOLATION.value,
        FailureCategory.INVALID_STATE.value,
        FailureCategory.EVALUATOR_ERROR.value,
    }
)
FAILURE_PRIORITY = {
    FailureCategory.EVALUATOR_ERROR.value: 0,
    FailureCategory.INVALID_STATE.value: 1,
    FailureCategory.FORBIDDEN_CONTACT.value: 2,
    FailureCategory.SAFETY_VIOLATION.value: 3,
    FailureCategory.TIMEOUT.value: 4,
    FailureCategory.TASK_FAILURE.value: 5,
    FailureCategory.INCOMPLETE.value: 6,
    FailureCategory.UNKNOWN.value: 7,
    FailureCategory.SUCCESS.value: 8,
}


def terminal_reason(
    category: FailureCategory | str,
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized terminal reason with a stable category and code."""
    category_value = str(
        category.value if isinstance(category, FailureCategory) else category
    )
    if category_value not in FAILURE_CATEGORIES:
        raise ValueError(
            f"Unknown failure category {category_value!r}; expected one of "
            f"{sorted(FAILURE_CATEGORIES)}"
        )
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Terminal reason code must be non-empty text.")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Terminal reason message must be non-empty text.")
    return {
        "category": category_value,
        "code": code,
        "message": message,
        "details": dict(details or {}),
    }


def normalize_terminal_reason(
    value: Any,
    *,
    task_success: bool | None,
) -> dict[str, Any]:
    """Normalize plugin output and never infer success from reward."""
    if isinstance(value, Mapping):
        category = str(value.get("category") or "")
        code = str(value.get("code") or "")
        message = str(value.get("message") or code or category)
        details = value.get("details")
        if category in FAILURE_CATEGORIES and code:
            return terminal_reason(
                category,
                code,
                message,
                details=details if isinstance(details, Mapping) else None,
            )
    if task_success is True:
        return terminal_reason(
            FailureCategory.SUCCESS,
            "success",
            "Independent success predicates passed.",
        )
    if task_success is False:
        return terminal_reason(
            FailureCategory.TASK_FAILURE,
            "independent_success_failed",
            "Independent success predicates did not pass.",
        )
    return terminal_reason(
        FailureCategory.UNKNOWN,
        "terminal_reason_unknown",
        "The evaluator did not provide a valid terminal reason.",
    )
