"""Strict, dependency-free schema for offline Qwen-VL event proposals.

This module intentionally depends only on the Python standard library so the
same JSON contract can be imported from the isolated Qwen environment and the
OpenPI/JAX environment.  Qwen is allowed to *propose* a patch; task adapters
and deterministic task managers remain responsible for accepting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from typing import Any

DECISIONS = frozenset(
    {
        "propose_update",
        "keep_state",
        "request_reobservation",
        "report_failure",
        "finish_task",
    }
)
OPERATIONS = frozenset(
    {
        "set_relation",
        "remove_relation",
        "exchange_entity_states",
        "set_attribute",
        "phase_transition",
        "no_state_change",
    }
)


class SchemaValidationError(ValueError):
    """Raised when a generated planner patch violates the public contract."""


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{key!r} must be a non-empty string")
    return value.strip()


def _optional_string(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{key!r} must be null or a non-empty string")
    return value.strip()


def _confidence(value: Any, *, key: str = "confidence") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaValidationError(f"{key!r} must be a number in [0, 1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise SchemaValidationError(f"{key!r} must be in [0, 1], got {result}")
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from a model response.

    This accepts fenced JSON and short explanatory prefixes while still
    rejecting trailing or leading fragments that do not contain an object.
    Braces inside quoted JSON strings are handled correctly.
    """
    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError("Qwen response is empty")
    start = text.find("{")
    if start < 0:
        raise SchemaValidationError("Qwen response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise SchemaValidationError("Qwen response contains an unbalanced JSON object")
    try:
        value = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError("Top-level Qwen output must be a JSON object")
    return value


@dataclass(frozen=True)
class StateDelta:
    """A constrained, task-agnostic state mutation proposal."""

    operation: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    subjects: tuple[str, ...] = ()
    attribute: str | None = None
    value: Any = None

    @classmethod
    def from_mapping(cls, mapping: Any) -> StateDelta:
        if not isinstance(mapping, dict):
            raise SchemaValidationError("event.state_delta must be a JSON object")
        operation = _required_string(mapping, "operation")
        if operation not in OPERATIONS:
            raise SchemaValidationError(f"Unsupported state-delta operation: {operation!r}")
        raw_subjects = mapping.get("subjects", [])
        if raw_subjects is None:
            raw_subjects = []
        if not isinstance(raw_subjects, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw_subjects
        ):
            raise SchemaValidationError("state_delta.subjects must be a list of entity strings")
        result = cls(
            operation=operation,
            subject=_optional_string(mapping, "subject"),
            predicate=_optional_string(mapping, "predicate"),
            object=_optional_string(mapping, "object"),
            subjects=tuple(item.strip() for item in raw_subjects),
            attribute=_optional_string(mapping, "attribute"),
            value=mapping.get("value"),
        )
        if operation in {"set_relation", "remove_relation"} and not all(
            (result.subject, result.predicate, result.object)
        ):
            raise SchemaValidationError(f"{operation} requires subject, predicate, and object")
        if operation == "exchange_entity_states" and (
            len(result.subjects) != 2 or result.subjects[0] == result.subjects[1]
        ):
            raise SchemaValidationError("exchange_entity_states requires two distinct subjects")
        if operation == "set_attribute" and not all((result.subject, result.attribute)):
            raise SchemaValidationError("set_attribute requires subject and attribute")
        return result

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"operation": self.operation}
        for key in ("subject", "predicate", "object", "attribute"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.subjects:
            result["subjects"] = list(self.subjects)
        if self.value is not None:
            result["value"] = self.value
        return result


@dataclass(frozen=True)
class EventProposal:
    event_id: str
    type: str
    entities: tuple[str, ...]
    state_delta: StateDelta
    confidence: float
    evidence: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Any) -> EventProposal:
        if not isinstance(mapping, dict):
            raise SchemaValidationError("event must be a JSON object")
        raw_entities = mapping.get("entities", ())
        raw_evidence = mapping.get("evidence", ())
        for key, values in (("entities", raw_entities), ("evidence", raw_evidence)):
            if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise SchemaValidationError(f"event.{key} must be a list of strings")
        return cls(
            event_id=_required_string(mapping, "event_id"),
            type=_required_string(mapping, "type"),
            entities=tuple(item.strip() for item in raw_entities),
            state_delta=StateDelta.from_mapping(mapping.get("state_delta")),
            confidence=_confidence(mapping.get("confidence")),
            evidence=tuple(item.strip() for item in raw_evidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "entities": list(self.entities),
            "state_delta": self.state_delta.to_dict(),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class SubgoalUpdate:
    subgoal_id: str
    old_status: str
    proposed_status: str
    confidence: float
    required_evidence: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Any) -> SubgoalUpdate:
        if not isinstance(mapping, dict):
            raise SchemaValidationError("Each subgoal update must be an object")
        return cls(
            subgoal_id=_required_string(mapping, "subgoal_id"),
            old_status=_required_string(mapping, "old_status"),
            proposed_status=_required_string(mapping, "proposed_status"),
            confidence=_confidence(mapping.get("confidence")),
            required_evidence=_optional_string(mapping, "required_evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "subgoal_id": self.subgoal_id,
            "old_status": self.old_status,
            "proposed_status": self.proposed_status,
            "confidence": self.confidence,
        }
        if self.required_evidence is not None:
            result["required_evidence"] = self.required_evidence
        return result


@dataclass(frozen=True)
class NextSubgoal:
    id: str
    instruction: str
    success_condition: str
    failure_condition: str
    focus_entities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Any) -> NextSubgoal:
        if not isinstance(mapping, dict):
            raise SchemaValidationError("next_subgoal must be null or an object")
        raw_entities = mapping.get("focus_entities", ())
        if not isinstance(raw_entities, list) or any(not isinstance(item, str) for item in raw_entities):
            raise SchemaValidationError("next_subgoal.focus_entities must be a list of strings")
        return cls(
            id=_required_string(mapping, "id"),
            instruction=_required_string(mapping, "instruction"),
            success_condition=_required_string(mapping, "success_condition"),
            failure_condition=_required_string(mapping, "failure_condition"),
            focus_entities=tuple(item.strip() for item in raw_entities),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "success_condition": self.success_condition,
            "failure_condition": self.failure_condition,
            "focus_entities": list(self.focus_entities),
        }


@dataclass(frozen=True)
class PlannerPatch:
    """Validated result of one low-frequency Qwen-VL call."""

    request_id: str
    event: EventProposal
    subgoal_updates: tuple[SubgoalUpdate, ...] = ()
    next_subgoal: NextSubgoal | None = None
    decision: str = "propose_update"
    request_reobservation: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> PlannerPatch:
        decision = _required_string(mapping, "decision")
        if decision not in DECISIONS:
            raise SchemaValidationError(f"Unsupported decision: {decision!r}")
        raw_updates = mapping.get("subgoal_updates", ())
        if not isinstance(raw_updates, list):
            raise SchemaValidationError("subgoal_updates must be a list")
        raw_reobserve = mapping.get("request_reobservation", False)
        if not isinstance(raw_reobserve, bool):
            raise SchemaValidationError("request_reobservation must be boolean")
        known = {
            "request_id",
            "event",
            "subgoal_updates",
            "next_subgoal",
            "decision",
            "request_reobservation",
        }
        return cls(
            request_id=_required_string(mapping, "request_id"),
            event=EventProposal.from_mapping(mapping.get("event")),
            subgoal_updates=tuple(SubgoalUpdate.from_mapping(item) for item in raw_updates),
            next_subgoal=(
                None if mapping.get("next_subgoal") is None else NextSubgoal.from_mapping(mapping["next_subgoal"])
            ),
            decision=decision,
            request_reobservation=raw_reobserve,
            extra={key: value for key, value in mapping.items() if key not in known},
        )

    @classmethod
    def from_text(cls, text: str) -> PlannerPatch:
        return cls.from_mapping(extract_json_object(text))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "event": self.event.to_dict(),
            "subgoal_updates": [update.to_dict() for update in self.subgoal_updates],
            "next_subgoal": None if self.next_subgoal is None else self.next_subgoal.to_dict(),
            "decision": self.decision,
            "request_reobservation": self.request_reobservation,
        }
        result.update(self.extra)
        return result
