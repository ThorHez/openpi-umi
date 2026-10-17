import pytest

from openpi.planning import qwenvl_event_schema as schema


def _swap_text():
    return """```json
    {"request_id":"r1","event":{"event_id":"e1","type":"container_exchange",
    "entities":["cup_a","cup_b"],"state_delta":{"operation":"exchange_entity_states",
    "subjects":["cup_a","cup_b"]},"confidence":0.8,"evidence":[]},
    "subgoal_updates":[],"next_subgoal":null,"decision":"propose_update",
    "request_reobservation":false}
    ```"""


def test_fenced_patch_round_trip():
    patch = schema.PlannerPatch.from_text(_swap_text())
    assert patch.event.state_delta.subjects == ("cup_a", "cup_b")
    assert schema.PlannerPatch.from_mapping(patch.to_dict()) == patch


def test_operation_without_subjects_is_valid():
    mapping = schema.extract_json_object(_swap_text())
    mapping["event"]["state_delta"] = {
        "operation": "set_relation",
        "subject": "ball",
        "predicate": "contained_by",
        "object": "cup_a",
    }
    assert schema.PlannerPatch.from_mapping(mapping).event.state_delta.subjects == ()


def test_duplicate_exchange_subject_is_rejected():
    mapping = schema.extract_json_object(_swap_text())
    mapping["event"]["state_delta"]["subjects"] = ["cup_a", "cup_a"]
    with pytest.raises(schema.SchemaValidationError):
        schema.PlannerPatch.from_mapping(mapping)
