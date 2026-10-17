from openpi.planning.qwenvl_event_schema import PlannerPatch
from openpi.tasks.shellgame import qwenvl_event_adapter as adapter


def _patch(operation, **delta):
    mapping = {
        "request_id": "r",
        "event": {
            "event_id": "e",
            "type": "event",
            "entities": [],
            "state_delta": {"operation": operation, **delta},
            "confidence": 0.9,
            "evidence": [],
        },
        "subgoal_updates": [],
        "next_subgoal": None,
        "decision": "propose_update",
        "request_reobservation": False,
    }
    return PlannerPatch.from_mapping(mapping)


def test_opponent_camera_coordinates_are_calibrated():
    reveal = _patch(
        "set_relation",
        subject="ball",
        predicate="contained_by",
        object="screen_right_cup",
    )
    swap = _patch(
        "exchange_entity_states",
        subjects=["screen_left_cup", "screen_middle_cup"],
    )
    assert adapter.initial_slot_from_patch(reveal) == "left"
    assert adapter.swap_pair_from_patch(swap) == ("middle", "right")


def test_world_labels_are_converted_to_canonical_screen_labels():
    assert adapter.screen_cup_from_world_slot("left") == "screen_right_cup"
    assert adapter.screen_cup_from_world_slot("middle") == "screen_middle_cup"
    assert adapter.screen_cup_from_world_slot("right") == "screen_left_cup"
    assert adapter.screen_pair_from_world_pair(("left", "middle")) == (
        "screen_middle_cup",
        "screen_right_cup",
    )
    assert adapter.screen_pair_from_world_pair(("right", "left")) == (
        "screen_left_cup",
        "screen_right_cup",
    )


def test_task_ledger_is_idempotent():
    patch = _patch("exchange_entity_states", subjects=["left_cup", "right_cup"])
    ledger = adapter.ShellGameTaskLedger("ep", target_slot="left")
    assert ledger.commit_swap(patch)
    assert ledger.target_slot == "right"
    assert not ledger.commit_swap(patch)
    assert ledger.target_slot == "right"
