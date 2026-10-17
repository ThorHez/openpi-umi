import pytest

from openpi.tasks.robomme import qwen3vl_unified_event_contract as contract


@pytest.mark.parametrize(
    "text",
    (
        contract.compact_response("target_covered", entity="red_cube", region_a="region_2"),
        contract.compact_response("swap_complete", region_a="region_0", region_b="region_3"),
        contract.compact_response("place_complete", region_a="region_1"),
        contract.compact_response("pick_complete", entity="blue_cube"),
        contract.compact_response("press_complete"),
        contract.compact_response("incomplete_event"),
    ),
)
def test_round_trip(text):
    assert contract.validate_compact_response(text)["event"]


def test_rejects_missing_fixed_fields():
    with pytest.raises(ValueError):
        contract.validate_compact_response('{"event":"no_completed_event"}')


def test_prompt_has_no_task_identifier():
    prompt = contract.prompt_for_goal("pick the red cube twice", focus_entity="red_cube")
    assert "Task:" not in prompt
    assert "VideoUnmask" not in prompt
