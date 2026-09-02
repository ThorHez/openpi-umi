import pytest

from openpi.tasks.robomme import four_task_state


def test_unmask_swap_tracks_two_targets_and_pick_rank():
    state = four_task_state.TargetIdentityState.empty(("red", "blue"))
    state = state.observe_target("red", "r2_c1", covered=True)
    state = state.observe_target("blue", "r2_c5", covered=True)
    state = state.apply_swap("r2_c1", "r2_c3")
    state = state.apply_swap("r2_c5", "r2_c3")
    assert state.target_cells == ("r2_c5", "r2_c3")
    assert state.completed_swap_count == 2
    assert state.complete_pick().next_pick_rank == 1


def test_place_order_keeps_ordinal_identity_through_swap():
    state = four_task_state.OrderedTargetState("green", queried_ordinal=2)
    state = state.place_complete(1, "r3_c1")
    state = state.place_complete(2, "r3_c4")
    state = state.place_complete(3, "r5_c6")
    assert state.queried_cell == "r3_c4"
    state = state.swap_complete("r3_c4", "r6_c2")
    assert state.queried_cell == "r6_c2"


def test_pick_count_is_local_event_driven():
    state = four_task_state.PickCountState("red", required_count=2)
    state = state.apply("pick_complete").apply("place_complete")
    assert state.completed_count == 1
    assert not state.ready_to_press
    state = state.apply("incomplete_event")
    assert state.completed_count == 1
    state = state.apply("pick_complete").apply("place_complete")
    assert state.ready_to_press
    assert state.apply("press_complete").done


def test_pick_count_rejects_early_press_and_duplicate_place():
    state = four_task_state.PickCountState("blue", required_count=1)
    with pytest.raises(ValueError):
        state.apply("press_complete")
    with pytest.raises(ValueError):
        state.apply("place_complete")

