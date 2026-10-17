from openpi.tasks.robomme.pickxtimes import semantic_memory_event
from openpi.training.mem import robomme_pickxtimes_dataset


def test_decoded_event_types_drive_semantic_state_targets():
    episode = {"required_count": 2}
    event_types = [
        semantic_memory_event.PICK_COMPLETE,
        semantic_memory_event.PLACE_COMPLETE,
        semantic_memory_event.PICK_COMPLETE,
        semantic_memory_event.PLACE_COMPLETE,
        semantic_memory_event.PRESS_COMPLETE,
    ]

    states = robomme_pickxtimes_dataset.PickXtimesWindowDataset.states_from_event_types(
        episode,
        event_types,
    )

    assert [state["completed_count"] for state in states] == [0, 1, 1, 2, 2]
    assert [state["holding"] for state in states] == [True, False, True, False, False]
    assert [state["remaining_count"] for state in states] == [2, 1, 1, 0, 0]
    assert [state["should_press"] for state in states] == [False, False, False, True, False]
    assert [state["done"] for state in states] == [False, False, False, False, True]
