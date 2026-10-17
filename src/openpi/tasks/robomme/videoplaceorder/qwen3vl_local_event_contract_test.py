from openpi.tasks.robomme.videoplaceorder import qwen3vl_local_event_contract as contract


def test_place_and_swap_contracts():
    place = contract.compact_place_response("r2_c4")
    assert contract.validate_compact_response(place) == {
        "event": "place_complete",
        "target_cell": "r2_c4",
    }
    swap = contract.compact_swap_response("r2_c4", "r6_c1")
    assert contract.validate_compact_response(swap)["target_cells"] == ["r2_c4", "r6_c1"]
