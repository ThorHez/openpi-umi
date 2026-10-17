from openpi.tasks.robomme.videounmaskswap import qwen3vl_local_event_contract as contract


def test_target_and_swap_contracts():
    target = contract.compact_target_response("target_covered", "blue", "slot_3")
    assert contract.validate_compact_response(target)["target_color"] == "blue"
    swap = contract.compact_swap_response("slot_3", "slot_1")
    assert contract.validate_compact_response(swap)["event"] == "swap_complete"
