from openpi.tasks.robomme.pickxtimes import qwen3vl_local_event_contract as contract


def test_local_contract_has_no_cumulative_count():
    text = contract.compact_response("place_complete", target_color="green")
    value = contract.validate_compact_response(text)
    assert value == {"event": "place_complete", "target_color": "green"}
    assert "count" not in text


def test_negative_contract():
    assert contract.validate_compact_response(contract.compact_response("incomplete_event")) == {
        "event": "incomplete_event"
    }

