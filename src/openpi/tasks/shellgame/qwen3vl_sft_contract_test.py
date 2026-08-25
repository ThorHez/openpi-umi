import pytest

from openpi.tasks.shellgame import qwen3vl_sft_contract as contract


def test_compact_targets_are_deterministic():
    assert contract.compact_response("reveal", "screen_left_cup") == ('{"screen_cup":"screen_left_cup"}')
    assert (
        contract.compact_response("swap", ("screen_left_cup", "screen_right_cup"))
        == '{"screen_pair":["screen_left_cup","screen_right_cup"]}'
    )
    assert contract.compact_response("no_event") == '{"event":"no_event"}'


def test_compact_targets_reject_noncanonical_pairs():
    with pytest.raises(ValueError, match="left-to-right"):
        contract.compact_response("swap", ("screen_right_cup", "screen_left_cup"))
