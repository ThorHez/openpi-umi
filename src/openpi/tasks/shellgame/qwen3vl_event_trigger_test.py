from openpi.tasks.shellgame import qwen3vl_event_trigger as trigger


def _positive(start, pair):
    return {"window_start": start, "prediction": {"screen_pair": list(pair)}}


def test_overlapping_windows_are_deduplicated_without_gt():
    ab = ("screen_left_cup", "screen_middle_cup")
    ac = ("screen_left_cup", "screen_right_cup")
    bc = ("screen_middle_cup", "screen_right_cup")
    windows = [
        _positive(18, ab),
        _positive(19, ab),
        _positive(21, ab),
        _positive(29, ac),
        _positive(30, ac),
        _positive(40, bc),
    ]
    events = trigger.cluster_event_windows(windows, max_positive_gap=3)
    assert [event.pair for event in events] == [ab, ac, bc]
    scores = trigger.score_event_triggers(
        events,
        gt_starts=[20, 30, 40],
        gt_pairs=[ab, ac, bc],
        tolerance=5,
    )
    assert scores["event_precision"] == 1.0
    assert scores["event_recall"] == 1.0
    assert scores["exact_relation_sequence"]


def test_separated_repeat_is_counted_as_duplicate_false_positive():
    pair = ("screen_left_cup", "screen_middle_cup")
    events = trigger.cluster_event_windows([_positive(20, pair), _positive(25, pair)], max_positive_gap=2)
    scores = trigger.score_event_triggers(events, gt_starts=[20], gt_pairs=[pair], tolerance=6)
    assert scores["num_triggers"] == 2
    assert scores["false_positive_or_duplicate_triggers"] == 1
    assert not scores["exact_relation_sequence"]


def test_no_positive_window_means_all_events_are_missed():
    events = trigger.cluster_event_windows([{"window_start": 20, "prediction": {"event": "incomplete_event"}}])
    scores = trigger.score_event_triggers(
        events,
        gt_starts=[20],
        gt_pairs=[("screen_left_cup", "screen_middle_cup")],
    )
    assert scores["missed_events"] == 1
    assert scores["event_recall"] == 0.0
