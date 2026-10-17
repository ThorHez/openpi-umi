import pytest

from openpi.tasks.robomme import four_task_temporal_contract as contract


def test_teacher_and_student_both_use_twelve_frames():
    assert contract.TEACHER_FRAME_COUNT == 12
    assert contract.STUDENT_CLIP_FRAME_COUNT == 12


def test_student_clips_overlap_and_keep_masked_tail():
    clips = contract.causal_student_clips(25)
    assert [clip.start for clip in clips] == [0, 6, 12, 18, 24]
    assert clips[0].frame_indices == tuple(range(12))
    assert clips[1].frame_indices == tuple(range(6, 18))
    assert clips[-1].frame_indices == (24,)
    assert sum(clips[-1].frame_mask) == 1
    assert len(clips[-1].frame_mask) == 12


def test_empty_episode_has_no_student_clips():
    assert contract.causal_student_clips(0) == ()


def test_teacher_validation_rejects_wrong_count_or_order():
    contract.validate_teacher_frame_indices(list(range(12)))
    with pytest.raises(ValueError, match="require 12"):
        contract.validate_teacher_frame_indices(list(range(6)))
    with pytest.raises(ValueError, match="chronological"):
        contract.validate_teacher_frame_indices([0, 1, 3, 2, *range(4, 12)])

