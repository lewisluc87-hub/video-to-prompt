from video2prompt.cv_analysis.composition import (
    estimate_composition,
    estimate_shot_size,
)
from video2prompt.schema import Subject


def test_shot_size_close_up_for_large_bbox():
    subj = Subject(label="person", confidence=0.9, bbox_norm=(0.1, 0.1, 0.9, 0.9))
    assert estimate_shot_size([subj]) == "close-up"


def test_shot_size_wide_for_small_bbox():
    subj = Subject(label="person", confidence=0.9, bbox_norm=(0.45, 0.45, 0.55, 0.55))
    assert estimate_shot_size([subj]) == "wide shot"


def test_shot_size_fallback_no_subjects():
    assert "wide" in estimate_shot_size([])


def test_composition_subject_left():
    subj = Subject(label="person", confidence=0.9, bbox_norm=(0.05, 0.2, 0.3, 0.8))
    out = estimate_composition([subj])
    assert "subject left" in out


def test_composition_subject_centered():
    subj = Subject(label="person", confidence=0.9, bbox_norm=(0.4, 0.2, 0.6, 0.8))
    out = estimate_composition([subj])
    assert "subject centered" in out
