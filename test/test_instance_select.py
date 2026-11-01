"""Prompted-instance proposals from CLIPSeg maps."""

from __future__ import annotations

import numpy as np
import pytest

from perception_models.instance_select import propose_instances


def test_two_blobs_ranked_by_score_and_scaled():
    probs = np.zeros((1, 20, 20), np.float32)
    probs[0, 2:6, 2:6] = 0.6
    probs[0, 10:16, 12:18] = 0.9
    inst = propose_instances(probs, (40, 80), threshold=0.5, min_area_frac=0.0, box_pad_frac=0.0)
    assert len(inst) == 2
    best = inst[0]
    assert best.score == pytest.approx(0.9)
    x0, y0, x1, y1 = best.box
    assert (x0, y0) == (12 * 4, 10 * 2)  # sx = 80/20, sy = 40/20
    assert (x1, y1) == (18 * 4, 16 * 2)


def test_prompts_thresholded_independently_and_capped():
    probs = np.zeros((2, 10, 10), np.float32)
    probs[0, 1:4, 1:4] = 0.8
    probs[0, 6:9, 6:9] = 0.7
    probs[1, 1:4, 1:4] = 0.95  # overlaps prompt 0: still reported
    inst = propose_instances(probs, (10, 10), threshold=0.5, max_instances_per_prompt=1, min_area_frac=0.0)
    assert [i.prompt_idx for i in inst] == [1, 0]


def test_min_area_and_empty():
    probs = np.zeros((1, 10, 10), np.float32)
    probs[0, 0, 0] = 1.0
    assert propose_instances(probs, (10, 10), min_area_frac=0.05) == []
    assert propose_instances(np.zeros((3, 5, 5), np.float32), (5, 5)) == []


def test_box_padding_clipped_to_image():
    probs = np.zeros((1, 10, 10), np.float32)
    probs[0, 0:5, 0:10] = 0.9
    (inst,) = propose_instances(probs, (10, 10), box_pad_frac=0.5, min_area_frac=0.0)
    x0, y0, x1, y1 = inst.box
    assert x0 == 0.0 and y0 == 0.0 and x1 == 9.0 and y1 <= 9.0


def test_bad_shape():
    with pytest.raises(ValueError):
        propose_instances(np.zeros((5, 5)), (5, 5))
