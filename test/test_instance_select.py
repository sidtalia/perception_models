"""Prompted-instance proposals from CLIPSeg maps."""

from __future__ import annotations

import numpy as np
import pytest

from perception_models.instance_select import instances_from_detections, propose_instances


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


# ----------------------------------------------------------- detector boxes
def test_detections_per_prompt_cap_and_order():
    boxes = np.array([[10, 10, 20, 20], [30, 30, 50, 50], [0, 0, 5, 5], [60, 10, 70, 30]], float)
    scores = np.array([0.3, 0.8, 0.5, 0.6])
    cls = np.array([0, 0, 0, 1])
    inst = instances_from_detections(boxes, scores, cls, (100, 100), max_instances_per_prompt=2, box_pad_frac=0.0)
    assert [(i.prompt_idx, i.score) for i in inst] == [(0, 0.8), (1, 0.6), (0, 0.5)]
    assert inst[0].box == (30.0, 30.0, 50.0, 50.0)
    assert inst[0].area_frac == pytest.approx(400 / 10000)


def test_detections_min_score_pad_and_clip():
    boxes = np.array([[0, 0, 10, 10], [90, 90, 99, 99]], float)
    inst = instances_from_detections(boxes, [0.05, 0.9], [0, 1], (100, 100), min_score=0.1, box_pad_frac=0.5)
    assert len(inst) == 1
    x0, y0, x1, y1 = inst[0].box
    assert (x0, y0) == (85.5, 85.5)
    assert (x1, y1) == (99.0, 99.0)  # clipped to the image


def test_detections_empty_and_degenerate():
    assert instances_from_detections(np.zeros((0, 4)), [], [], (10, 10)) == []
    assert instances_from_detections([[5, 5, 5, 9]], [0.9], [0], (10, 10), box_pad_frac=0.0) == []
    with pytest.raises(ValueError):
        instances_from_detections([[0, 0, 1, 1]], [0.9, 0.8], [0], (10, 10))
