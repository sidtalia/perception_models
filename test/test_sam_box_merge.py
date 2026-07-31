"""Unit tests for NanoSAM nav / manip box helpers."""

from __future__ import annotations

from perception_models.sam_box_merge import (
    clipseg_nav_boxes,
    merge_sam_box_prompts,
    parse_yolo_detections_json,
    yolo_manip_boxes,
)


def test_parse_yolo_detections_json():
    raw = '{"detections":[{"xyxy":[1,2,3,4],"name":"person","conf":0.9}]}'
    dets = parse_yolo_detections_json(raw)
    assert len(dets) == 1
    assert dets[0]["name"] == "person"


def test_clipseg_nav_drops_people():
    cands = [
        (0, 0.3, (10, 10, 50, 50)),
        (1, 0.2, (60, 60, 90, 90)),
        (2, 0.4, (20, 20, 40, 80)),
    ]
    classes, boxes = clipseg_nav_boxes(
        cands, people_group_idx=2, drop_people=True, max_boxes=8
    )
    assert classes == [0, 1]
    assert len(boxes) == 2


def test_clipseg_nav_keeps_people_when_not_dropped():
    cands = [
        (0, 0.3, (10, 10, 50, 50)),
        (2, 0.4, (20, 20, 40, 80)),
    ]
    classes, _boxes = clipseg_nav_boxes(
        cands, people_group_idx=2, drop_people=False, max_boxes=8
    )
    assert classes == [0, 2]


def test_clipseg_nav_respects_budget():
    cands = [(0, float(10 - i), (i, i, i + 5, i + 5)) for i in range(10)]
    classes, boxes = clipseg_nav_boxes(
        cands, people_group_idx=2, drop_people=True, max_boxes=3
    )
    assert len(classes) == 3
    assert len(boxes) == 3


def test_yolo_manip_people_and_objects():
    yolo = [
        {"xyxy": [1, 1, 10, 10], "name": "person", "conf": 0.9},
        {"xyxy": [20, 20, 30, 30], "name": "chair", "conf": 0.8},
        {"xyxy": [40, 40, 50, 50], "name": "door", "conf": 0.7},
    ]
    classes, boxes = yolo_manip_boxes(
        yolo,
        people_group_idx=2,
        object_group_idx=3,
        max_boxes=8,
    )
    assert classes == [2, 3, 3]
    assert len(boxes) == 3


def test_yolo_manip_skips_objects_without_group():
    yolo = [
        {"xyxy": [1, 1, 10, 10], "name": "person", "conf": 0.9},
        {"xyxy": [20, 20, 30, 30], "name": "chair", "conf": 0.8},
    ]
    classes, boxes = yolo_manip_boxes(
        yolo, people_group_idx=2, object_group_idx=None, max_boxes=8
    )
    assert classes == [2]
    assert len(boxes) == 1


def test_yolo_manip_budget_independent_of_nav():
    """Many people still fit manip budget; does not touch nav list."""
    yolo = [
        {"xyxy": [i, i, i + 5, i + 5], "name": "person", "conf": 0.5}
        for i in range(0, 40, 5)
    ]
    classes, boxes = yolo_manip_boxes(
        yolo, people_group_idx=2, max_boxes=4
    )
    assert len(boxes) == 4
    assert all(c == 2 for c in classes)


def test_legacy_merge_nav_first_not_people_first():
    """Legacy merge keeps trav when truncating (nav first, then YOLO)."""
    cands = [(0, 0.5, (0, 0, 10, 10))]
    yolo = [
        {"xyxy": [i, i, i + 5, i + 5], "name": "person", "conf": 0.5}
        for i in range(0, 40, 5)
    ]
    classes, boxes = merge_sam_box_prompts(
        cands,
        yolo,
        people_group_idx=2,
        skip_clipseg_people=True,
        max_boxes=4,
    )
    assert len(boxes) == 4
    assert classes[0] == 0
    assert classes.count(2) == 3
