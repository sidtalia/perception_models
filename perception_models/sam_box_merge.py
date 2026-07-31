"""NanoSAM box prompts from CLIPSeg (nav) and YOLO-World (manip).

Nav and manip use **separate** decode budgets so people boxes never starve
traversability.
"""

from __future__ import annotations

import json
from typing import Any, Sequence


def parse_yolo_detections_json(data: str | bytes | dict) -> list[dict[str, Any]]:
    """Parse ``/yolo_world/detections`` String payload → list of box dicts."""
    if isinstance(data, dict):
        payload = data
    else:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
    return list(payload.get("detections") or [])


def clipseg_nav_boxes(
    clipseg_cands: Sequence[tuple[int, float, tuple[int, int, int, int]]],
    *,
    people_group_idx: int,
    drop_people: bool,
    max_boxes: int = 16,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """CLIPSeg CC boxes for nav SAM decode (trav/nontrav; optional people).

    When ``drop_people`` (YOLO owns people), exclude people-group CCs.
    Candidates are already area-sorted; keep order and cap at ``max_boxes``.
    """
    classes: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    for cls, _area, box in clipseg_cands:
        if drop_people and int(cls) == int(people_group_idx):
            continue
        classes.append(int(cls))
        boxes.append(tuple(int(v) for v in box))  # type: ignore[arg-type]
        if len(boxes) >= max_boxes:
            break
    return classes, boxes


def yolo_manip_boxes(
    yolo_dets: Sequence[dict[str, Any]],
    *,
    people_group_idx: int,
    object_group_idx: int | None = None,
    person_names: Sequence[str] = ("person",),
    max_boxes: int = 16,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """YOLO instance boxes for manip SAM decode (people + optional objects)."""
    person_set = {str(n).lower() for n in person_names}
    classes: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    for d in yolo_dets:
        if len(boxes) >= max_boxes:
            break
        name = str(d.get("name", "")).lower()
        xyxy = d.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        if name in person_set:
            classes.append(int(people_group_idx))
            boxes.append(box)
        elif object_group_idx is not None:
            classes.append(int(object_group_idx))
            boxes.append(box)
    return classes, boxes


def merge_sam_box_prompts(
    clipseg_cands: Sequence[tuple[int, float, tuple[int, int, int, int]]],
    yolo_dets: Sequence[dict[str, Any]],
    *,
    people_group_idx: int,
    skip_clipseg_people: bool,
    object_group_idx: int | None = None,
    person_names: Sequence[str] = ("person",),
    max_boxes: int = 16,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Legacy single-list merge (prefer separate nav/manip helpers + dual decode).

    Kept for tests/callers; does **not** prefer people when truncating — nav
    boxes first, then YOLO, so trav is not starved if this path is still used.
    """
    nav_c, nav_b = clipseg_nav_boxes(
        clipseg_cands,
        people_group_idx=people_group_idx,
        drop_people=skip_clipseg_people,
        max_boxes=max_boxes,
    )
    remain = max(0, max_boxes - len(nav_b))
    man_c, man_b = yolo_manip_boxes(
        yolo_dets,
        people_group_idx=people_group_idx,
        object_group_idx=object_group_idx,
        person_names=person_names,
        max_boxes=remain,
    )
    return nav_c + man_c, nav_b + man_b
