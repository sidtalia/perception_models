"""Wire format, prompt parsing and camera sync (no GPU, no dora daemon)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from perception_models.dora_io import (
    LatestFrameSync,
    WireError,
    decode,
    decode_camera_frame,
    encode,
    encode_detections,
    encode_seg_masks,
    is_camera_input,
    join_stamp,
    parse_prompts,
    split_stamp,
)


def _frame(name="scene", stamp=12.5, h=4, w=6, depth=True):
    body = {"name": name, "rgb": np.full((h, w, 3), 7, np.uint8), "stamp": stamp}
    if depth:
        body["depth"] = np.ones((h, w), np.float32)
        body["intrinsics"] = np.eye(3)
    return encode("camera_frame", body)


def test_roundtrip_nested_arrays_and_lists():
    body = {"a": np.arange(6, dtype=np.int16).reshape(2, 3), "l": [1, "x", np.zeros(2)], "e": {}}
    t, out = decode(encode("thing", body))
    assert t == "thing"
    np.testing.assert_array_equal(out["a"], body["a"])
    assert out["a"].dtype == np.int16
    assert out["l"][0] == 1 and out["l"][1] == "x"
    assert out["e"] == {}


def test_matches_praxis_codec_if_available():
    praxis = pytest.importorskip("praxis.core.messages")
    msg = _frame()
    frame = praxis.camera_frame_from_message(msg)
    assert frame.name == "scene" and frame.rgb.shape == (4, 6, 3)
    t, body = decode(praxis.encode("camera_frame", {"name": "x", "rgb": np.zeros((2, 2, 3), np.uint8), "stamp": 1.0}))
    assert t == "camera_frame" and body["rgb"].shape == (2, 2, 3)


def test_decode_camera_frame_validates():
    f = decode_camera_frame(_frame())
    assert f["rgb"].dtype == np.uint8 and f["stamp"] == 12.5
    with pytest.raises(WireError):
        decode_camera_frame(encode("event", {"kind": "x"}))
    with pytest.raises(WireError):
        decode_camera_frame(encode("camera_frame", {"name": "d", "rgb": None, "stamp": 0.0}))
    with pytest.raises(WireError):
        decode(pa.array([1, 2]))


@pytest.mark.parametrize("stamp", [0.0, 1.5, 12345.678901234, 99.999999999])
def test_stamp_split_join(stamp):
    s, ns = split_stamp(stamp)
    assert 0 <= ns < 1_000_000_000
    assert abs(join_stamp(s, ns) - stamp) < 2e-9


def test_parse_prompts_variants():
    assert parse_prompts(pa.array(["red mug", "", " banana "])) == ["red mug", "banana"]
    assert parse_prompts(encode("prompt", {"prompts": ["a", "b"]})) == ["a", "b"]
    assert parse_prompts(encode("prompt", {"prompt": "the mustard bottle"})) == ["the mustard bottle"]
    assert parse_prompts(pa.array([], type=pa.string())) == []


def test_seg_masks_message_shape():
    cam = {
        "name": "scene",
        "stamp": 3.0,
        "height": 4,
        "width": 5,
        "masks": np.zeros((2, 4, 5), np.uint8),
        "prompt_idx": np.array([0, 0], np.int32),
        "scores": np.array([0.9, 0.5], np.float32),
        "boxes": np.zeros((2, 4), np.float32),
    }
    t, body = decode(encode_seg_masks(stamp=3.0, prompts=["mug"], prompt_version=4, cameras=[cam]))
    assert t == "seg_masks" and body["prompt_version"] == 4
    assert body["cameras"][0]["masks"].shape == (2, 4, 5)


def test_seg_masks_empty_instances():
    cam = {"name": "s", "stamp": 1.0, "height": 2, "width": 2, "masks": np.zeros((0, 2, 2), np.uint8),
           "prompt_idx": np.zeros(0, np.int32), "scores": np.zeros(0, np.float32), "boxes": np.zeros((0, 4), np.float32)}
    _, body = decode(encode_seg_masks(stamp=1.0, prompts=["x"], prompt_version=1, cameras=[cam]))
    assert body["cameras"][0]["masks"].shape == (0, 2, 2)


def test_latest_frame_sync_leader_triggers():
    sync = LatestFrameSync(["a", "b"])
    assert sync.push("b", {"i": 1}) is None  # follower alone
    assert sync.push("a", {"i": 2}) == [{"i": 2}, {"i": 1}]
    assert sync.push("b", {"i": 3}) is None  # follower update never triggers
    assert sync.push("a", {"i": 4}) == [{"i": 4}, {"i": 3}]
    assert sync.push("zzz", {}) is None
    with pytest.raises(ValueError):
        LatestFrameSync([])


def test_is_camera_input():
    assert is_camera_input("cam_scene", [])
    assert is_camera_input("image_left", [])
    assert not is_camera_input("prompt", [])
    assert is_camera_input("scene", ["scene"]) and not is_camera_input("cam_x", ["scene"])


@pytest.mark.parametrize("k", [0, 3])
def test_detections_message_roundtrip(k):
    cam = {"name": "scene", "stamp": 7.25, "height": 480, "width": 640,
           "boxes": np.arange(4 * k, dtype=np.float32).reshape(k, 4),
           "scores": np.linspace(0.9, 0.1, k).astype(np.float32), "class_idx": np.arange(k, dtype=np.int32)}
    t, body = decode(encode_detections(stamp=7.25, prompts=["mug", "can", "bottle"], prompt_version=2, cameras=[cam]))
    assert t == "detections" and body["stamp"] == 7.25 and body["prompts"] == ["mug", "can", "bottle"]
    c = body["cameras"][0]
    assert c["boxes"].shape == (k, 4) and c["class_idx"].dtype == np.int32
    np.testing.assert_array_equal(c["scores"], cam["scores"])
