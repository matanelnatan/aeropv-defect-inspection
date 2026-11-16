"""Automated unit tests for preprocessing, temporal tracking, and cascaded inference."""

import numpy as np
import torch

from src.models import load_stage1_classifier, load_stage2_segmenter
from src.pipeline import CascadedSolarInspector
from src.preprocess import auto_crop_black_borders, compute_sharpness_score, unpad_and_resize_mask
from src.tracker import TemporalDefectTracker


def test_auto_crop_black_borders():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[30:170, 40:260] = 180
    cropped = auto_crop_black_borders(img, threshold=12)
    assert cropped.shape == (140, 220, 3)


def test_compute_sharpness_score():
    blurry = np.full((64, 64, 3), 128, dtype=np.uint8)
    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255
    assert compute_sharpness_score(sharp) > compute_sharpness_score(blurry)


def test_unpad_and_resize_mask():
    padded = np.zeros((384, 384), dtype=np.uint8)
    padded[100:250, 80:300] = 1
    restored = unpad_and_resize_mask(padded, orig_h=540, orig_w=960, target_size=384)
    assert restored.shape == (540, 960)
    assert restored.max() == 1


def test_temporal_defect_tracker_consolidation():
    tracker = TemporalDefectTracker(max_frame_gap=6, min_iou_match=0.12, min_confirmed_area_pct=0.05)
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:60, 20:60] = 1
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[22:62, 22:62] = 1

    id1, _ = tracker.update(frame_idx=0, defect_class="Physical-Damage", conf=0.92, mask=mask1, area_pct=4.5)
    id2, _ = tracker.update(frame_idx=3, defect_class="Physical-Damage", conf=0.95, mask=mask2, area_pct=6.2)
    assert id1 == id2 == 1

    report = tracker.finalize_report("test.mp4", total_frames=10, fps=30.0)
    assert report["unique_defects_detected"] == 1
    assert report["defect_events"][0]["maintenance_severity"] == "HIGH"


def test_cascaded_inspector_end_to_end(tmp_path):
    cls_path = tmp_path / "cls.pth"
    seg_path = tmp_path / "seg.pth"
    torch.save(load_stage1_classifier("convnext_tiny", 5).state_dict(), cls_path)
    torch.save(load_stage2_segmenter("resnet34", 4).state_dict(), seg_path)

    inspector = CascadedSolarInspector(
        str(cls_path), str(seg_path), backend="pytorch", device="cpu", healthy_exit_threshold=0.999
    )
    dummy_frame = np.random.randint(20, 230, (240, 320, 3), dtype=np.uint8)
    res = inspector.inspect_frame(dummy_frame)
    assert res.stage2_triggered is True
    assert res.mask.shape == (240, 320)
    assert res.latency_ms > 0
