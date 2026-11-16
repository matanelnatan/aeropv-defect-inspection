"""Temporal IoU defect tracker for deduplicating aerial video detections into maintenance events."""

import numpy as np


def box_iou(b1: list[int], b2: list[int]) -> float:
    x_a, y_a = max(b1[0], b2[0]), max(b1[1], b2[1])
    x_b, y_b = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(area1 + area2 - inter)


class TemporalDefectTracker:
    """Consolidates consecutive per-frame masks into unique physical PV defect records."""

    def __init__(self, max_frame_gap: int = 6, min_iou_match: float = 0.12, min_confirmed_area_pct: float = 0.05):
        self.max_frame_gap = max_frame_gap
        self.min_iou_match = min_iou_match
        self.min_confirmed_area_pct = min_confirmed_area_pct
        self.tracks: list[dict] = []
        self._next_id = 1

    def update(self, frame_idx: int, defect_class: str, conf: float, mask: np.ndarray, area_pct: float):
        if area_pct < self.min_confirmed_area_pct:
            return None, None

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None, None
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        matched = None
        best_iou = 0.0
        for tr in self.tracks:
            if tr["defect_class"] == defect_class and (frame_idx - tr["end_frame"]) <= self.max_frame_gap:
                score = box_iou(tr["last_bbox"], bbox)
                if score >= self.min_iou_match and score >= best_iou:
                    best_iou = score
                    matched = tr

        if matched is None:
            matched = {
                "track_id": self._next_id,
                "defect_class": defect_class,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "frame_count": 1,
                "last_bbox": bbox,
                "peak_area_pct": round(area_pct, 2),
                "peak_confidence": round(conf * 100.0, 1),
            }
            self.tracks.append(matched)
            self._next_id += 1
        else:
            matched["end_frame"] = frame_idx
            matched["frame_count"] += 1
            matched["last_bbox"] = bbox
            matched["peak_area_pct"] = max(matched["peak_area_pct"], round(area_pct, 2))
            matched["peak_confidence"] = max(matched["peak_confidence"], round(conf * 100.0, 1))

        return matched["track_id"], bbox

    def finalize_report(self, video_name: str, total_frames: int, fps: float) -> dict:
        events = []
        for idx, tr in enumerate(self.tracks, start=1):
            sev = "HIGH" if tr["peak_area_pct"] >= 5.0 else ("MEDIUM" if tr["peak_area_pct"] >= 1.5 else "LOW")
            events.append({
                "defect_id": f"PV-DEF-{idx:02d}",
                "defect_type": tr["defect_class"],
                "timestamp_sec": f"{tr['start_frame']/fps:.1f}s - {tr['end_frame']/fps:.1f}s",
                "frames_observed": tr["frame_count"],
                "peak_affected_area_pct": tr["peak_area_pct"],
                "peak_calibrated_confidence_pct": tr["peak_confidence"],
                "maintenance_severity": sev,
            })
        return {
            "video_source": video_name,
            "processed_frames": total_frames,
            "unique_defects_detected": len(events),
            "defect_events": events,
        }
