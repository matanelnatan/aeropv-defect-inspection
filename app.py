"""Interactive Gradio Web App for AeroPV Calibrated Cascaded Solar Panel Inspection."""

import os
import cv2
import gradio as gr
import numpy as np

from src.pipeline import CascadedSolarInspector

CLS_ONNX = os.environ.get("AEROPV_CLS_WEIGHTS", "models/convnext_tiny_static_int8.onnx")
SEG_ONNX = os.environ.get("AEROPV_SEG_WEIGHTS", "models/unet_resnet34_static_int8.onnx")

COLOR_MAP_RGB = {
    1: np.array([255, 60, 60], dtype=np.uint8),    # Physical-Damage (Red)
    2: np.array([255, 190, 0], dtype=np.uint8),    # Electrical-damage (Amber)
    3: np.array([0, 210, 255], dtype=np.uint8),    # Bird-drop (Cyan)
}

_inspector_cache = {}


def get_inspector(threshold: float) -> CascadedSolarInspector:
    if not (os.path.exists(CLS_ONNX) and os.path.exists(SEG_ONNX)):
        raise FileNotFoundError(
            f"ONNX checkpoints not found at '{CLS_ONNX}' and '{SEG_ONNX}'. "
            "Download the Static QDQ INT8 files from GitHub Releases into models/."
        )
    if "inst" not in _inspector_cache:
        _inspector_cache["inst"] = CascadedSolarInspector(CLS_ONNX, SEG_ONNX, backend="onnx", device="cpu")
    _inspector_cache["inst"].healthy_exit_threshold = threshold
    return _inspector_cache["inst"]


def run_inspection(image_rgb: np.ndarray, healthy_threshold: float):
    if image_rgb is None:
        return None, {}
    inspector = get_inspector(healthy_threshold)
    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    res = inspector.inspect_frame(frame_bgr)

    vis = image_rgb.copy()
    if res.stage2_triggered and res.mask is not None:
        h, w = vis.shape[:2]
        msk = cv2.resize(res.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        for cid, col in COLOR_MAP_RGB.items():
            vis[msk == cid] = (0.45 * vis[msk == cid] + 0.55 * col).astype(np.uint8)

    telemetry = {
        "Stage-1 Prediction": res.pred_class,
        "Calibrated Confidence (%)": round(res.calibrated_confidence * 100.0, 2),
        "Calibrated Healthy Mass P(Clean+Dusty) (%)": round(res.healthy_prob_mass * 100.0, 2),
        "Stage-2 Segmentation Triggered": res.stage2_triggered,
        "Damaged Area (%)": round(res.defect_area_pct, 2),
        "Maintenance Severity": res.severity,
        "End-to-End CPU Latency (ms)": round(res.latency_ms, 2),
    }
    return vis, telemetry


samples = (
    [[os.path.join("assets", "samples", fn), 0.85] for fn in sorted(os.listdir("assets/samples"))]
    if os.path.exists("assets/samples")
    else []
)

demo = gr.Interface(
    fn=run_inspection,
    inputs=[
        gr.Image(type="numpy", label="Aerial Solar Panel Image (RGB)"),
        gr.Slider(0.50, 0.98, value=0.85, step=0.01, label="Early-Exit Safety Threshold P(Clean + Dusty)"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Defect Segmentation Overlay"),
        gr.JSON(label="Inspection & Calibration Telemetry"),
    ],
    title="AeroPV: Calibrated Two-Stage Aerial Solar Panel Inspection",
    description="Static QDQ INT8 ONNX Runtime demo featuring Temperature-Calibrated Early Exit (T=1.0217) and conditional U-Net (ResNet34) defect segmentation.",
    examples=samples,
)

if __name__ == "__main__":
    demo.launch()
