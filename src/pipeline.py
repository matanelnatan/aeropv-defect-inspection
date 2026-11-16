"""End-to-End Two-Stage Cascaded Inspection Pipeline with Calibrated Early Exit & Static INT8 Support."""

import argparse
from dataclasses import dataclass
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms

from src.models import load_stage1_classifier, load_stage2_segmenter
from src.preprocess import auto_crop_black_borders, unpad_and_resize_mask

CLS_CLASSES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage"]
HEALTHY_INDICES = [1, 2]
DEFECT_INDICES = [0, 3, 4]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FrameInspectionResult:
    pred_class: str
    calibrated_confidence: float
    healthy_prob_mass: float
    stage2_triggered: bool
    defect_area_pct: float
    severity: str
    latency_ms: float
    mask: np.ndarray | None = None


class CascadedSolarInspector:
    """Calibrated two-stage cascaded inspector supporting PyTorch and Static QDQ INT8 ONNX Runtime."""

    def __init__(
        self,
        cls_weights: str,
        seg_weights: str,
        backend: str = "onnx",
        device: str = "cpu",
        temperature: float = 1.0217,
        healthy_exit_threshold: float = 0.85,
    ):
        self.backend = backend
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.healthy_exit_threshold = healthy_exit_threshold

        self.cls_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.seg_tf = A.Compose([
            A.LongestMaxSize(max_size=384),
            A.PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        if backend == "onnx":
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.cls_sess = ort.InferenceSession(cls_weights, providers=providers)
            self.seg_sess = ort.InferenceSession(seg_weights, providers=providers)
        else:
            self.classifier = load_stage1_classifier("convnext_tiny", len(CLS_CLASSES), cls_weights, str(self.device))
            self.segmenter = load_stage2_segmenter("resnet34", 4, seg_weights, str(self.device))

    def inspect_frame(self, frame_bgr: np.ndarray) -> FrameInspectionResult:
        t0 = time.perf_counter()
        clean_bgr = auto_crop_black_borders(frame_bgr)
        h, w = clean_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: Temperature-Calibrated Classification Triage
        inp_cls = self.cls_tf(frame_rgb).unsqueeze(0)
        if self.backend == "onnx":
            logits = self.cls_sess.run(None, {"input": inp_cls.numpy()})[0][0] / self.temperature
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
        else:
            with torch.no_grad():
                logits_t = self.classifier(inp_cls.to(self.device)) / self.temperature
                probs = torch.softmax(logits_t, dim=1)[0].cpu().numpy()

        p_healthy = float(probs[HEALTHY_INDICES].sum())
        pred_idx = int(np.argmax(probs))
        pred_label = CLS_CLASSES[pred_idx]
        conf = float(probs[pred_idx])

        # Asymmetric Recall-Driven Routing: skip Stage 2 only when calibrated P(Clean+Dusty) >= threshold
        if p_healthy >= self.healthy_exit_threshold:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return FrameInspectionResult(
                pred_label, conf, p_healthy, False, 0.0, "NONE (EARLY EXIT)", latency_ms, None
            )

        # Stage 2: Conditional Pixel-Level Defect Segmentation
        defect_label = (
            pred_label
            if pred_idx in DEFECT_INDICES
            else CLS_CLASSES[DEFECT_INDICES[int(np.argmax(probs[DEFECT_INDICES]))]]
        )
        defect_conf = float(probs[CLS_CLASSES.index(defect_label)])

        inp_seg = self.seg_tf(image=frame_rgb)["image"].unsqueeze(0)
        if self.backend == "onnx":
            seg_logits = self.seg_sess.run(None, {"input": inp_seg.numpy()})[0][0]
            seg_pad = np.argmax(seg_logits, axis=0)
        else:
            with torch.no_grad():
                seg_pad = torch.argmax(self.segmenter(inp_seg.to(self.device)), dim=1)[0].cpu().numpy()

        mask = unpad_and_resize_mask(seg_pad, h, w, 384)
        area_pct = (np.count_nonzero(mask) / float(h * w)) * 100.0
        severity = "HIGH" if area_pct >= 5.0 else ("MEDIUM" if area_pct >= 1.5 else "LOW")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return FrameInspectionResult(defect_label, defect_conf, p_healthy, True, area_pct, severity, latency_ms, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AeroPV Calibrated Cascaded Inspection on an image.")
    parser.add_argument("--image", required=True, help="Path to input aerial solar panel image")
    parser.add_argument("--cls-weights", required=True, help="Path to Stage-1 classifier checkpoint (.onnx or .pth)")
    parser.add_argument("--seg-weights", required=True, help="Path to Stage-2 segmenter checkpoint (.onnx or .pth)")
    parser.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    args = parser.parse_args()

    inspector = CascadedSolarInspector(args.cls_weights, args.seg_weights, backend=args.backend)
    img = cv2.imread(args.image)
    res = inspector.inspect_frame(img)
    print(
        f"Prediction: {res.pred_class} ({res.calibrated_confidence*100:.1f}%) | "
        f"P_healthy: {res.healthy_prob_mass*100:.1f}% | Stage-2 Seg: {res.stage2_triggered} | "
        f"Defect Area: {res.defect_area_pct:.2f}% | Severity: {res.severity} | Latency: {res.latency_ms:.2f} ms"
    )
