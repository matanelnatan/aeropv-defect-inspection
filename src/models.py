"""Model builders for Stage-1 Triage Classifier and Stage-2 Defect Segmenter."""

import segmentation_models_pytorch as smp
import timm
import torch


def load_stage1_classifier(
    arch: str = "convnext_tiny",
    num_classes: int = 5,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Instantiate and optionally load trained weights for Stage-1 triage classification."""
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    return model.to(device).eval()


def load_stage2_segmenter(
    encoder_name: str = "resnet34",
    num_classes: int = 4,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Instantiate and optionally load trained weights for Stage-2 pixel-level segmentation."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    return model.to(device).eval()
