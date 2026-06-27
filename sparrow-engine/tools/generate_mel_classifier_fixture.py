#!/usr/bin/env python3
"""Generate the tiny mel-input softmax audio-classifier ONNX fixture."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TinyMelClassifier(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, num_classes),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel)


def initialise_deterministic(model: TinyMelClassifier) -> None:
    with torch.no_grad():
        conv = model.net[0]
        linear = model.net[4]
        assert isinstance(conv, nn.Conv2d)
        assert isinstance(linear, nn.Linear)
        conv.weight.fill_(0.015)
        conv.bias.copy_(torch.tensor([0.02, -0.01, 0.03, 0.0], dtype=torch.float32))
        linear.weight.copy_(
            torch.tensor(
                [
                    [0.20, -0.10, 0.05, 0.12],
                    [-0.04, 0.18, 0.08, -0.02],
                    [0.10, 0.03, -0.12, 0.16],
                ],
                dtype=torch.float32,
            )
        )
        linear.bias.copy_(torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "sparrow-engine-core" / "tests" / "fixtures" / "audio" / "mel_classifier_tiny"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"

    torch.manual_seed(7)
    model = TinyMelClassifier(num_classes=3).eval()
    initialise_deterministic(model)
    example = torch.randn(2, 1, 64, 48, dtype=torch.float32)

    torch.onnx.export(
        model,
        example,
        model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
    )
    size = model_path.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"fixture too large: {size} bytes")
    print(f"wrote {model_path} ({size} bytes)")


if __name__ == "__main__":
    main()
