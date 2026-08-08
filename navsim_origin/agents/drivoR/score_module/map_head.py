import math

import torch
import torch.nn as nn


class MapHead(nn.Module):
    """Convert proposal-level tokens into dense BEV segmentation logits."""

    def __init__(self, config):
        super().__init__()

        self.d_model = config.tf_d_model
        self.num_classes = getattr(config, "bev_semantic_num_classes", 20)
        self.target_size = tuple(getattr(config, "bev_semantic_frame", (64, 64)))

        hidden_channels = min(128, self.d_model)
        self.proj = nn.Sequential(
            nn.Conv2d(self.d_model, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_classes, kernel_size=1),
        )

    def forward(self, keyval: torch.Tensor) -> torch.Tensor:
        if keyval.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got shape {tuple(keyval.shape)}")

        batch_size, num_tokens, channels = keyval.shape
        if channels != self.d_model:
            raise ValueError(f"Expected feature dim {self.d_model}, got {channels}")

        grid_size = int(math.isqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Token count {num_tokens} cannot be reshaped into a square grid")

        map_bev = keyval.reshape(batch_size, grid_size, grid_size, channels).permute(0, 3, 1, 2)
        logits = self.proj(map_bev)

        if logits.shape[-2:] != self.target_size:
            logits = nn.functional.interpolate(
                logits,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits
