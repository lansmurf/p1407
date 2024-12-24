import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ProjectorConfig:
    """Minimal config for vision->language projection"""
    vision_size: int = 1024
    language_size: int = 2048
    downsample_factor: float = 0.5
    eps: float = 1e-6

class VisionProjector(nn.Module):
    """Projects vision features to language space using InternVL's method"""
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        # Calculate downsample size (matches InternVL)
        downsample_size = int(config.vision_size * (1/config.downsample_factor)**2)
        
        self.layers = nn.Sequential(
            nn.LayerNorm(downsample_size, eps=config.eps),
            nn.Linear(downsample_size, config.language_size),
            nn.GELU(),
            nn.Linear(config.language_size, config.language_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects pre-downsampled features from vision encoder
        Args:
            x: [B, N, C] tensor where N is already downsampled
        Returns:
            [B, N, language_size] tensor
        """
        return self.layers(x)