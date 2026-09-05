from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    bands: int = 6
    hidden: int = 64
    depth: int = 3
    latent_channels: int = 6
    grid_divisors: tuple[int, ...] = (64, 32, 16)


class FourierFeatures(nn.Module):
    def __init__(self, bands: int) -> None:
        super().__init__()
        self.bands = int(bands)
        freqs = (2.0 ** torch.arange(self.bands, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_features(self) -> int:
        return 2 + 4 * self.bands

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: [N, 2] in [-1, 1]
        phase = xy[..., None] * self.freqs
        enc = [xy, torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)]
        return torch.cat(enc, dim=-1)


class MTR3Field(nn.Module):
    """Tiny per-image implicit decoder with multi-resolution latent feature grids."""

    def __init__(self, height: int, width: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.cfg = cfg
        self.fourier = FourierFeatures(cfg.bands)

        grids: list[nn.Parameter] = []
        for div in cfg.grid_divisors:
            gh = max(2, math.ceil(self.height / div))
            gw = max(2, math.ceil(self.width / div))
            p = nn.Parameter(torch.zeros(1, cfg.latent_channels, gh, gw))
            nn.init.normal_(p, mean=0.0, std=0.01)
            grids.append(p)
        self.grids = nn.ParameterList(grids)

        in_features = self.fourier.out_features + cfg.latent_channels * len(grids)
        layers: list[nn.Module] = []
        d = in_features
        for _ in range(cfg.depth):
            layers += [nn.Linear(d, cfg.hidden), nn.GELU()]
            d = cfg.hidden
        layers += [nn.Linear(d, 3)]
        self.decoder = nn.Sequential(*layers)

    def _sample_grids(self, xy: torch.Tensor, enabled_levels: int | None = None) -> torch.Tensor:
        # grid_sample wants [N, Hout, Wout, 2]. A single row keeps this vectorized.
        nlevels = len(self.grids) if enabled_levels is None else max(0, min(enabled_levels, len(self.grids)))
        sample_grid = xy.view(1, -1, 1, 2)
        sampled: list[torch.Tensor] = []
        for i, grid in enumerate(self.grids):
            if i < nlevels:
                z = F.grid_sample(grid, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
                sampled.append(z[0, :, :, 0].T)
            else:
                sampled.append(torch.zeros((xy.shape[0], self.cfg.latent_channels), device=xy.device, dtype=xy.dtype))
        return torch.cat(sampled, dim=-1)

    def forward(self, xy: torch.Tensor, enabled_levels: int | None = None) -> torch.Tensor:
        f = torch.cat([self.fourier(xy), self._sample_grids(xy, enabled_levels)], dim=-1)
        return torch.sigmoid(self.decoder(f))


def make_xy(height: int, width: int, device: torch.device | str = "cpu") -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def iter_named_float_tensors(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor):
            yield name, tensor
