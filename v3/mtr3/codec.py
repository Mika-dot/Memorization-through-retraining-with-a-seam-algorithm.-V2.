from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .bitstream import load_bitstream, save_bitstream
from .model import MTR3Field, ModelConfig, make_xy, parameter_count
from .seam import seam_importance


@dataclass(frozen=True)
class EncodePreset:
    model: ModelConfig
    steps: int
    batch: int
    lr: float
    rate_lambda: float
    patience: int


PRESETS: dict[str, EncodePreset] = {
    "fast": EncodePreset(ModelConfig(bands=5, hidden=48, depth=2, latent_channels=4, grid_divisors=(64, 32, 16)), 800, 32768, 2e-3, 2e-7, 160),
    "balanced": EncodePreset(ModelConfig(bands=6, hidden=64, depth=3, latent_channels=6, grid_divisors=(64, 32, 16)), 2400, 65536, 1.5e-3, 1e-7, 300),
    "max": EncodePreset(ModelConfig(bands=7, hidden=96, depth=4, latent_channels=8, grid_divisors=(96, 48, 24, 12)), 6000, 98304, 1e-3, 5e-8, 600),
}


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def encode(
    input_path: str | Path,
    output_path: str | Path,
    preset: str = "balanced",
    device: str = "auto",
    steps: int | None = None,
    seed: int = 7,
    quiet: bool = False,
) -> dict[str, float | int | str]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(PRESETS)}")
    p = PRESETS[preset]
    total_steps = int(steps or p.steps)
    dev = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    img = Image.open(input_path).convert("RGB")
    rgb_u8 = np.asarray(img)
    h, w = rgb_u8.shape[:2]
    target = torch.from_numpy(rgb_u8.copy()).to(dev, dtype=torch.float32).reshape(-1, 3) / 255.0
    xy = make_xy(h, w, dev)

    importance_np = seam_importance(rgb_u8)
    importance = torch.from_numpy(importance_np.reshape(-1)).to(dev)
    probs = importance / importance.sum()

    model = MTR3Field(h, w, p.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, betas=(0.9, 0.99), weight_decay=0.0)

    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = float("inf")
    stale = 0
    t0 = time.perf_counter()

    for step in range(total_steps):
        frac = step / max(1, total_steps - 1)
        if frac < 0.25:
            enabled_levels = 1
        elif frac < 0.55:
            enabled_levels = min(2, len(model.grids))
        else:
            enabled_levels = len(model.grids)

        n = min(p.batch, target.shape[0])
        idx = torch.multinomial(probs, n, replacement=True)
        xyb = xy[idx]
        yb = target[idx]
        wb = importance[idx]

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            pred = model(xyb, enabled_levels=enabled_levels)
            pixel_loss = torch.mean(torch.abs(pred - yb) * wb[:, None])
            rate_proxy = torch.stack([g.abs().mean() for g in model.grids]).mean()
            loss = pixel_loss + p.rate_lambda * rate_proxy
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        current = float(loss.detach().cpu())
        if current + 1e-6 < best:
            best = current
            stale = 0
        else:
            stale += 1
        if stale >= p.patience and step > total_steps // 2:
            total_steps = step + 1
            break
        if not quiet and (step == 0 or (step + 1) % max(50, total_steps // 20) == 0):
            print(f"step {step + 1:5d}/{total_steps}  loss={current:.6f}")

    train_seconds = time.perf_counter() - t0
    size = save_bitstream(output_path, model, metadata={"source": Path(input_path).name, "preset": preset, "steps": total_steps})

    qmodel, _ = load_bitstream(output_path, dev)
    with torch.inference_mode():
        recon = _render(qmodel, dev, chunk=262144)
    psnr = _psnr(recon, target)

    return {
        "width": w,
        "height": h,
        "parameters": parameter_count(model),
        "bytes": size,
        "bpp": (size * 8.0) / (w * h),
        "psnr_db": psnr,
        "encode_seconds": train_seconds,
        "steps": total_steps,
        "device": str(dev),
    }


def _render(model: MTR3Field, device: torch.device, chunk: int = 262144) -> torch.Tensor:
    xy = make_xy(model.height, model.width, device)
    parts: list[torch.Tensor] = []
    for i in range(0, xy.shape[0], chunk):
        parts.append(model(xy[i:i + chunk]))
    return torch.cat(parts, dim=0)


def decode(input_path: str | Path, output_path: str | Path, device: str = "auto") -> dict[str, float | int | str]:
    dev = _device(device)
    model, _ = load_bitstream(input_path, dev)
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = _render(model, dev)
    elapsed = time.perf_counter() - t0
    arr = (pred.reshape(model.height, model.width, 3).clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    Image.fromarray(arr, mode="RGB").save(output_path)
    return {"width": model.width, "height": model.height, "decode_seconds": elapsed, "device": str(dev)}
