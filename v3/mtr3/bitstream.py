from __future__ import annotations

import io
import json
import struct
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import MTR3Field, ModelConfig

MAGIC = b"MTR3\x01"


def _quantize_symmetric(t: torch.Tensor) -> tuple[np.ndarray, float]:
    a = t.detach().cpu().float().numpy()
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
    return q, scale


def save_bitstream(path: str | Path, model: MTR3Field, metadata: dict[str, Any] | None = None, level: int = 9) -> int:
    entries: list[dict[str, Any]] = []
    payload = io.BytesIO()
    for name, tensor in model.state_dict().items():
        if not torch.is_floating_point(tensor):
            continue
        q, scale = _quantize_symmetric(tensor)
        raw = q.tobytes(order="C")
        entries.append({
            "name": name,
            "shape": list(q.shape),
            "scale": scale,
            "offset": payload.tell(),
            "length": len(raw),
        })
        payload.write(raw)

    compressed = zlib.compress(payload.getvalue(), level=level)
    header = {
        "height": model.height,
        "width": model.width,
        "model": asdict(model.cfg),
        "tensors": entries,
        "payload_codec": "zlib",
        "metadata": metadata or {},
    }
    header_raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    out = MAGIC + struct.pack("<I", len(header_raw)) + header_raw + compressed
    Path(path).write_bytes(out)
    return len(out)


def load_bitstream(path: str | Path, device: str | torch.device = "cpu") -> tuple[MTR3Field, dict[str, Any]]:
    raw = Path(path).read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError("Not an MTR3 bitstream")
    pos = len(MAGIC)
    (header_len,) = struct.unpack("<I", raw[pos:pos + 4])
    pos += 4
    header = json.loads(raw[pos:pos + header_len].decode("utf-8"))
    pos += header_len
    payload = zlib.decompress(raw[pos:])

    cfg_raw = header["model"]
    cfg_raw["grid_divisors"] = tuple(cfg_raw["grid_divisors"])
    cfg = ModelConfig(**cfg_raw)
    model = MTR3Field(header["height"], header["width"], cfg).to(device)
    state = model.state_dict()

    for e in header["tensors"]:
        q = np.frombuffer(payload[e["offset"]: e["offset"] + e["length"]], dtype=np.int8).copy()
        q = q.reshape(e["shape"])
        t = torch.from_numpy(q.astype(np.float32) * float(e["scale"]))
        state[e["name"]] = t.to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, header
