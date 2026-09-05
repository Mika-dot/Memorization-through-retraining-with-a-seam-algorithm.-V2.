from __future__ import annotations

import numpy as np
from PIL import Image

from mtr3.codec import decode, encode


def test_roundtrip(tmp_path):
    y, x = np.mgrid[:24, :32]
    rgb = np.stack([(x * 8) % 256, (y * 10) % 256, ((x + y) * 5) % 256], axis=-1).astype(np.uint8)
    src = tmp_path / "src.png"
    bit = tmp_path / "src.mtr3"
    out = tmp_path / "out.png"
    Image.fromarray(rgb).save(src)

    info = encode(src, bit, preset="fast", device="cpu", steps=3, quiet=True)
    dec = decode(bit, out, device="cpu")

    assert bit.read_bytes().startswith(b"MTR3\x01")
    assert Image.open(out).size == (32, 24)
    assert info["bytes"] > 0
    assert dec["width"] == 32
