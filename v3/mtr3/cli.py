from __future__ import annotations

import argparse
import json

from .codec import decode, encode


def main() -> None:
    parser = argparse.ArgumentParser(prog="mtr3", description="MTRSA V3 experimental per-image neural codec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="train a compact per-image representation and write .mtr3")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--preset", choices=["fast", "balanced", "max"], default="balanced")
    enc.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    enc.add_argument("--steps", type=int, default=None, help="override preset optimization steps")
    enc.add_argument("--quiet", action="store_true")

    dec = sub.add_parser("decode", help="decode .mtr3 to a conventional image")
    dec.add_argument("input")
    dec.add_argument("output")
    dec.add_argument("--device", default="auto")

    args = parser.parse_args()
    if args.cmd == "encode":
        result = encode(args.input, args.output, args.preset, args.device, args.steps, quiet=args.quiet)
    else:
        result = decode(args.input, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
