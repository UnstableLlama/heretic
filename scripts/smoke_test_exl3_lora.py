#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test EXL3 runtime LoRA behavior (baseline -> adapter -> reset) "
            "for stop/go integration gating."
        )
    )
    parser.add_argument("model_path", help="Path to EXL3 model directory")
    parser.add_argument("--adapter-path", required=True, help="Path to PEFT adapter dir")
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", default="-", help="JSON output path ('-' for stdout)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from heretic.backends import Exl3Backend
    except Exception as error:
        print(f"ERROR: failed to import Exl3Backend: {error}", file=sys.stderr)
        return 2

    backend = Exl3Backend()

    result: dict[str, object] = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "prompt": args.prompt,
        "status": "failed",
    }

    try:
        backend.load_model(args.model_path)
        backend.load_tokenizer(None)

        baseline = backend.generate_raw_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        backend.apply_adapter(args.adapter_path)
        adapted = backend.generate_raw_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        backend.reset_adapters()
        reset = backend.generate_raw_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        result.update(
            {
                "status": "ok",
                "baseline_text": baseline,
                "adapted_text": adapted,
                "reset_text": reset,
                "adapter_changed_output": baseline != adapted,
                "reset_matches_baseline": baseline == reset,
            }
        )
    except Exception as error:
        result["error"] = str(error)

    payload = json.dumps(result, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload + "\n")

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
