#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FAILURE_IMPORT = "import_error"
FAILURE_MODEL_LOAD = "model_load_error"
FAILURE_ADAPTER_APPLY = "adapter_mismatch"
FAILURE_NO_DELTA = "no_delta"
FAILURE_RESET = "reset_failed"
FAILURE_RUNTIME = "runtime_error"


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
    parser.add_argument("--strict", action="store_true", help="Fail if no adapter delta or failed reset")
    parser.add_argument("--out", default="-", help="JSON output path ('-' for stdout)")
    return parser.parse_args()


def _write_output(payload: dict[str, object], out: str) -> None:
    text = json.dumps(payload, indent=2)
    if out == "-":
        print(text)
    else:
        Path(out).write_text(text + "\n")


def main() -> int:
    args = parse_args()

    result: dict[str, object] = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "status": "failed",
        "failure_category": None,
    }

    try:
        from heretic.backends import Exl3Backend
    except Exception as error:
        result["failure_category"] = FAILURE_IMPORT
        result["error"] = str(error)
        _write_output(result, args.out)
        return 2

    backend = Exl3Backend()

    try:
        backend.load_model(args.model_path)
        backend.load_tokenizer(None)
    except Exception as error:
        result["failure_category"] = FAILURE_MODEL_LOAD
        result["error"] = str(error)
        _write_output(result, args.out)
        return 1

    try:
        baseline = backend.generate_raw_text(args.prompt, max_new_tokens=args.max_new_tokens)
        result["baseline_text"] = baseline

        try:
            backend.apply_adapter(args.adapter_path)
        except Exception as error:
            result["failure_category"] = FAILURE_ADAPTER_APPLY
            result["error"] = str(error)
            _write_output(result, args.out)
            return 1

        adapted = backend.generate_raw_text(args.prompt, max_new_tokens=args.max_new_tokens)
        result["adapted_text"] = adapted

        backend.reset_adapters()
        reset = backend.generate_raw_text(args.prompt, max_new_tokens=args.max_new_tokens)
        result["reset_text"] = reset

        adapter_changed_output = baseline != adapted
        reset_matches_baseline = baseline == reset

        result["adapter_changed_output"] = adapter_changed_output
        result["reset_matches_baseline"] = reset_matches_baseline

        if args.strict and not adapter_changed_output:
            result["failure_category"] = FAILURE_NO_DELTA
            _write_output(result, args.out)
            return 1

        if args.strict and not reset_matches_baseline:
            result["failure_category"] = FAILURE_RESET
            _write_output(result, args.out)
            return 1

        result["status"] = "ok"
        _write_output(result, args.out)
        return 0

    except Exception as error:
        result["failure_category"] = FAILURE_RUNTIME
        result["error"] = str(error)
        _write_output(result, args.out)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
