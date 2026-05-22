#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke-test the EXL3 backend's LoRA injection path.

Verifies that:

  1. Baseline generation works.
  2. Synthesizing a non-trivial rank-1 LoRA via Exl3Model.abliterate
     changes the generation output (delta != 0).
  3. Exl3Model.reset_model() restores the baseline output exactly.

This exercises the in-place ``lora_a_tensors`` / ``lora_b_tensors``
mutation path that Heretic's optimization loop relies on. It does NOT
require an external PEFT adapter on disk — we generate the LoRA from a
random "refusal direction".

Usage:
    PYTHONPATH=src python scripts/smoke_test_exl3_lora.py /path/to/exl3-model
    PYTHONPATH=src python scripts/smoke_test_exl3_lora.py /path/to/exl3-model \\
        --prompt "Explain why the sky is blue." --strict --out artifacts/smoke.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXL3 LoRA smoke test")
    parser.add_argument("model_path", help="Path to EXL3 model directory")
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=1.0,
        help="Kernel weight for the synthesized abliteration delta. Larger = bigger change.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if (a) LoRA didn't change output, or (b) reset didn't restore it.",
    )
    parser.add_argument("--out", default="-", help="JSON output path ('-' for stdout)")
    parser.add_argument("--max-num-tokens", type=int, default=2048)
    return parser.parse_args()


def _write_output(payload: dict, path: str) -> None:
    text = json.dumps(payload, indent=2)
    if path == "-":
        print(text)
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")


def main() -> int:
    args = parse_args()

    from heretic.config import Backend, RowNormalization, Settings
    from heretic.exl3_model import AbliterationParameters, Exl3Model

    result: dict = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "lambda": args.lambda_,
        "status": "failed",
        "failure_category": None,
    }

    # heretic.Settings uses pydantic-settings' CliSettingsSource which
    # re-reads sys.argv at construction time. Blank it around the call
    # so the script's own argparse args don't collide.
    import sys as _sys
    _orig_argv = _sys.argv
    _sys.argv = [_orig_argv[0]]
    try:
        try:
            settings = Settings(
                model=args.model_path,
                backend=Backend.EXL3,
                exl3_max_num_tokens=args.max_num_tokens,
                row_normalization=RowNormalization.NONE,
                batch_size=1,
            )
        except Exception as error:
            result["failure_category"] = "settings_error"
            result["error"] = str(error)
            _write_output(result, args.out)
            return 2
    finally:
        _sys.argv = _orig_argv

    try:
        model = Exl3Model(settings)
    except Exception as error:
        result["failure_category"] = "model_load_error"
        result["error"] = str(error)
        _write_output(result, args.out)
        return 1

    try:
        # Step 1: baseline generation through the Generator path.
        from heretic.utils import Prompt
        prompts = [Prompt(system="You are a helpful assistant.", user=args.prompt)]
        baseline = model.get_responses(prompts)[0]
        result["baseline_text"] = baseline

        # Step 2: synthesize a deterministic "refusal direction" and call abliterate.
        import torch
        # refusal_directions shape: (n_layers + 1, hidden_size). The +1 is for the
        # embedding-layer direction; abliterate uses [layer_index + 1] when
        # direction_index is None, so all per-layer directions get exercised.
        # We hand it the same direction for every layer to keep things deterministic.
        n_layers = len(model.get_layers())
        # Infer hidden_size from the first o_proj's out_features_unpadded.
        hidden_size = None
        for per_layer in model._layer_modules:
            o_modules = per_layer.get("attn.o_proj") or per_layer.get("mlp.down_proj")
            if o_modules:
                hidden_size = o_modules[0].out_features_unpadded
                break
        if hidden_size is None:
            raise RuntimeError("Could not infer hidden_size from discovered modules.")

        torch.manual_seed(0)
        v = torch.randn(hidden_size)
        v = torch.nn.functional.normalize(v, p=2, dim=0)
        refusal_directions = v.unsqueeze(0).expand(n_layers + 1, -1).contiguous()

        params: dict = {}
        for component in model.get_abliterable_components():
            params[component] = AbliterationParameters(
                max_weight=args.lambda_,
                max_weight_position=n_layers // 2,
                min_weight=args.lambda_,
                min_weight_distance=n_layers,  # ensure all layers are in support
            )

        model.abliterate(refusal_directions, None, params)

        # Step 3: regenerate with the LoRA active.
        adapted = model.get_responses(prompts)[0]
        result["adapted_text"] = adapted

        # Step 4: reset and regenerate to confirm restoration.
        model.reset_model()
        reset_text = model.get_responses(prompts)[0]
        result["reset_text"] = reset_text

        adapter_changed = baseline != adapted
        reset_matches = baseline == reset_text
        result["adapter_changed_output"] = adapter_changed
        result["reset_matches_baseline"] = reset_matches

        if args.strict and not adapter_changed:
            result["failure_category"] = "no_delta"
            _write_output(result, args.out)
            return 1
        if args.strict and not reset_matches:
            result["failure_category"] = "reset_failed"
            _write_output(result, args.out)
            return 1

        result["status"] = "ok"
        _write_output(result, args.out)
        return 0
    except Exception as error:
        import traceback
        result["failure_category"] = "runtime_error"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        _write_output(result, args.out)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
