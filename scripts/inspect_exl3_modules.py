#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect the module structure of an EXL3-quantized model.

Dumps every module ``.key`` discovered by walking the loaded model, and
shows which of them Heretic groups under ``attn.o_proj`` / ``mlp.down_proj``.
Useful for confirming the key regex covers a new architecture before
running the full optimization loop.

Usage:
    PYTHONPATH=src python scripts/inspect_exl3_modules.py /path/to/exl3-model
    PYTHONPATH=src python scripts/inspect_exl3_modules.py /path/to/exl3-model --out modules.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect EXL3 model module structure")
    parser.add_argument("model_path", help="Path to EXL3 model directory")
    parser.add_argument(
        "--out",
        default="-",
        help="JSON output path ('-' for stdout, default: stdout)",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=512,
        help="Cache size; smaller is fine for module inspection. Must be a multiple of 256.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # We construct an Exl3Model directly with a synthesized Settings
    # because the full Settings loader runs CLI parsing.
    from heretic.config import Backend, Settings
    from heretic.exl3_model import Exl3Model

    settings = Settings(
        model=args.model_path,
        backend=Backend.EXL3,
        exl3_max_num_tokens=args.max_num_tokens,
    )
    model = Exl3Model(settings)

    # Group target Linears by layer (and component, the way Heretic uses them).
    per_layer: list[dict[str, list[str]]] = []
    for layer_index, components in enumerate(model._layer_modules):
        per_layer.append(
            {component: [m.key for m in modules] for component, modules in components.items()}
        )

    target_keys = sorted(
        key
        for layer in per_layer
        for keys in layer.values()
        for key in keys
    )

    payload = {
        "model_path": args.model_path,
        "num_layers_inferred": len(model._layer_modules),
        "components_seen": sorted(
            {c for layer in per_layer for c in layer.keys()}
        ),
        "target_modules_per_layer": per_layer,
        "target_module_count": len(target_keys),
        "target_modules": target_keys,
        "all_module_count": len(model._all_module_keys),
        "all_modules": model._all_module_keys,
    }

    text = json.dumps(payload, indent=2)
    if args.out == "-":
        print(text)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"Wrote inspection report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
