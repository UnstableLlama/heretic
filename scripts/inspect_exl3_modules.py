#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect available EXL3 modules")
    parser.add_argument("model_path", help="Path to EXL3 model directory")
    parser.add_argument("--out", default="-", help="JSON output path ('-' for stdout)")
    return parser.parse_args()


def _role_from_name(module_name: str) -> str | None:
    if module_name.endswith(".o_proj"):
        return "attn_out"
    if module_name.endswith(".down_proj"):
        return "mlp_down"
    return None


def main() -> int:
    args = parse_args()
    from heretic.backends import Exl3Backend

    backend = Exl3Backend()
    backend.load_model(args.model_path)

    modules = backend.list_modules()
    target_modules = backend.list_target_modules()
    mapping = [{"module": name, "role": _role_from_name(name)} for name in target_modules]

    payload = {
        "model_path": args.model_path,
        "module_count": len(modules),
        "modules": modules,
        "target_module_count": len(target_modules),
        "target_modules": target_modules,
        "heretic_mapping": mapping,
    }

    text = json.dumps(payload, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
