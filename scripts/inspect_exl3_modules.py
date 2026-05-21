#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json

from heretic.backends import Exl3Backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect available EXL3 modules")
    parser.add_argument("model_path", help="Path to EXL3 model directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = Exl3Backend()
    backend.load_model(args.model_path)
    backend.load_tokenizer(None)

    modules = backend.list_modules()
    payload = {
        "model_path": args.model_path,
        "module_count": len(modules),
        "modules": modules,
        "target_modules": backend.list_target_modules(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
