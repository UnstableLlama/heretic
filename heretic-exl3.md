# Heretic EXL3 Integration Notes

This repository includes an EXL3 integration for Heretic, with EXL3 selection wired into existing quantization semantics.

## What was changed

- Added EXL3 as a quantization option:
  - `QuantizationMethod.EXL3 = "exl3"`
  - EXL3 is now selected via `--quantization exl3` (or `quantization = "exl3"` in TOML).
- Removed the separate backend selection path for EXL3 from runtime settings.
- Updated model construction logic to instantiate `Exl3Model` when `settings.quantization == QuantizationMethod.EXL3`.
- Updated EXL3-specific save/upload behavior checks to use quantization mode.
- Updated default config comments to include `exl3` under quantization options.
- Removed temporary EXL3 helper scripts that were used during bring-up.

## Current operator-facing behavior

Use EXL3 by setting:

- CLI: `--quantization exl3`
- Config: `quantization = "exl3"`

EXL3 mode continues to use adapter-side save/upload behavior where merge-into-quantized storage is not supported.

## Files touched for EXL3 integration

- `src/heretic/config.py`
- `src/heretic/main.py`
- `config.default.toml`

## Cleanup choices

- Removed handoff/debug documents in favor of this single concise EXL3 repo note.
- Removed ad-hoc EXL3 scripts from `scripts/`.
