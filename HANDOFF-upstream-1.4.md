# Handoff: bring heretic fork up to date with upstream v1.4

Repo: `/home/unstable/exl3/private/heretic`, branch `master`, clean tree.
**Do not re-run broad recon — everything needed is below.**

## Goal
Merge `upstream/master` (23 commits, v1.3→v1.4) into our fork, preserving the EXL3
backend, our ARA/ARA-LoRA work, and our configs.

## Established facts (verified — do not re-investigate)

- **ARA-LoRA PR #332 merged 2026-07-05 as `25979ad` into the `upstream/ara` BRANCH, not
  master.** `upstream/ara` is 67 commits behind master — dead end. **Pull nothing from it.**
- Our ARA-LoRA derives from that PR (attribution at `src/heretic/config.py:352`) and is a
  **strict superset**: + EXL3 backend, `ara_lora_regularization`, steer clamp, non-finite
  guards, adapter-rank fix, and fixes for 2 bugs still in the official version.
- Merge base is `d68a41f`. Our fork diff = `git diff d68a41f master` → 10 files:
  `config.default.toml`, `heretic-exl3.md`, `pyproject.toml`, `config.py`, `evaluator.py`,
  `exl3_model.py` (new, 1615 lines), `main.py`, `markers/conscious.jsonl` (new),
  `model.py`, `utils.py`.

## What upstream changed (the reason this is hard)

- `evaluator.py` **rewritten** into a plugin/scorer system (`plugin.py`, `scorer.py`,
  `scorers/keyword_rate.py`, `scorers/kl_divergence.py`). `Evaluator.get_score()` is GONE →
  `get_scores() -> list[tuple[str, Score]]`. **Both** our evaluator hunks live in the deleted
  function.
- `main.py`: 930 lines changed. Renames: `refusal_directions`→`residual_directions`,
  `get_logprobs[_batched]`→`get_logits[_batched]` (softmax moved into the scorer),
  `trial.user_attrs["kl_divergence"/"refusals"]`→`["scores"]`,
  `obtain_merge_strategy`/`"adapter"`→`obtain_export_strategy`/`ExportStrategy` enum.
- All `prompt_select/prompt_text/prompt_path/prompt_password` and `utils.set_seed` **deleted**
  → headless `utils.ask_if_unset(settings.X, questionary.Y(...))`. Our `main.py` utils import
  block references 5 deleted names → **immediate ImportError after merge**.
- **Deleted settings**: `trust_remote_code`, `refusal_markers`, `good_evaluation_prompts`,
  `bad_evaluation_prompts`, `kl_divergence_scale`, `kl_divergence_target`.
  `print_responses` → moved onto the KeywordRate scorer.
- `model.py`: `abliterate` reworked (`W_org` #398; `max_weight == 0` early-continue #387),
  new `torch.manual_seed` before `svd_lowrank`.

## Decisions already made

1. **Keep our ARA-LoRA.** Adopt nothing from `upstream/ara`.
2. **`invert_target`: use upstream's native `optimization = "maximize"`** in the new
   `ScorerConfig`. Drop our evaluator scoring hunk and the Pareto sort-sign flip in `main.py`.
   Migrate our configs accordingly.
3. **venv → `/home/unstable/exl3/tabbyAPI/venv/`** (existing, see risk below).

## Plan

1. Branch `staging/upstream-1.4`. Commit the 13 untracked configs + `help.txt` +
   `src/heretic/markers/{funny,sad}.jsonl`. Add `runs/` (16 GB!) and `artifacts/` to
   `.gitignore`. **Never commit `runs/`.**
2. `git merge upstream/master`. Mechanical resolution: `config.py`, `model.py`, `utils.py`,
   `pyproject.toml`, `config.default.toml`.
3. Re-author (not merge): NaN/non-finite guard → into `scorers/kl_divergence.py`;
   `invert_target` → native `maximize`.
4. **Audit `exl3_model.py`** — merges with zero conflicts then silently breaks. Known breaks:
   calls `settings.trust_remote_code` (`:1395`, deleted), exposes `get_logprobs_batched`
   (`:1230`, renamed upstream), **no `processor` attribute** (`main.py:995`/`:1142` will
   `AttributeError`). Must also satisfy the new `plugin.Context` protocol.
5. Migrate the 13 configs to the new `[scorer.*]` schema.
6. Verify: `uv sync`, `heretic --help`, import `Exl3Model`, load every config. No model runs.

## Biggest risk — check this early

`/home/unstable/exl3/tabbyAPI/venv` has **exllamav3 1.1.0**, torch 2.8.0, transformers 5.10.2.
Our `pyproject.toml` pins `exllamav3==0.0.33`, and its own comment warns the internals we
target (`Module` iteration, `TransformerBlock.export_state`, `LinearEXL3.get_weight_tensor`,
`Linear.lora_a/b_tensors`) change across point releases. **0.0.33 → 1.1.0 is a major jump; the
EXL3 backend likely needs an API re-audit independent of the upstream merge.** Decide with the
user whether to (a) install 0.0.33 into that venv, or (b) port `exl3_model.py` to exllamav3 1.x.

## Do not delete without asking
`src/heretic/markers/{conscious,funny,sad}.jsonl` are orphaned — the `--markers` flag was
reverted (`6c46c82`). They may still be wanted. Surface, don't remove.
