# Self-calibrated KL divergence

A scorer plugin that measures how far a modified model has drifted from the
original on the original model's *own* responses to realistic prompts, at
every response position. It lives in
`src/heretic/scorers/self_calibrated_kl.py` and is configured under
`[scorer.SelfCalibratedKL]` in `config.default.toml`.

## Why

Heretic's stock `KLDivergence` scorer compares the two models at exactly one
position: the first response token, on 100 Alpaca-style prompts. Two things
have changed since that was designed:

- Heavily post-trained models (Gemma 4, Qwen 3.5 and later) have nearly
  deterministic first tokens: a thinking tag, "Sure", the start of a tool call.
  KL divergence between two near-one-hot distributions is hypersensitive to tiny
  logit shifts and says nothing about the hundreds of tokens that follow.
- Those models are tuned for agentic, long-context, system-prompted chat.
  Short Alpaca instructions sit outside that regime, so both the measurement and
  the harmless residual means that drive the refusal direction are taken in an
  unrepresentative part of activation space.

The ExLlamaV3 quantizer hit the same wall on the same model families and fixed
it with *self-calibration*: generate the calibration corpus from the unmodified
model itself using in-domain prompts, then measure the quantized model
teacher-forced against those generations. This scorer ports that idea.

## What it computes

1. Before optimization, the original model greedily answers the configured
   prompts. The response token ids are cached in memory and, by default, on
   disk under `<study_checkpoint_dir>/self_calibrated_kl/`, keyed on the model,
   the exact prompts and the generation settings, so later runs on the same
   original model (further studies, `--evaluate-model` over many checkpoints)
   skip generation. Greedy decoding makes the corpus deterministic, so reusing
   it does not affect reproducibility.
2. Its distribution at every response position is summarised as the `top_k`
   most likely tokens plus the total mass of everything else (one "tail"
   bucket).
3. Each trial runs one teacher-forced forward pass per sequence on the modified
   model and computes KL(original ‖ modified) at every response position
   between the two distributions coarsened onto that partition. This is a
   proper KL divergence and, by the data processing inequality, a lower bound on
   the exact one; it is tight when the tail mass is small, which the scorer
   reports at start-up. Mass the modified model moves onto tokens outside the
   original's top-k (garbage tokens, for instance) lands in the tail bucket and
   is counted. `top_k = 0` keeps full distributions and gives the exact value at
   the cost of prompts × tokens × vocabulary floats of memory.
4. The score is the mean over prompts of the mean per-token KL. Alongside it the
   console shows the per-position 95th percentile and maximum, and, when the
   responses contain a reasoning span (`answer_start_markers`), the mean over
   thinking positions and over answer positions separately. A low mean can hide a
   sharp lesion at a few decisive positions; these companions are there to
   expose it.

Teacher forcing holds the original trajectory fixed, so the divergences at
different positions are directly comparable and do not compound. The flip side
is that it does not measure what the modified model does once it leaves that
trajectory. A capability benchmark (the `BenchmarkScore` scorer) is the
complementary check for free-running behaviour.

## Cost

Per trial: one prefill per reference sequence, no generation. Comparable to the
`KeywordRate` scorer, and much richer than one token per prompt. Only the last
`max response length + 1` positions are projected onto the vocabulary, on both
the Hugging Face and the EXL3 backend, and the batch size is derived from the
vocabulary size and response length so the transient logits stay around two
gigabytes.

## Running the measurement experiment

`config.selfcal-experiment.toml` evaluates an existing modified model against
its original with four KL variants at once, varying prompt domain and token
coverage independently:

|                   | first token only       | every response token        |
|-------------------|------------------------|-----------------------------|
| Alpaca prompts    | `KLDivergence`         | `SelfCalibratedKL - alpaca` |
| in-domain prompts | `KLDivergence - domain`| `SelfCalibratedKL - domain` |

Heretic reads its settings from `config.toml` in the working directory (there
is no `--config` flag), so run it from a directory where the experiment config
is named `config.toml`:

```
mkdir selfcal && cp config.selfcal-experiment.toml selfcal/config.toml && cd selfcal
heretic --model <original> --evaluate-model <modified>
```

Run it over several checkpoints spanning the damage range (a few Pareto-front
trials plus some deliberately over-modified ones) and add a `BenchmarkScore`
entry to the same config. The question is which of the four variants best
predicts benchmark loss, and whether in-domain prompts matter only once the
whole response is scored, which is what the peaked-first-token explanation
predicts.

## Backend notes

Both backends gained two methods used through the plugin context:
`get_response_token_ids` (greedy generation returning token ids, cut after the
first end-of-sequence token) and `get_response_logits` (teacher-forced logits at
the positions predicting each response token). On the EXL3 backend the
generator returns text, so completions are re-tokenized with the Hugging Face
tokenizer, and the EOS id is appended when generation stopped on it. The
reference sequence need not be token-identical to what was sampled; it only has
to be a fixed, in-distribution sequence scored identically by both models.
