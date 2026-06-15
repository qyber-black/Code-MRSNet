<!--
SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein <frank@langbein.org>, Cardiff University
-->

# Top-k simulation-vs-benchmark rank comparison (Part 2)

`run_topk.json` retrains the **top-5 CNN** and **top-5 CAEQ (YAE)** model-selection
candidates (ranked by simulated validation concentration MAE, `edit_off`/`edit_on`,
`imaginary-real`) at the **deployment protocol** (100k spectra, 2000 epochs, no
validation split, batch 16, sum-norm) and benchmarks each on the phantom series.

This produces, for each architecture family, a within-family comparison of the
*simulation* ranking against the *phantom* ranking — the question being whether the
near-tied top candidates (CNN val MAE 0.0073–0.0080; CAEQ 0.0105–0.0107) reorder
under domain shift. It complements Part 1 (the eight deployed architectures; see the
paper's "Simulation--Benchmark Ranking Stability" section and
`paper-mrsnet-autoencoder/scripts/sim2real/rank_comparison.py`).

## Run locally (not Slurm)

```bash
# from the code-mrsnet repo root, with the project venv
venv/bin/python run.py etc/run_topk.json --dry-run     # inspect the commands first
venv/bin/python run.py etc/run_topk.json               # train + benchmark (long: ~hours/model)
```

`run.py` checks for existing results and skips completed train/benchmark steps, so the
job is resumable. Trained models land in `data/model/<model>/...`; benchmark writes
`*_max_concentration_errors.json` under each model folder. The dataset path resolves to
`data/sim-spectra-megapress/fid-a-2d_2000_4096/.../100000-1` (present locally).

Note: `cnn_top1` is the already-deployed detailed CNN
(`cnn_-1_-3_9_9_5_3_0.0_0.3_512_512_2048_sigmoid`); its phantom number already exists in
`data/model-dist`. It is kept here for a uniform, self-contained top-5 set, but can be
dropped from the config to save one training if its `model-dist` benchmark is reused.

## Harvest

```bash
# aggregate the new top-k benchmark results
venv/bin/python data/model-dist/aggregate/analyze_model_dist.py \
    --models-dir data/model \
    --agg-csv /tmp/topk_all_results.csv \
    --out-dir /tmp/topk_agg
```

Then the paper-side within-family rank analysis (Spearman/Kendall of simulation vs
phantom rank, slopegraph) is added to
`paper-mrsnet-autoencoder/scripts/sim2real/` and folded into the ranking-stability
section. (Written against the real output once this job has run.)

## Training-seed replication (`run_topk_seeds.json`)

To check whether the *between-candidate* phantom spread is real or just training
stochasticity, `run_topk_seeds.json` retrains a small spread-defining subset with
independent random seeds: **CNN-2** (phantom-best), **CNN-4** (phantom-worst),
**CNN-5** (sim-best); **CAEQ-1** (sim-best / phantom-worst), **CAEQ-2** (phantom-best).
Each `train` draws a fresh OS-entropy seed (`mrsnet/train.py`) and lands in a new
`NoValidation-N` instance, so `--force` (which bypasses the skip-if-exists check)
accumulates independent seeds rather than re-using the existing one.

```bash
# from the code-mrsnet repo root, with the project venv -- run LOCALLY (not Slurm)
venv/bin/python run.py etc/run_topk_seeds.json --dry-run      # inspect first
for i in 1 2 3 4; do venv/bin/python run.py etc/run_topk_seeds.json --force; done
# -> 4 extra seeds per candidate (= 5 incl. the existing run_topk instance)
venv/bin/python aggregate.py data/model                       # refresh aggregate
```

The paper-side analysis auto-detects every `NoValidation-*` instance and reports the
across-seed mean/std of the phantom MAE (no code change needed):

```bash
# from the paper repo
/path/to/code-mrsnet/venv/bin/python scripts/sim2real/rank_comparison_topk.py
```

With the seeds present, the within-family table gains a `±std` on each Bench. MAE and
the prose can state how the seed-to-seed spread compares with the between-candidate
spread. Until then the analysis runs on the single existing instance per candidate.
