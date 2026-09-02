# Sensor-Failure Processes and Missing-Modality Evaluation in Wearable HAR

Code, extracted failure traces, simulator, and result tables for the paper:

> Random masking overestimates missing-modality robustness in wearable
> human activity recognition: evaluation under empirically calibrated
> sensor-failure processes.
> Submitted to *Pervasive and Mobile Computing*, 2026.

The study characterizes how data loss actually occurs in the raw
recordings of public HAR benchmarks (PAMAP2, OPPORTUNITY), builds a
trace-calibrated failure-process simulator from those measurements, and
re-evaluates missing-modality robustness and common defenses under
matched loss budgets across three datasets with leave-one-subject-out
cross-validation.

## Repository layout

```
notebook/          end-to-end pipeline (Kaggle notebook, self-healing cells)
model/failure_sim.py   the failure-process simulator (regimes: burst,
                       persistent, gradual, iid)
output/            all result CSVs and split/device metadata
fig/               figure sources: per-figure data CSVs, generation
                   scripts output (PNG+PDF, 300 dpi), draw.io sources
```

## Environment

Experiments ran on Kaggle (2x T4 GPU) with the default image: Python
3.12.13, PyTorch 2.10.0 (CUDA 12.8), NumPy 2.0.2, pandas 2.3.3, SciPy
1.16.3, scikit-learn 1.6.1, matplotlib 3.10.0; exact pins are in
`requirements.txt`. No package outside the stock image is required. Total compute for a cold
reproduction is roughly 8 to 10 GPU hours; all training and evaluation
cells checkpoint their work and skip completed items, so partial reruns
are cheap.

## Data

The three datasets are public and are downloaded automatically by the
notebook from the UCI repository:

- PAMAP2 (raw, uncleaned protocol files; wireless IMUs at 100 Hz)
- OPPORTUNITY (raw session files; wired IMUs at 30 Hz plus wireless
  accelerometers used for failure characterization)
- MHEALTH (Shimmer devices at 50 Hz)

Raw files are parsed once into per-subject window tensors under
`dataset/proc/`; the raw NaN patterns are consumed before any
interpolation to extract the failure traces.

## Reproducing the results

1. Open the notebook on Kaggle (or any machine with a GPU) and run all
   cells in order. The bootstrap cell creates the folder tree and
   downloads the datasets; the restore cell reuses artifacts from an
   attached previous version when present.
2. Constants live in a single configuration cell: window length 2 s,
   50% overlap, label purity 0.5, training seeds (42, 43, 44), mask
   seeds (0, 1, 2), modality-dropout rate 0.2, epochs 30, batch 128,
   Adam 1e-3, and the calibration map
   `{pamap2: pamap2, opportunity: opportunity, mhealth: opportunity}`.
   Every number reported in the paper derives from these constants and
   the seeds; nothing is tuned per condition.
3. Tables in the paper map to CSVs as listed below; the figure data CSVs
   under `fig/` regenerate every figure without rerunning experiments.

## Using the simulator alone

```python
import failure_sim as fsim
fsim.set_pools(fsim.load_pools("output"))     # empirical duration pools

sim = fsim.FailureSim("burst", calibration="opportunity",
                      seed=0, rate_scale=10)
mask = sim.mask(n_samples=54000, n_devices=3)  # True = available
```

Regimes: `burst` (on-off renewal resampled from empirical durations;
`rate_scale` divides gap durations), `persistent` (one uniformly chosen
device dies at a uniform onset), `gradual` (60 s linear ramp to complete
loss), `iid` (Bernoulli comparison condition, `iid_p`). Parametric fits
(discretized Pareto, lognormal, spliced lognormal-body Pareto-tail) are
included as a portable fallback for users without traces.

## Output files

Evaluation results (one row per dataset x fold x condition x mask seed):

| file | content |
|---|---|
| `eval_results.csv` | clean-trained baseline under structured regimes and budget-matched iid twins (`iid@<regime>` rows), seed 42 |
| `eval_results_seedmain.csv` | same protocol for training seeds 42-44 with three comparators: `structured`, `iid_sample`, `iid_window` |
| `eval_results_defenses.csv` | mdrop and faware2 under structured regimes |
| `eval_results_variants.csv` | single-component variants fawareP and fawareB |
| `eval_results_imputer.csv` | reconstruction front-end (ConvImputer + frozen baseline) |

Aggregated tables (paper tables derive from these):

| file | paper table |
|---|---|
| `main_table_median.csv`, `main_table_3seed.csv` | structured vs iid gaps (Table: main comparison) |
| `matrix_table.csv`, `defense_table_median.csv` | defense families |
| `wilcoxon_tests.csv`, `wilcoxon_holm.csv` | paired tests with Holm correction |
| `ablation_devices.csv`, `ablation_window.csv` | device-subset and window-length ablations |
| `calib_loso.csv` | leave-one-unit-out calibration fidelity |
| `baseline_results.csv`, `*_clean_results.csv` | clean-test scores |

Failure characterization:

| file | content |
|---|---|
| `pamap2_run_lengths.csv`, `opportunity_run_lengths.csv` | every interior missing run (device, session, duration, boundary flag) |
| `pamap2_gaps.csv`, `opportunity_gaps.csv` | availability gaps between runs |
| `splits.json`, `device_maps.json` | LOSO folds and device-to-channel maps |

## Figures

Each figure ships with its plotted data (`fig/fig*_*.csv`) and both PNG
and PDF at 300 dpi. `fig6_architecture.drawio` is the editable source of
the system diagram.

## Citation

If you use the traces, the simulator, or the evaluation protocol, please
cite the paper above. A BibTeX entry will be added upon publication.
