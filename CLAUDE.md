# ExoReliability Lab — Engineering Guide

## 1. Project Mission

ExoReliability Lab is an experimental scientific-ML platform for studying the reliability of machine-learning systems used to detect exoplanet transits in stellar light curves.

The primary research question is NOT:

> "Can a neural network detect exoplanets?"

The primary research question is:

> "When should we trust a model that says it has detected an exoplanet, and under what observational conditions does that model fail?"

This repository should therefore prioritize:

1. reproducible scientific experiments,
2. proper classical baselines,
3. model calibration,
4. controlled perturbation experiments,
5. uncertainty and failure analysis,
6. interpretable visualizations,
7. scientifically defensible data splitting,
8. clean engineering and reproducibility.

Do not turn this project into a generic binary-classification demo.

---

# 2. Core Scientific Questions

The system should eventually be capable of answering questions such as:

- How does exoplanet-detection performance change as transit depth decreases?
- How does stellar or observational noise affect detection?
- How much missing cadence can models tolerate?
- Are neural-network confidence scores calibrated?
- Which models become confidently wrong?
- Does model reliability change with orbital period?
- Does reliability change with transit duration?
- Does reliability change with stellar magnitude or stellar properties?
- Which false positives most strongly fool each model?
- Does a learned model outperform classical Box Least Squares under particular noise regimes?
- Does a model trained on one observational regime generalize to another?
- Can model uncertainty identify examples where predictions should not be trusted?

Every major feature should support one or more of these research questions.

---

# 3. Initial Scope

## Phase 1 mission

Use Kepler data first.

Do NOT start with TESS, K2, multiple surveys, raw target-pixel files, or giant-scale downloads.

Phase 1 should establish a rigorous end-to-end pipeline using a manageable subset of Kepler Q1-Q17 DR25 data.

Use:

- NASA Exoplanet Archive for catalog metadata and dispositions.
- MAST / Lightkurve for light-curve retrieval.
- Astropy for astronomical utilities and Box Least Squares.
- PyTorch for learned models.
- scikit-learn for evaluation/calibration utilities where appropriate.

TESS support is a later milestone after the Kepler pipeline is stable.

---

# 4. Scientific Data Policy

## 4.1 Dataset version

Prefer a single internally consistent Kepler data release such as Q1-Q17 DR25 rather than mixing heterogeneous KOI releases.

Do not silently substitute the cumulative KOI table if the experiment is intended to represent a uniform population.

Record:

- source table,
- query,
- retrieval timestamp,
- relevant catalog fields,
- light-curve product author,
- cadence,
- quarters used,
- preprocessing configuration.

## 4.2 Labels

Never invent label mappings.

Before implementing supervised dataset generation:

1. inspect the selected NASA Exoplanet Archive table schema,
2. identify the documented disposition field,
3. document the exact mapping in `docs/data_contract.md`,
4. decide which ambiguous/candidate cases are excluded,
5. preserve the original catalog disposition alongside derived binary labels.

For the first supervised experiment, a conservative approach is preferred:

- positive: reliably labelled planetary objects,
- negative: documented false positives,
- unresolved/ambiguous candidates: excluded from supervised training unless explicitly included in a later experiment.

The exact catalog column and mapping must be verified from the selected DR25 schema.

## 4.3 Leakage prevention

This is mandatory.

Never perform a random light-curve-row split if multiple examples can originate from the same star.

Train/validation/test splitting must happen by a stable source identifier such as host/KIC/target identifier.

The same physical target must never occur across multiple splits.

Where possible, preserve class distributions using grouped stratification.

Save split manifests to disk so experiments are repeatable.

Example:

data/splits/
    train.csv
    val.csv
    test.csv

Do not regenerate splits automatically unless explicitly requested.

---

# 5. Repository Structure

Target structure:

```text
exoreliability/
├── CLAUDE.md
├── AGENTS.md
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── Makefile
├── docker-compose.yml
│
├── apps/
│   ├── api/
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   ├── routers/
│   │   │   ├── health.py
│   │   │   ├── targets.py
│   │   │   ├── predictions.py
│   │   │   └── experiments.py
│   │   └── schemas/
│   │       ├── targets.py
│   │       ├── predictions.py
│   │       └── experiments.py
│   │
│   └── web/
│       ├── package.json
│       ├── tsconfig.json
│       ├── next.config.ts
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   ├── target/
│       │   │   └── [id]/
│       │   │       └── page.tsx
│       │   └── experiments/
│       │       └── page.tsx
│       ├── components/
│       │   ├── LightCurvePlot.tsx
│       │   ├── PredictionCard.tsx
│       │   ├── ReliabilityChart.tsx
│       │   ├── PerturbationControls.tsx
│       │   └── MetricCard.tsx
│       └── lib/
│           ├── api.ts
│           └── types.ts
│
├── src/
│   └── exoreliability/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── archive.py
│       │   ├── mast.py
│       │   ├── cache.py
│       │   ├── catalog.py
│       │   ├── contracts.py
│       │   └── splits.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── clean.py
│       │   ├── normalize.py
│       │   ├── detrend.py
│       │   ├── phase_fold.py
│       │   └── resample.py
│       │
│       ├── baselines/
│       │   ├── __init__.py
│       │   └── bls.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cnn.py
│       │   ├── transformer.py
│       │   └── registry.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── trainer.py
│       │   ├── checkpoints.py
│       │   └── reproducibility.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   ├── calibration.py
│       │   ├── uncertainty.py
│       │   └── robustness.py
│       │
│       ├── perturbations/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── gaussian_noise.py
│       │   ├── missing_cadence.py
│       │   ├── transit_depth.py
│       │   ├── stellar_variability.py
│       │   └── outliers.py
│       │
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── runner.py
│       │   ├── sweep.py
│       │   └── results.py
│       │
│       └── inference/
│           ├── __init__.py
│           └── service.py
│
├── configs/
│   ├── data/
│   │   ├── kepler_dr25_small.yaml
│   │   └── kepler_dr25_full.yaml
│   ├── models/
│   │   ├── cnn.yaml
│   │   └── transformer.yaml
│   └── experiments/
│       ├── bls_smoke.yaml
│       ├── cnn_baseline.yaml
│       ├── calibration.yaml
│       └── reliability_sweep.yaml
│
├── scripts/
│   ├── fetch_catalog.py
│   ├── fetch_lightcurves.py
│   ├── build_dataset.py
│   ├── create_splits.py
│   ├── run_bls.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── run_reliability_sweep.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bls_exploration.ipynb
│   └── 03_failure_analysis.ipynb
│
├── data/
│   ├── raw/
│   │   ├── catalogs/
│   │   └── lightcurves/
│   ├── interim/
│   ├── processed/
│   └── splits/
│
├── artifacts/
│   ├── models/
│   ├── experiments/
│   ├── figures/
│   └── reports/
│
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── experiment_protocol.md
│   ├── model_cards/
│   └── decisions/
│       └── 0001-kepler-first.md
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_bls.py
│   │   ├── test_perturbations.py
│   │   ├── test_metrics.py
│   │   └── test_splits.py
│   └── integration/
│       ├── test_catalog_pipeline.py
│       └── test_api.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

Not every file needs full implementation immediately.

Create scaffolding when needed, but do not populate placeholder modules with fake logic.

---

# 6. Storage Rules

Large astronomical data and trained weights must NOT be committed to Git.

Git-ignore:

```text
data/raw/*
data/interim/*
data/processed/*
artifacts/models/*
artifacts/experiments/*
```

Retain `.gitkeep` files where useful.

Small split manifests, example fixtures, configs, and documentation may be committed.

Never store downloaded FITS files in Git.

---

# 7. Data Acquisition Layer

## `src/exoreliability/data/archive.py`

Responsibilities:

- query NASA Exoplanet Archive,
- use supported programmatic interfaces,
- retrieve catalog metadata,
- cache query results,
- expose typed records,
- preserve source metadata.

Do not scatter HTTP queries throughout notebooks or scripts.

All archive access goes through this module.

## `src/exoreliability/data/mast.py`

Responsibilities:

- search Kepler light-curve products,
- download selected products through Lightkurve/MAST,
- record author/cadence/quarter metadata,
- cache downloads,
- handle missing targets gracefully.

Network access should never occur during import.

## `cache.py`

Implement deterministic cache paths based on target/product metadata.

Never repeatedly download the same light curve.

---

# 8. Preprocessing

The preprocessing pipeline must be explicit and configurable.

Possible operations:

1. remove invalid flux values,
2. apply quality masks,
3. normalize flux,
4. optionally detrend,
5. combine observations,
6. phase-fold using catalog ephemeris,
7. resample or bin to fixed-length representation,
8. retain metadata required for later analysis.

Each operation should be testable independently.

Avoid irreversible preprocessing whenever possible.

Persist preprocessing parameters with generated examples.

Do not normalize train/validation/test using global statistics computed across all sets.

---

# 9. Classical Baseline

Implement Box Least Squares before training neural models.

File:

`src/exoreliability/baselines/bls.py`

The BLS implementation should wrap Astropy's BoxLeastSquares rather than reimplementing the algorithm.

Expose results such as:

- best period,
- best duration,
- transit time,
- estimated depth,
- BLS power,
- signal-to-noise statistic where available.

The baseline must be evaluated on the same held-out data used for learned models whenever comparison is scientifically meaningful.

Never write conclusions such as "AI beats BLS" unless the experiment actually supports that statement.

---

# 10. First Learned Model

The first neural model should be deliberately modest:

## 1D CNN

Input:

fixed-length normalized/phase-folded light curve

Suggested architecture:

```text
Input
→ Conv1D
→ BatchNorm
→ ReLU
→ MaxPool
→ Conv1D
→ BatchNorm
→ ReLU
→ MaxPool
→ Conv1D
→ Global Average Pool
→ Dropout
→ Linear
→ Logit
```

Requirements:

- configurable channel widths,
- configurable kernel sizes,
- binary logit output,
- no sigmoid inside the model when using BCEWithLogitsLoss,
- CPU fallback,
- CUDA support,
- reproducible seeds,
- checkpoint best validation model,
- early stopping,
- no giant network.

The first goal is experimental reliability, not leaderboard accuracy.

---

# 11. Transformer

Do not implement the transformer until:

- the data pipeline is stable,
- BLS baseline works,
- CNN training works,
- evaluation works,
- calibration works,
- perturbation experiments work.

A transformer is Milestone 6+, not an MVP requirement.

---

# 12. Probability Calibration

Accuracy alone is insufficient.

Record:

- accuracy,
- precision,
- recall,
- F1,
- ROC-AUC,
- PR-AUC,
- confusion matrix,
- Brier score,
- log loss,
- reliability diagram,
- expected calibration error (ECE).

Distinguish:

- discrimination,
- calibration,
- robustness.

A model can classify well while being poorly calibrated.

Where appropriate later, compare:

- raw probabilities,
- temperature scaling,
- isotonic calibration,
- Platt/sigmoid calibration.

Calibration methods must be fitted using validation/calibration data, NEVER test data.

---

# 13. Perturbation Framework

This is one of the defining features of the project.

Every perturbation should implement a common interface.

Conceptually:

```python
class Perturbation:
    name: str

    def apply(
        self,
        light_curve,
        severity: float,
        rng,
        metadata=None,
    ): ...
```

Every perturbation must:

- accept deterministic RNG/state,
- have documented severity semantics,
- preserve input shape unless explicitly documented,
- record parameters applied,
- be independently unit tested.

Initial perturbations:

## Gaussian observational noise

Increase flux noise in controlled steps.

Example severities:

```text
0.0
0.25
0.5
1.0
2.0
```

Severity semantics must be explicitly defined.

## Missing cadence

Randomly mask/remove a controlled fraction of observations.

Example:

```text
0%
5%
10%
25%
40%
```

Later add structured contiguous gaps.

## Transit-depth attenuation

Using known transit ephemeris, reduce the apparent depth of the transit while preserving the out-of-transit baseline.

Example multipliers:

```text
1.0
0.8
0.6
0.4
0.2
```

Do not simply multiply the entire light curve.

## Stellar variability

Later milestone.

Inject controlled low-frequency/quasi-periodic variability.

## Outliers/artifacts

Later milestone.

Inject sparse extreme observations or systematic artifacts.

---

# 14. Reliability Experiment

A reliability experiment is not simply:

```text
perturb → predict
```

It should record:

```text
target_id
original_label
model_name
model_version
perturbation
severity
random_seed
predicted_probability
predicted_label
correct
metadata
```

Store experiment results in a machine-readable format.

Parquet is preferred for larger result tables.

Example output:

```text
artifacts/experiments/
└── 2026-xx-xx_cnn_noise_sweep/
    ├── config.yaml
    ├── results.parquet
    ├── metrics.json
    ├── environment.json
    └── figures/
        ├── accuracy_vs_noise.png
        ├── calibration_vs_noise.png
        └── confidence_vs_noise.png
```

Every experiment folder should contain enough information to understand how it was produced.

---

# 15. Reproducibility

Every experiment must record at minimum:

- random seed,
- Git commit hash if available,
- Python version,
- PyTorch version,
- CUDA availability,
- GPU name if available,
- config file,
- dataset/split identifier,
- model checkpoint identifier.

Set seeds for:

- Python,
- NumPy,
- PyTorch.

Avoid claiming perfect determinism across GPU environments.

---

# 16. Configuration

Do not hard-code important experiment parameters.

Use YAML config files for:

- dataset selection,
- preprocessing,
- model parameters,
- training parameters,
- perturbation parameters,
- evaluation parameters.

Example:

```yaml
seed: 42

dataset:
  mission: kepler
  release: q1_q17_dr25
  max_targets: 200
  cadence: long

preprocessing:
  normalize: true
  detrend: true
  phase_fold: true
  bins: 2048

model:
  name: cnn

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 25
  early_stopping_patience: 5
```

Configs must be validated before running expensive jobs.

---

# 17. API

Use FastAPI.

Initial endpoints:

```text
GET /health

GET /targets/{target_id}

GET /targets/{target_id}/lightcurve

POST /predict

POST /experiments/perturb

GET /experiments/{experiment_id}
```

Do not put model-training jobs directly inside HTTP request handlers.

API handlers should call application/service functions.

Pydantic models should define request/response contracts.

---

# 18. Frontend

Use Next.js + TypeScript.

The web interface should ultimately emphasize scientific exploration, not generic SaaS styling.

Primary future screen:

```text
┌───────────────────────────────────────────────────────────┐
│ ExoReliability Lab                                       │
├───────────────────────────────────────────────────────────┤
│ Target: KIC XXXXXXXX                                     │
│ Known disposition: ...                                   │
│ Period: ...                 Transit depth: ...            │
├───────────────────────────────────────────────────────────┤
│                                                           │
│                   LIGHT CURVE                             │
│                                                           │
├─────────────────────┬─────────────────────────────────────┤
│ Perturbation Lab    │ Model Predictions                   │
│                     │                                     │
│ Noise        [---]  │ CNN            0.87                 │
│ Missing data [---]  │ BLS statistic  ...                  │
│ Depth         [---] │ Calibrated CNN 0.79                 │
├─────────────────────┴─────────────────────────────────────┤
│ Reliability / failure visualization                      │
└───────────────────────────────────────────────────────────┘
```

Do not spend significant effort polishing UI until the underlying experiment pipeline is functional.

Use a proper chart library suitable for interactive scientific plots.

Keep visualization logic separate from data/API logic.

---

# 19. Notebooks

Notebooks are allowed for exploration only.

They must NOT become the production pipeline.

Bad:

```text
notebook downloads data
notebook cleans data
notebook trains model
notebook evaluates model
```

Good:

```text
library performs pipeline
notebook imports library
notebook visualizes/analyzes results
```

Any reusable functionality discovered in a notebook must move into `src/exoreliability`.

---

# 20. Testing Requirements

Important scientific transforms require tests.

At minimum test:

## Preprocessing

- NaNs handled correctly.
- normalization has expected baseline.
- phase folding maps values correctly.
- resampling returns expected shape.

## Data splitting

- target identifiers never overlap across splits.
- split generation is deterministic.

## Perturbations

- severity zero behaves as identity where appropriate.
- fixed seed produces identical result.
- missing-cadence fraction is approximately correct.
- transit attenuation affects transit region rather than entire curve.

## Metrics

Use hand-computable tiny examples.

## API

- health endpoint,
- valid prediction request,
- invalid request handling.

No external archive calls should occur in ordinary unit tests.

Use fixtures/mocks.

---

# 21. Code Quality

Python:

- Python 3.11+
- type hints for public APIs,
- Ruff,
- pytest,
- mypy or pyright where practical,
- pathlib instead of manual path concatenation,
- logging instead of print in library code,
- docstrings on scientifically important functions.

TypeScript:

- strict TypeScript,
- ESLint,
- avoid `any`,
- shared API types where reasonable.

Prefer small explicit modules over premature abstraction.

---

# 22. Dependency Philosophy

Do not add libraries without a concrete need.

Expected Python dependencies may include:

```text
numpy
pandas
scipy
astropy
astroquery
lightkurve
scikit-learn
torch
pyarrow
pydantic
fastapi
uvicorn
httpx
pyyaml
matplotlib
```

Development dependencies may include:

```text
pytest
pytest-cov
ruff
mypy
```

Do not introduce:

- TensorFlow alongside PyTorch,
- Spark,
- Kubernetes,
- Kafka,
- MLflow,
- Airflow,
- distributed training,
- a vector database,

unless a later requirement clearly justifies it.

This project should remain runnable on one developer laptop.

---

# 23. Hardware Constraints

Primary development environment:

- laptop-class NVIDIA GPU,
- approximately 8 GB VRAM,
- Linux,
- local training.

Design accordingly.

Requirements:

- models must support mini-batching,
- avoid loading the entire dataset into GPU memory,
- use CPU-side caching,
- optionally support mixed precision,
- default experiments should be modest,
- smoke-test configurations should complete with very small datasets.

Never make successful tests depend on CUDA.

---

# 24. Experiment Scale

Always support at least three scales.

## Smoke

Very small dataset.

Purpose:

- test code,
- verify pipeline,
- CI-friendly where possible.

## Small

Hundreds to a few thousand targets.

Purpose:

- local iteration,
- debugging,
- initial comparisons.

## Full

Larger scientifically meaningful dataset.

Purpose:

- final experiments.

Never require the full dataset to verify that code works.

---

# 25. Documentation

Maintain:

## `README.md`

For humans visiting GitHub:

- what problem is being studied,
- screenshot/demo eventually,
- quick start,
- architecture summary,
- example experiment,
- scientific limitations.

## `docs/data_contract.md`

Document:

- source tables,
- catalog fields,
- label mapping,
- target identifiers,
- preprocessing outputs,
- split methodology.

## `docs/experiment_protocol.md`

Document:

- evaluation procedure,
- perturbations,
- metrics,
- calibration procedure,
- statistical comparisons.

## `docs/architecture.md`

Explain system components and data flow.

## ADRs

Use `docs/decisions/` for important technical/scientific decisions.

---

# 26. Research Integrity

Do not claim that a model discovered an exoplanet merely because it predicts a high probability.

Use language such as:

- "model prediction,"
- "candidate-like signal,"
- "transit-like pattern,"
- "model confidence."

Avoid:

- "new planet discovered,"
- "confirmed planet,"
- "NASA-quality detection,"

unless supported by an appropriate scientific validation process.

Likewise, distinguish:

```text
model confidence ≠ physical probability that a planet exists
```

unless calibration and problem formulation justify that interpretation.

---

# 27. Milestones

## Milestone 0 — Repository foundation

Create:

- Python package,
- project configuration,
- lint/test tooling,
- API shell,
- frontend shell,
- docs,
- data directories,
- configs.

## Milestone 1 — Data + classical baseline

Implement:

- DR25 catalog query,
- small target manifest,
- MAST/Lightkurve light-curve downloader,
- cache,
- basic cleaning,
- normalization,
- example phase folding,
- BLS wrapper,
- tests,
- minimal API exposure,
- minimal frontend light-curve visualization.

No neural-network training required to complete this milestone.

## Milestone 2 — Reproducible ML dataset

Implement:

- fixed-length representations,
- grouped train/val/test splits,
- dataset manifests,
- PyTorch Dataset/DataLoader,
- class-distribution report.

## Milestone 3 — CNN baseline

Implement:

- 1D CNN,
- training loop,
- checkpointing,
- evaluation,
- classification metrics.

## Milestone 4 — Calibration

Implement:

- reliability diagrams,
- Brier score,
- ECE,
- calibration methods,
- calibrated-vs-uncalibrated comparison.

## Milestone 5 — Reliability Lab

Implement:

- perturbation framework,
- noise sweep,
- missing-cadence sweep,
- transit-depth sweep,
- robustness plots,
- experiment result storage.

This is the first major research-quality release.

## Milestone 6 — Additional models

Possible:

- transformer,
- temporal CNN,
- ensembles,
- uncertainty methods.

Only add if experimentally justified.

## Milestone 7 — Cross-domain testing

Possible:

- TESS evaluation,
- survey/domain shift,
- new stellar regimes.

## Milestone 8 — Product-quality demo

Polish:

- interactive perturbation sliders,
- target browser,
- side-by-side models,
- reliability diagrams,
- downloadable experiment reports.

---

# 28. Agent Behaviour

When working autonomously:

1. Read this file first.
2. Inspect existing code before creating new abstractions.
3. Do not rewrite working components unnecessarily.
4. Implement one milestone at a time.
5. Run tests after meaningful changes.
6. Fix failures caused by your changes.
7. Do not silently change the scientific methodology.
8. Do not download large datasets without a small-mode option.
9. Do not train expensive models unless explicitly requested.
10. Prefer reproducible scripts to manual commands.
11. Update documentation when architecture or methodology changes.
12. Never fabricate experiment results.
13. Never place fake metrics in README screenshots or reports.
14. Clearly mark mocked/sample/demo data.
15. Do not commit generated astronomical datasets or model weights.
16. Do not create dozens of unnecessary abstractions.
17. Keep the project runnable locally.

---

# 29. Before Implementing a Scientific Feature

Ask internally:

1. What scientific question does this answer?
2. What data supports it?
3. Could leakage contaminate this result?
4. What baseline should it be compared against?
5. What metric measures the thing we actually care about?
6. Can another person reproduce it?
7. What assumptions are being made?

If these questions cannot be answered, do not rush into implementation.

---

# 30. Definition of Done

A feature is not done because the code exists.

A feature is done when:

- implementation works,
- tests exist where appropriate,
- configuration is explicit,
- errors are handled,
- documentation is updated,
- reproducibility is considered,
- generated outputs can be traced back to inputs/config,
- no obvious leakage has been introduced,
- it works on the intended local environment.

Scientific correctness takes priority over adding more features.