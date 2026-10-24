# SAE Emotion

Research code accompanying **Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders**, by Bangzhao Shu, Arinjay Singh, and Mai ElSherief.

This working copy organizes the supplied research archive around the paper. It contains the original experiment implementations, cleaned notebooks, local data, and a traceable migration record. **It is an organized release draft, not yet a verified reproduction of the final paper.** Missing experiment artifacts and code/paper differences are listed in [the release review](docs/RELEASE_REVIEW.md).

## Where to start

| Location | Contents |
| --- | --- |
| `src/sae_emotion/` | Shared model, SAE, feature-selection, intervention, and plotting code |
| `experiments/rq1_features/` | Hidden-state extraction, candidate selection, single-feature causal effects |
| `experiments/rq2_causal/` | Necessity, sufficiency, and adjacent-layer mediation |
| `experiments/rq3_steering/` | Targeted feature amplification and global feature steering |
| `experiments/evaluation/` | CPU lexical preprocessing and general-capability evaluation |
| `notebooks/` | Feature analysis, baselines, robustness, statistics, and paper figures |
| `data/` | Supplied datasets, normalized filenames, and data inventory; CSVs ignored by Git |
| `environment/`, `requirements/` | Uploaded cluster environment and exact direct-dependency version records |
| `results/` | Supplied overlap summaries and executed lexical-preprocessing, environment, and split checks |
| `archive/` | Distinct exploratory implementations and unique notebook checkpoints |
| `docs/` | Paper-to-code map, reproduction guide, and unresolved release questions |
| `migration_manifest.csv` | Original filename, new location, source hash, and action for every source file |

The original ZIP and additional uploads are the authoritative source backups. Large `.npy` activation files are intentionally excluded: users generate them by running the pipeline. No model weights or final steering checkpoints were supplied; these can also be generated from source. EXPRESS filtering code is outside this release's current scope. Notebook cell order and research parameters are retained except for the author-requested external-dataset restriction, explicit saved-split routing, and documented correction to nested robustness prompts; output images, execution state, embedded credentials, and machine-specific cache paths were removed. Different variants have not been silently merged.

The local package includes the shared steering split: **2,400 training rows (400 per emotion)** and **1,974 held-out rows**. Global steering loads these files directly, and both emotion-vector training notebooks use the same paths. They exactly reproduce the original seeded split and row order. Active external evaluation covers **EmoEvent and XED only**. See [data/README.md](data/README.md) for the confirmed overlap and checkpoint-selection caveats.

## Inspect on your Mac

From this repository folder:

```bash
python3 scripts/check_repository.py
python3 scripts/run.py --list
```

These commands use only Python's standard library and do not download models or start experiments. They validate syntax, notebook cleanup, credential patterns, package imports, and the supplied data's basic schema/counts. Passing these checks does not validate scientific results. The separate [lexical-control workflow](docs/LEXICAL_CONTROL.md) has also been executed locally through dataset construction, reproducing 1,332 balanced examples; model scores have not been rerun.

For notebook browsing, use a Python 3.10+ environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks]'
jupyter lab
```

Open notebooks inside this repository and run their first setup cell. That cell locates the repository, makes the shared modules importable, and resolves the original `data/...` paths from the repository root.

## Run research experiments

The original model experiments target CUDA GPUs. A Mac can inspect/edit the release and run lexical preprocessing; the full model experiments have not been validated on Apple Silicon or CPU. The [recorded cluster environment](docs/ENVIRONMENT.md) uses **Python 3.11.14, PyTorch 2.13.0+cu126, Transformers 4.57.6, and SAELens 6.37.3**. Exact direct-dependency versions are in `requirements/research-cluster.txt`, with the complete uploaded inventory in `environment/package_versions.txt`. These are recorded versions, not an independently validated installation lock. The optional `research` extras in `pyproject.toml` remain unpinned; use the cluster records for reproduction.

In the existing cluster environment, register this package without changing its dependencies:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Edit the configuration block at the top of the relevant experiment script, then use the launcher. For example:

```bash
python scripts/run.py extract-activations
```

The launcher prints the selected script and runs it from the repository root. It does **not** harmonize model settings across stages. Follow [REPRODUCING.md](docs/REPRODUCING.md) for inputs, outputs, and the required order. Refer to [PAPER_MAP.md](docs/PAPER_MAP.md) for the paper section associated with each entry point.

Use your own Hugging Face account with the required model access. Credentials are read from the environment or your existing local login. Never put access tokens in a notebook or commit them.

## Before publishing

Resolve the methodological items in [RELEASE_REVIEW.md](docs/RELEASE_REVIEW.md), especially checkpoint selection, evaluation overlap, corrected robustness-score reruns, experiment settings, and final statistical-testing scope. The shared split files and lexical-control implementation are now included. Large activation files and pretrained run outputs are not required uploads for this source-based release. Steering variants are retained for readers to rerun; none is labeled the best without a comparable validation result. Choose a code license and confirm dataset redistribution terms. No license has been invented for the authors.

The supplied CSVs are available locally but ignored by Git until their release provenance is settled. Check `git status` before committing. No commits or pushes were made during this organization pass.
