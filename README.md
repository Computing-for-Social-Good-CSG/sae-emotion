# SAE Emotion

Research code accompanying **Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders**, by Bangzhao Shu, Arinjay Singh, and Mai ElSherief.

## Setup

The research environment uses **Python 3.11**, **CUDA 12.6**, and an **NVIDIA H200**. The six core library versions are listed in [requirements.txt](requirements.txt): PyTorch, Transformers, SAELens, NNsight, Accelerate, and Datasets. Analysis and notebook dependencies are listed in `pyproject.toml`.

```bash
python -m pip install -r requirements.txt -e '.[research,notebooks]'
```

If you already have the research environment, register this package without changing its installed dependencies:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Use your Hugging Face login or `HF_TOKEN` for model access. Set `HF_HOME` if you want a specific cache location.

## Repository

| Location | Contents |
| --- | --- |
| `src/sae_emotion/` | Shared SAE, feature-selection, intervention, and plotting utilities |
| `experiments/rq1_features/` | Activation extraction, candidate selection, and causal effects |
| `experiments/rq2_causal/` | Feature ablation, activation patching, and adjacent-layer mediation |
| `experiments/rq3_steering/` | Targeted amplification and learned global steering |
| `experiments/evaluation/` | Lexical-control preprocessing and general-capability evaluation |
| `notebooks/` | Feature analysis, baselines, steering analysis, evaluation, and figures |
| `data/` | The five source CSV files and their column descriptions |

## Data

The actual CSVs are included in `data/` and enabled for Git:

| File | Rows | Purpose |
| --- | ---: | --- |
| `combined_emotions.csv` | 9,870 | Combined emotion dataset, without model predictions |
| `train_df.csv` | 2,400 | Shared steering training split; 400 examples per emotion |
| `val_df.csv` | 1,974 | Shared steering held-out split |
| `emoevent_en.csv` | 7,303 | External EmoEvent dataset; tab-separated, with `others` excluded during evaluation |
| `xed-en.csv` | 2,803 | External XED dataset |

See [data/README.md](data/README.md) for columns and split details. Model predictions are generated during extraction and saved separately as `data/combined_zeroshot_preds_<model>.csv`; downstream feature analysis uses those generated predictions. Large activation arrays, checkpoints, and generated outputs stay ignored by Git.

## Run the pipeline

Edit the model, SAE, and run settings at the top of each experiment before running it. Launch commands from the repository folder:

```bash
python scripts/run.py extract-activations
python scripts/run.py select-candidates
python scripts/run.py estimate-effects
```

Next, run `notebooks/rq1_features/03_select_core_features.ipynb` to create the final selected-feature tables and graph edges. With those generated inputs, use the relevant stage:

```bash
python scripts/run.py ablate-features
python scripts/run.py patch-activations
python scripts/run.py adjacent-mediation
python scripts/run.py targeted-amplification
python scripts/run.py train-steering
```

Global steering loads the supplied training and held-out CSVs directly. External evaluation uses EmoEvent and XED. The targeted multiplier sweep uses the shared held-out data. Each stage saves the outputs needed by later stages under `data/`.

For baselines, use the notebooks in `notebooks/baselines/`; both the 400-step and 600-step emotion-vector variants are available. For lexical control, run `python scripts/run.py prepare-lexical-control`, then `notebooks/evaluation/02_evaluate_lexical_control.ipynb` using generated steering checkpoints. Prompt robustness recovers raw sentences before applying each alternative prompt once; rerun it to obtain corrected scores in `data/prompt_robustness_evaluation_raw_text/`.

Run the first setup cell when opening a notebook. It sets the repository root and shared-module imports. List all launcher commands or check the source files without loading models:

```bash
python scripts/run.py --list
python scripts/check_repository.py
```

CPU data and input-construction checks have passed. GPU experiments and paper scores have not been independently reproduced in this cleanup.
