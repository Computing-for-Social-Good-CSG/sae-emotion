# SAE Emotion

Research code accompanying **Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders**, by Bangzhao Shu, Arinjay Singh, and Mai ElSherief.

## Setup

The research environment uses **Python 3.11**, **CUDA 12.6**, and an **NVIDIA H200**. Core versions of PyTorch, Transformers, SAELens, NNsight, Accelerate, and Datasets are listed in [requirements.txt](requirements.txt). Analysis and notebook dependencies are listed in `pyproject.toml`.

From the repository folder, install the dependencies and package:

```bash
python -m pip install -r requirements.txt -e '.[research,notebooks]'
```

If these dependencies are already installed in your environment, install only the local package:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Use your Hugging Face login or `HF_TOKEN` for model access. Set `HF_HOME` to choose a cache location.

## Repository

| Location | Contents |
| --- | --- |
| `src/sae_emotion/` | SAE, feature-selection, intervention, and plotting utilities |
| `experiments/rq1_features/` | Activation extraction, candidate selection, and causal effects |
| `experiments/rq2_causal/` | Feature ablation, activation patching, and adjacent-layer mediation |
| `experiments/rq3_steering/` | Targeted amplification and learned global steering |
| `experiments/evaluation/` | Lexical-control preprocessing and general-capability evaluation |
| `notebooks/` | Feature analysis, steering analysis, evaluation, and figures |
| `data/` | Source CSV files and steering splits and their column descriptions |

## Data

Datasets and steering splits are included in `data/`. Regenerate the shared splits with `python scripts/prepare_steering_splits.py`:

| File | Rows | Purpose |
| --- | ---: | --- |
| `combined_emotions.csv` | 9,870 | Combined emotion dataset (ISEAR, enISEAR, enVent, and EXPRESS) |
| `train_df.csv` | 1,919 | Steering training split |
| `val_df.csv` | 479 | Checkpoint and hyperparameter selection |
| `test_df.csv` | 1,974 | Final main-dataset evaluation |
| `feature_discovery_train.csv` | 1,917 | Training sentences with raw text for feature discovery |
| `emoevent_en.csv` | 7,303 | External EmoEvent dataset |
| `xed-en.csv` | 2,803 | External XED dataset |

See [data/README.md](data/README.md) for columns, file formats, and split details. Activation extraction generates model predictions in `data/combined_zeroshot_preds_<model>.csv`, along with the activation arrays needed for feature analysis. These generated files and subsequent checkpoints are stored locally under `data/` and excluded from Git by `.gitignore`.

## Run the pipeline

Set the model, SAE, and run parameters at the top of each experiment. Use the same model and SAE settings across stages, and run commands from the repository folder.

First, extract activations, select candidate features, and estimate their causal effects:

```bash
python scripts/run.py extract-activations
python scripts/run.py select-candidates
python scripts/run.py estimate-effects
```

Next, run `notebooks/rq1_features/03_select_core_features.ipynb` to generate the selected-feature tables and graph edges. Use these outputs for causal analysis and steering:

```bash
python scripts/run.py ablate-features
python scripts/run.py patch-activations
python scripts/run.py adjacent-mediation
python scripts/run.py targeted-amplification
python scripts/run.py train-steering
```

Global steering trains on `train_df.csv`, selects checkpoints on `val_df.csv`, and evaluates the selected checkpoint on `test_df.csv`, EmoEvent, and XED. Targeted amplification selects its multiplier on validation before evaluating the test set. Feature discovery uses `feature_discovery_train.csv`, excluding validation and test sentences. Each stage writes its outputs under `data/`.

For this protocol, regenerate activations, feature selections, and steering checkpoints from the first stage in a fresh output directory. Previous full-corpus feature tables and checkpoints are not inputs to these runs.

## Evaluation

Run the first setup cell when opening a notebook. It sets the repository root and imports the shared modules.

- **Lexical control:** run `python scripts/run.py prepare-lexical-control`, then `notebooks/evaluation/02_evaluate_lexical_control.ipynb` with the selected-feature files and trained steering checkpoints.
- **Prompt robustness:** run `notebooks/evaluation/prompt_robustness.ipynb`. It recovers the original sentences for the held-out rows and applies each alternative prompt once. Outputs are saved to `data/prompt_robustness_evaluation_raw_text/`.
- **General capabilities:** run `python scripts/run.py general-capabilities` with trained steering checkpoints.
- **Analysis and figures:** use the corresponding RQ notebooks and `notebooks/figures/paper_figures.ipynb` after generating experiment outputs.

List all available launcher commands:

```bash
python scripts/run.py --list
```
