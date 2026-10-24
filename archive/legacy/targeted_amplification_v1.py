import os

CACHE_ROOT = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_ROOT}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_ROOT}/hub"
os.environ["TORCH_HOME"] = f"{CACHE_ROOT}/torch"
os.environ["XDG_CACHE_HOME"] = CACHE_ROOT

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Authentication is read from HF_TOKEN or the local Hugging Face login.
# Authentication is read from HF_TOKEN or the local Hugging Face login.
import gc
from typing import List, Tuple, Dict, Optional

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from sae_emotion.logit_ablation_utils import (
    load_llm,
    build_emotion_token_map,
    SAECache,
    zero_shot_template,
    cleanup_memory,
)


# ============================================================
# Config
# ============================================================

MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.1-8B",
]

FEATURE_SETS_BY_MODEL = {
    "google/gemma-2-2b": [
        [(22, 11640)],
        [(22, 15295)],
    ],
    "google/gemma-2-9b": [
        [(35, 14786), (35, 10444)],
        [(35, 15406)],
    ],
    "google/gemma-3-4b-pt": [
        [(31, 1519)],
        [(31, 13001), (31, 1195)],
    ],
    "google/gemma-3-4b-it": [
        [(31, 2247), (31, 13175), (31, 2247), (31, 799)],
        [(31, 13001)],
    ],
    "google/gemma-3-12b-pt": [
        [(42, 12673), (42, 10933), (42, 8678)],
        [(42, 3753)],
    ],
    "google/gemma-3-12b-it": [
        [(42, 8678)],
        [(42, 3753)],
    ],
    "meta-llama/Llama-3.1-8B": [
        [(26, 4352)],
        [(26, 11515)],
    ],
}

MULTIPLIER_VALUES = [1.2, 1.5, 2, 4, 8, 12]

DATASET_CSV_PATH = "data/self_reported_ekman_zeroshot_preds.csv"

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

DEVICE = "cuda"
DTYPE = torch.bfloat16
CACHE_DIR = None

TEXT_COL = "input_text"
LABEL_COL = "emotion"

BATCH_SIZE = 128
APPLY_PROMPT_TEMPLATE = False
MAX_EXAMPLES = None


# ============================================================
# Global runtime objects
# These are reset for each model in the model loop
# ============================================================

llm = None
emotion_to_token = None
emotion_token_ids = None
sae_cache = None


# ============================================================
# Helpers
# ============================================================

def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def get_sae_size_for_model(model_name: str):
    if "gemma-3" in model_name:
        return "small"
    return None

def normalize_feature_set(target_features):
    if isinstance(target_features, tuple):
        return [target_features]
    return list(target_features)


def feature_set_tag(target_features) -> str:
    target_features = normalize_feature_set(target_features)
    return "__".join([f"L{l}_F{f}" for l, f in target_features])


def build_steering_dict(
    target_features: List[Tuple[int, int]],
    multipliers: List[float],
) -> Dict[int, Dict[int, float]]:
    if len(target_features) != len(multipliers):
        raise ValueError("target_features and multipliers must have the same length.")

    steering_dict = {}

    for (layer_idx, feature_idx), multiplier in zip(target_features, multipliers):
        layer_idx = int(layer_idx)
        feature_idx = int(feature_idx)
        multiplier = float(multiplier)

        if layer_idx not in steering_dict:
            steering_dict[layer_idx] = {}

        steering_dict[layer_idx][feature_idx] = multiplier

    return steering_dict


def multiply_sae_features(
    sae,
    source_sae_features: torch.Tensor,
    feature_to_multiplier: Dict[int, float],
    residual: torch.Tensor,
) -> torch.Tensor:
    modified_sae_features = source_sae_features.clone()

    for feature_idx, multiplier in feature_to_multiplier.items():
        feature_idx = int(feature_idx)
        multiplier = float(multiplier)

        modified_sae_features[:, feature_idx] = (
            modified_sae_features[:, feature_idx] * multiplier
        )

    modified_hidden = sae.decode(modified_sae_features)
    return modified_hidden + residual


# ============================================================
# Prediction with optional steering
# ============================================================

@torch.no_grad()
def predict_batch_with_optional_steering(
    examples: List[str],
    steering_dict: Optional[Dict[int, Dict[int, float]]] = None,
) -> List[str]:
    global llm, emotion_token_ids, sae_cache

    if steering_dict is None or len(steering_dict) == 0:
        with llm.trace(examples):
            emotion_logits = llm.output.logits[:, -1, emotion_token_ids].save()

        pred_idx = emotion_logits.argmax(dim=-1).detach().cpu().tolist()
        return [EMOTIONS[i] for i in pred_idx]

    layer_cache = {}

    with llm.trace(examples):
        for layer_idx in sorted(steering_dict.keys()):
            layer_idx = int(layer_idx)

            source_hidden = (
                llm.model.layers[layer_idx]
                .output[:, -1, :]
                .save()
            )

            layer_cache[layer_idx] = source_hidden

    steered_hidden_by_layer = {}

    for layer_idx, source_hidden in layer_cache.items():
        sae = sae_cache.load(layer_idx)

        source_sae_features = sae.encode(source_hidden)
        source_hidden_decoded = sae.decode(source_sae_features)
        residual = source_hidden - source_hidden_decoded

        steered_hidden = multiply_sae_features(
            sae=sae,
            source_sae_features=source_sae_features,
            feature_to_multiplier=steering_dict[layer_idx],
            residual=residual,
        )

        steered_hidden_by_layer[layer_idx] = steered_hidden

    with llm.trace(examples):
        for layer_idx, steered_hidden in steered_hidden_by_layer.items():
            layer_idx = int(layer_idx)

            llm.model.layers[layer_idx].output[:, -1, :] = steered_hidden

        emotion_logits = llm.output.logits[:, -1, emotion_token_ids].save()

    pred_idx = emotion_logits.argmax(dim=-1).detach().cpu().tolist()
    pred_emotions = [EMOTIONS[i] for i in pred_idx]

    cleanup_memory()

    return pred_emotions


# ============================================================
# Evaluation
# ============================================================

def evaluate_before_after_steering(
    dataset_csv_path: str,
    steering_dict: Dict[int, Dict[int, float]],
    batch_size: int = 128,
    text_col: str = "input_text",
    label_col: str = "emotion",
    apply_prompt_template: bool = False,
    max_examples: Optional[int] = None,
):
    df = pd.read_csv(dataset_csv_path)

    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(EMOTIONS)].copy()

    if max_examples is not None:
        df = df.iloc[:max_examples].copy()

    if apply_prompt_template:
        df["_prompt"] = df[text_col].apply(zero_shot_template)
    else:
        df["_prompt"] = df[text_col].astype(str)

    preds_before = []
    preds_after = []

    for start in tqdm(range(0, len(df), batch_size), desc="Evaluating"):
        batch_df = df.iloc[start:start + batch_size]
        examples = batch_df["_prompt"].tolist()

        batch_preds_before = predict_batch_with_optional_steering(
            examples=examples,
            steering_dict=None,
        )

        batch_preds_after = predict_batch_with_optional_steering(
            examples=examples,
            steering_dict=steering_dict,
        )

        preds_before.extend(batch_preds_before)
        preds_after.extend(batch_preds_after)

    df["pred_before"] = preds_before
    df["pred_after"] = preds_after

    df["correct_before"] = df["pred_before"] == df[label_col]
    df["correct_after"] = df["pred_after"] == df[label_col]
    df["changed"] = df["pred_before"] != df["pred_after"]

    y_true = df[label_col].tolist()
    y_before = df["pred_before"].tolist()
    y_after = df["pred_after"].tolist()

    before_macro_f1 = f1_score(
        y_true, y_before, labels=EMOTIONS, average="macro", zero_division=0
    )
    after_macro_f1 = f1_score(
        y_true, y_after, labels=EMOTIONS, average="macro", zero_division=0
    )

    before_weighted_f1 = f1_score(
        y_true, y_before, labels=EMOTIONS, average="weighted", zero_division=0
    )
    after_weighted_f1 = f1_score(
        y_true, y_after, labels=EMOTIONS, average="weighted", zero_division=0
    )

    before_acc = df["correct_before"].mean()
    after_acc = df["correct_after"].mean()

    summary = {
        "before_accuracy": float(before_acc),
        "after_accuracy": float(after_acc),
        "delta_accuracy": float(after_acc - before_acc),
        "before_macro_f1": float(before_macro_f1),
        "after_macro_f1": float(after_macro_f1),
        "delta_macro_f1": float(after_macro_f1 - before_macro_f1),
        "before_weighted_f1": float(before_weighted_f1),
        "after_weighted_f1": float(after_weighted_f1),
        "delta_weighted_f1": float(after_weighted_f1 - before_weighted_f1),
        "n_examples": int(len(df)),
        "n_changed_predictions": int(df["changed"].sum()),
        "changed_prediction_rate": float(df["changed"].mean()),
    }

    before_report = classification_report(
        y_true,
        y_before,
        labels=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )

    after_report = classification_report(
        y_true,
        y_after,
        labels=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )

    per_emotion_rows = []

    for emotion in EMOTIONS:
        emotion_mask = df[label_col] == emotion

        row = {
            "emotion": emotion,
            "n": int(emotion_mask.sum()),

            "precision_before": float(before_report[emotion]["precision"]),
            "recall_before": float(before_report[emotion]["recall"]),
            "f1_before": float(before_report[emotion]["f1-score"]),

            "precision_after": float(after_report[emotion]["precision"]),
            "recall_after": float(after_report[emotion]["recall"]),
            "f1_after": float(after_report[emotion]["f1-score"]),

            "delta_precision": float(
                after_report[emotion]["precision"]
                - before_report[emotion]["precision"]
            ),
            "delta_recall": float(
                after_report[emotion]["recall"]
                - before_report[emotion]["recall"]
            ),
            "delta_f1": float(
                after_report[emotion]["f1-score"]
                - before_report[emotion]["f1-score"]
            ),

            "change_rate": float(df.loc[emotion_mask, "changed"].mean()),
        }

        per_emotion_rows.append(row)

    per_emotion = pd.DataFrame(per_emotion_rows)

    print("\nOverall performance")
    print(pd.DataFrame([summary]))

    print("\nPer-emotion F1 summary")
    print(per_emotion)

    return summary, per_emotion, df


# ============================================================
# Multiplier sweep
# ============================================================

def run_multiplier_sweep(
    model_name: str,
    sae_size,
    target_features,
    multiplier_values: List[float],
    dataset_csv_path: str,
    output_csv_path: str,
    batch_size: int = 128,
    text_col: str = "input_text",
    label_col: str = "emotion",
    apply_prompt_template: bool = False,
    max_examples: Optional[int] = None,
):
    target_features = normalize_feature_set(target_features)

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    per_emotion_path = output_csv_path.replace(".csv", "_per_emotion.csv")
    all_predictions_path = output_csv_path.replace(".csv", "_all_predictions.csv")

    rows = []
    per_emotion_dfs = []
    all_pred_dfs = []

    for multiplier in multiplier_values:
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")
        print(f"SAE size: {sae_size}")
        print(f"Target features: {target_features}")
        print(f"Multiplier: {multiplier}")
        print("=" * 80)

        steering_dict = build_steering_dict(
            target_features=target_features,
            multipliers=[multiplier] * len(target_features),
        )

        summary, per_emotion, pred_df = evaluate_before_after_steering(
            dataset_csv_path=dataset_csv_path,
            steering_dict=steering_dict,
            batch_size=batch_size,
            text_col=text_col,
            label_col=label_col,
            apply_prompt_template=apply_prompt_template,
            max_examples=max_examples,
        )

        row = {
            "model_name": model_name,
            "sae_size": sae_size,
            "feature_set": feature_set_tag(target_features),
            "features": str(target_features),
            "multiplier": float(multiplier),
            **summary,
        }

        rows.append(row)

        per_emotion_copy = per_emotion.copy()
        per_emotion_copy["model_name"] = model_name
        per_emotion_copy["sae_size"] = sae_size
        per_emotion_copy["feature_set"] = feature_set_tag(target_features)
        per_emotion_copy["features"] = str(target_features)
        per_emotion_copy["multiplier"] = float(multiplier)
        per_emotion_dfs.append(per_emotion_copy)

        pred_df_copy = pred_df.copy()
        pred_df_copy["model_name"] = model_name
        pred_df_copy["sae_size"] = sae_size
        pred_df_copy["feature_set"] = feature_set_tag(target_features)
        pred_df_copy["features"] = str(target_features)
        pred_df_copy["multiplier"] = float(multiplier)
        all_pred_dfs.append(pred_df_copy)

        pd.DataFrame(rows).to_csv(output_csv_path, index=False)

        pd.concat(per_emotion_dfs, ignore_index=True).to_csv(
            per_emotion_path,
            index=False,
        )

        pd.concat(all_pred_dfs, ignore_index=True).to_csv(
            all_predictions_path,
            index=False,
        )

        print(f"Saved summary to: {output_csv_path}")
        print(f"Saved per-emotion results to: {per_emotion_path}")
        print(f"Saved all predictions to: {all_predictions_path}")

    sweep_df = pd.DataFrame(rows)
    per_emotion_df = pd.concat(per_emotion_dfs, ignore_index=True)
    all_predictions_df = pd.concat(all_pred_dfs, ignore_index=True)

    return sweep_df, per_emotion_df, all_predictions_df


# ============================================================
# Run all models
# ============================================================

all_sweep_dfs = []
all_per_emotion_dfs = []

for MODEL_NAME in MODELS:
    print("\n" + "#" * 100)
    print(f"Loading model: {MODEL_NAME}")
    print("#" * 100)

    SAE_SIZE = get_sae_size_for_model(MODEL_NAME)
    model_slug = model_safe_name(MODEL_NAME)

    llm = load_llm(
        model_name=MODEL_NAME,
        device=DEVICE,
        dtype=DTYPE,
        cache_dir=CACHE_DIR,
    )

    emotion_to_token = build_emotion_token_map(
        llm=llm,
        emotions=EMOTIONS,
    )

    emotion_token_ids = [emotion_to_token[e] for e in EMOTIONS]

    sae_cache = SAECache(
        model_name=MODEL_NAME,
        device=DEVICE,
        sae_size=SAE_SIZE,
    )

    for run_idx, target_features in enumerate(FEATURE_SETS_BY_MODEL[MODEL_NAME], start=1):
        tag = feature_set_tag(target_features)

        output_dir = f"data/{model_slug}-feature-act/manual_weak_emotion_boosting"
        os.makedirs(output_dir, exist_ok=True)

        sweep_output_path = os.path.join(
            output_dir,
            f"multiplier_sweep_run{run_idx}_{tag}.csv",
        )

        sweep_df, per_emotion_df, all_predictions_df = run_multiplier_sweep(
            model_name=MODEL_NAME,
            sae_size=SAE_SIZE,
            target_features=target_features,
            multiplier_values=MULTIPLIER_VALUES,
            dataset_csv_path=DATASET_CSV_PATH,
            output_csv_path=sweep_output_path,
            batch_size=BATCH_SIZE,
            text_col=TEXT_COL,
            label_col=LABEL_COL,
            apply_prompt_template=APPLY_PROMPT_TEMPLATE,
            max_examples=MAX_EXAMPLES,
        )

        sweep_df["run_idx"] = run_idx
        per_emotion_df["run_idx"] = run_idx

        all_sweep_dfs.append(sweep_df)
        all_per_emotion_dfs.append(per_emotion_df)

    del llm
    del sae_cache
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Save aggregate files
# ============================================================

all_sweeps_df = pd.concat(all_sweep_dfs, ignore_index=True)
all_per_emotion_df = pd.concat(all_per_emotion_dfs, ignore_index=True)

all_sweeps_df.to_csv(
    "data/all_models_manual_weak_emotion_boosting_summary.csv",
    index=False,
)

all_per_emotion_df.to_csv(
    "data/all_models_manual_weak_emotion_boosting_per_emotion.csv",
    index=False,
)

print("\nDone.")
print("Saved overall summary to:")
print("data/all_models_manual_weak_emotion_boosting_summary.csv")
print("Saved per-emotion summary to:")
print("data/all_models_manual_weak_emotion_boosting_per_emotion.csv")