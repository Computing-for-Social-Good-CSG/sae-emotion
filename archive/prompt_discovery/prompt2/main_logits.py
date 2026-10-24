# ============================================================
# Local setup
# ============================================================
import os
def setup_hf_cache(cache_root: str = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))):
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_DATASETS_CACHE"] = f"{cache_root}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cache_root}/hub"
    os.environ["TORCH_HOME"] = f"{cache_root}/torch"
    os.environ["XDG_CACHE_HOME"] = cache_root

setup_hf_cache()
    
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

import os
import torch
import pandas as pd

from logit_ablation_utils import (
    run_feature_logit_ablation_pipeline,
    model_safe_name,
)


def zero_shot_prompt2(text: str) -> str:
    return f"What is the inferred emotion in the following context?\nContext: {text}.\nAnswer:"


def zero_shot_prompt3(text: str) -> str:
    return f"Consider this list of emotions: anger, disgust, fear, joy, sadness, surprise.\nWhat is the inferred emotion in the following context?\nContext: {text}.\nAnswer:"


def zero_shot_prompt4(text: str) -> str:
    return f"Guess the emotion.\nContext: {text}.\nAnswer:"


def deduplicate_feature_file(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    required_cols = ["layer_idx", "feature_idx", "emotion"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature file: {missing}")

    stat_cols = [
        "avg_activation",
        "avg_act_when_active",
        "activation_count",
        "activation_rate",
    ]

    missing_stats = [c for c in stat_cols if c not in df.columns]
    if missing_stats:
        raise ValueError(f"Missing stat columns in feature file: {missing_stats}")

    before = len(df)

    df["layer_idx"] = df["layer_idx"].astype(int)
    df["feature_idx"] = df["feature_idx"].astype(int)
    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()

    rows = []

    for (layer_idx, feature_idx), g in df.groupby(["layer_idx", "feature_idx"], sort=True):
        emotions = sorted(g["emotion"].dropna().unique().tolist())

        out = {
            "layer_idx": int(layer_idx),
            "feature_idx": int(feature_idx),
            "emotions": ",".join(emotions),
            "num_source_emotions": len(emotions),
        }

        # keep non-emotion-specific metadata if present
        for meta_col in ["description", "description_from_cache"]:
            if meta_col in g.columns:
                vals = g[meta_col].dropna().astype(str).unique().tolist()
                out[meta_col] = vals[0] if len(vals) > 0 else pd.NA

        # store stats as stat_{emotion}
        for _, row in g.iterrows():
            emotion = row["emotion"]

            for col in stat_cols:
                out[f"{col}_{emotion}"] = row[col]

        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Optional: stable column order
    base_cols = ["layer_idx", "feature_idx", "emotions", "num_source_emotions"]
    meta_cols = [c for c in ["description", "description_from_cache"] if c in out_df.columns]

    emotion_order = ["joy", "surprise", "disgust", "anger", "sadness", "fear"]
    stat_emotion_cols = []

    for emotion in emotion_order:
        for col in stat_cols:
            full_col = f"{col}_{emotion}"
            if full_col in out_df.columns:
                stat_emotion_cols.append(full_col)

    other_cols = [
        c for c in out_df.columns
        if c not in base_cols + meta_cols + stat_emotion_cols
    ]

    out_df = out_df[base_cols + meta_cols + stat_emotion_cols + other_cols]

    after = len(out_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print("Deduplicated feature file:")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")
    print(f"  before rows: {before}")
    print(f"  after unique features: {after}")
    print(f"  removed duplicate rows: {before - after}")

    return output_path


# ============================================================
# Config
# ============================================================

MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-3-1b-it",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # commented out
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
]

MODEL_NAME = MODELS[-2]
SAE_SIZE = "big"

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

SAFE_NAME = model_safe_name(MODEL_NAME)

FEATURE_DIR = f"data/{SAFE_NAME}-feature-act-{SAE_SIZE}"

RAW_INPUT_FEATURE_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_top0.1_by_avg_activation.csv"
)

DEDUP_INPUT_FEATURE_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_top0.1_by_avg_activation_dedup.csv"
)

PROMPT_NAME = "prompt2"
PROMPT_FN = zero_shot_prompt2

OUTPUT_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_filled_logits_signed_{PROMPT_NAME}.csv"
)

DATASET_CSV_PATH = (
    f"data/combined_zeroshot_filtered_{SAFE_NAME}.csv"
)

TEXT_COL = "input_text"
LABEL_COL = "emotion"

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

SAMPLE_SIZE_PER_EMOTION = 300
ABLATION_BATCH_SIZE = 128
SAVE_EVERY = 50

USE_ABS_LOGIT_CHANGE = False

# Important:
# If OUTPUT_CSV_PATH already exists and contains duplicates, delete it or use a new output path.
RESUME_FROM_OUTPUT = True
PRESERVE_EXISTING_INPUT_LOGITS = False
APPLY_PROMPT_TEMPLATE = True


# ============================================================
# Main
# ============================================================

def main():
    setup_hf_cache()

    device = get_device()
    print(f"Using device: {device}")

    if device != "cuda":
        raise RuntimeError("This ablation script is intended to run on CUDA.")

    dedup_input_path = deduplicate_feature_file(
        input_path=RAW_INPUT_FEATURE_CSV_PATH,
        output_path=DEDUP_INPUT_FEATURE_CSV_PATH,
    )

    df_logits = run_feature_logit_ablation_pipeline(
        model_name=MODEL_NAME,
        sae_size=SAE_SIZE,
        input_feature_csv_path=dedup_input_path,
        output_csv_path=OUTPUT_CSV_PATH,
        dataset_csv_path=DATASET_CSV_PATH,
        emotions=EMOTIONS,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        device=device,
        dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        sample_size_per_emotion=SAMPLE_SIZE_PER_EMOTION,
        random_state=42,
        apply_prompt_template=APPLY_PROMPT_TEMPLATE,
        ablation_batch_size=ABLATION_BATCH_SIZE,
        use_abs=USE_ABS_LOGIT_CHANGE,
        save_every=SAVE_EVERY,
        resume_from_output=RESUME_FROM_OUTPUT,
        preserve_existing_input_logits=PRESERVE_EXISTING_INPUT_LOGITS,
        backup_existing_output=True,
        prompt_fn=PROMPT_FN,
    )

    print("Final dataframe shape:", df_logits.shape)
    print("Saved to:", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()