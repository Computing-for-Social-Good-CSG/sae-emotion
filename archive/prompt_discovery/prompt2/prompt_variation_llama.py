# ============================================================
# Local setup
# ============================================================

import os
from typing import Callable, List, Optional


def setup_hf_cache(cache_root: str = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))):
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_DATASETS_CACHE"] = f"{cache_root}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cache_root}/hub"
    os.environ["TORCH_HOME"] = f"{cache_root}/torch"
    os.environ["XDG_CACHE_HOME"] = cache_root


setup_hf_cache()

import numpy as np
import pandas as pd
import torch

from utils import (
    cleanup_memory,
    extract_hidden_states_and_predictions,
    load_llm,
    model_safe_name,
    run_sae_activation_pipeline,
)
from feature_utils import (
    enrich_with_descriptions,
    extract_nonzero_activated_features,
    save_feature_tables,
)
from logit_ablation_utils import (
    run_feature_logit_ablation_pipeline,
    run_topk_joint_ablation_pipeline,
    run_topk_selection_pipeline,
)
from emotion_entropy_utils import run_entropy_classification_pipeline


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# Prompt templates
# ============================================================


def zero_shot_prompt2(text: str) -> str:
    return (
        "What is the inferred emotion in the following context?\n"
        f"Context: {text}.\n"
        "Answer:"
    )


def zero_shot_prompt3(text: str) -> str:
    return (
        "Consider this list of emotions: anger, disgust, fear, joy, sadness, surprise.\n"
        "What is the inferred emotion in the following context?\n"
        f"Context: {text}.\n"
        "Answer:"
    )


def zero_shot_prompt4(text: str) -> str:
    return f"Guess the emotion.\nContext: {text}.\nAnswer:"


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
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
]

MODEL_NAME = MODELS[5]
SAE_SIZE = "small"

PROMPT_NAME = "prompt2"
PROMPT_FN = zero_shot_prompt2

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

DATA_DIR = "data"
INPUT_CSV = f"{DATA_DIR}/self_reported_ekman_zeroshot_preds.csv"
TEXT_COL = "text"
LABEL_COL = "emotion"

HIDDEN_BATCH_SIZE = 16
SAE_BATCH_SIZE = 4096

ACTIVATION_RATE_MIN = 0.01
TOP_PERCENT = 0.1
RANKING_METRIC = "avg_activation"
FETCH_DESCRIPTIONS_AFTER_FILTERING = True

SAMPLE_SIZE_PER_EMOTION = 300
ABLATION_BATCH_SIZE = 128
SAVE_EVERY = 50
USE_ABS_LOGIT_CHANGE = False
RESUME_FROM_OUTPUT = True
PRESERVE_EXISTING_INPUT_LOGITS = False
APPLY_PROMPT_TEMPLATE = True

ENTROPY_THRESHOLD = None
SHOW_ENTROPY_PLOTS = True
TOP_K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
MIN_SIGNED_DROP = 0.1
COVERAGE_RATIO = 0.80

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

# Set any stage to False when its output already exists and you do not want to rerun it.
RUN_HIDDEN_AND_SAE = False
RUN_FEATURE_EXTRACTION = False
RUN_LOGIT_ABLATION = True
RUN_ENTROPY_CLASSIFICATION = True
RUN_TOPK_ABLATION_AND_SELECTION = True


# ============================================================
# Derived paths
# ============================================================

SAFE_NAME = model_safe_name(MODEL_NAME)
FEATURE_DIR = f"{DATA_DIR}/{SAFE_NAME}-feature-act"

SAE_ACTS_PATH = f"{DATA_DIR}/sae_acts_combined_zeroshot_{SAFE_NAME}.npy"
FILTERED_CSV_PATH = f"{DATA_DIR}/combined_zeroshot_filtered_{SAFE_NAME}.csv"

RAW_INPUT_FEATURE_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_top{TOP_PERCENT}_by_{RANKING_METRIC}.csv"
)
DEDUP_INPUT_FEATURE_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_top{TOP_PERCENT}_by_{RANKING_METRIC}_dedup.csv"
)
LOGIT_OUTPUT_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_filled_logits_signed_{PROMPT_NAME}.csv"
)
ENTROPY_OUTPUT_CSV_PATH = (
    f"{FEATURE_DIR}/core_emotion_features_filled_logits_signed_{PROMPT_NAME}_entropy.csv"
)
PLOT_DIR = f"{FEATURE_DIR}/visuals"

JOINT_OUT_PATH = (
    f"{FEATURE_DIR}/topk_joint_ablation_logit_drop_{PROMPT_NAME}.csv"
)
SELECTED_OUT_PATH = (
    f"{FEATURE_DIR}/auto_selected_emotion_specific_topk_features_{PROMPT_NAME}.csv"
)
UNIQUE_OUT_PATH = (
    f"{FEATURE_DIR}/auto_selected_unique_features_with_affected_emotions_{PROMPT_NAME}.csv"
)


# ============================================================
# Pipeline helpers
# ============================================================


def run_hidden_state_pipeline(
    model_name: str,
    input_csv: str,
    prompt_fn: Callable[[str], str],
    output_dir: str = "data",
    text_col: str = "text",
    label_col: Optional[str] = "emotion",
    emotions: Optional[List[str]] = None,
    batch_size: int = 4,
    device_map: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    llm = load_llm(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
    )

    df = pd.read_csv(input_csv)
    safe_name = model_safe_name(model_name)

    pred_path = f"{output_dir}/combined_zeroshot_preds_{safe_name}.csv"
    hidden_path = f"{output_dir}/combined_zeroshot_hidden_states_{safe_name}.npy"

    correct_path = None
    if label_col is not None:
        correct_path = f"{output_dir}/combined_zeroshot_filtered_{safe_name}.csv"

    df_out, activations = extract_hidden_states_and_predictions(
        llm=llm,
        df=df,
        text_col=text_col,
        emotions=emotions,
        batch_size=batch_size,
        save_hidden_path=hidden_path,
        save_preds_path=pred_path,
        save_correct_path=correct_path,
        label_col=label_col,
        prompt_fn=prompt_fn,
    )

    return df_out, activations


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

    for (layer_idx, feature_idx), group in df.groupby(
        ["layer_idx", "feature_idx"], sort=True
    ):
        emotions = sorted(group["emotion"].dropna().unique().tolist())

        out = {
            "layer_idx": int(layer_idx),
            "feature_idx": int(feature_idx),
            "emotions": ",".join(emotions),
            "num_source_emotions": len(emotions),
        }

        for meta_col in ["description", "description_from_cache"]:
            if meta_col in group.columns:
                values = group[meta_col].dropna().astype(str).unique().tolist()
                out[meta_col] = values[0] if values else pd.NA

        for _, row in group.iterrows():
            emotion = row["emotion"]
            for col in stat_cols:
                out[f"{col}_{emotion}"] = row[col]

        rows.append(out)

    out_df = pd.DataFrame(rows)

    base_cols = ["layer_idx", "feature_idx", "emotions", "num_source_emotions"]
    meta_cols = [
        c
        for c in ["description", "description_from_cache"]
        if c in out_df.columns
    ]

    emotion_order = ["joy", "surprise", "disgust", "anger", "sadness", "fear"]
    stat_emotion_cols = []

    for emotion in emotion_order:
        for col in stat_cols:
            full_col = f"{col}_{emotion}"
            if full_col in out_df.columns:
                stat_emotion_cols.append(full_col)

    other_cols = [
        c
        for c in out_df.columns
        if c not in base_cols + meta_cols + stat_emotion_cols
    ]

    out_df = out_df[base_cols + meta_cols + stat_emotion_cols + other_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    after = len(out_df)
    print("Deduplicated feature file:")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")
    print(f"  before rows: {before}")
    print(f"  after unique features: {after}")
    print(f"  removed duplicate rows: {before - after}")

    return output_path


# ============================================================
# Main
# ============================================================


def main():
    setup_hf_cache()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT_NAME}")

    if device != "cuda":
        raise RuntimeError("This pipeline is intended to run on CUDA.")

    if RUN_HIDDEN_AND_SAE:
        df_out, activations = run_hidden_state_pipeline(
            model_name=MODEL_NAME,
            input_csv=INPUT_CSV,
            output_dir=DATA_DIR,
            text_col=TEXT_COL,
            label_col=LABEL_COL,
            prompt_fn=PROMPT_FN,
            emotions=EMOTIONS,
            batch_size=HIDDEN_BATCH_SIZE,
            device_map=device,
            dtype=torch.bfloat16,
        )

        print("Finished hidden-state extraction.")

        sae_activations = run_sae_activation_pipeline(
            model_name=MODEL_NAME,
            output_dir=DATA_DIR,
            batch_size=SAE_BATCH_SIZE,
            device=device,
            sae_size=SAE_SIZE,
        )

        if sae_activations is not None:
            print("Finished SAE activation extraction.")
            print("SAE activations shape:", sae_activations.shape)

        del df_out, activations, sae_activations
        cleanup_memory()

    if RUN_FEATURE_EXTRACTION:
        sae_activations = np.load(SAE_ACTS_PATH)
        print("sae_activations shape:", sae_activations.shape)

        df_correct = pd.read_csv(FILTERED_CSV_PATH, index_col=0)
        print("df_correct shape:", df_correct.shape)

        df_features = extract_nonzero_activated_features(
            sae_activations=sae_activations,
            df_correct=df_correct,
            model_name=MODEL_NAME,
            emotions=EMOTIONS,
            emotion_col="pred_emotion",
            top_percent=None,
            ranking_metric=RANKING_METRIC,
            fetch_descriptions=False,
        )

        tables = save_feature_tables(
            df_features=df_features,
            output_dir=FEATURE_DIR,
            model_name=MODEL_NAME,
            activation_rate_min=ACTIVATION_RATE_MIN,
            top_percent=TOP_PERCENT,
            ranking_metric=RANKING_METRIC,
        )

        print("Saved feature tables.")
        print("nonzero:", tables["nonzero"].shape)
        print("effective:", tables["effective"].shape)
        print("core:", tables["core"].shape)

        if FETCH_DESCRIPTIONS_AFTER_FILTERING:
            core_path = tables["paths"]["core"]

            df_core_desc = enrich_with_descriptions(
                df_input=tables["core"],
                save_path=core_path,
                sleep_seconds=0.01,
                save_every=100,
            )

            print("Descriptions added.")
            print("core with descriptions:", df_core_desc.shape)
            del df_core_desc

        del sae_activations, df_correct, df_features, tables
        cleanup_memory()

    if RUN_LOGIT_ABLATION:
        dedup_input_path = deduplicate_feature_file(
            input_path=RAW_INPUT_FEATURE_CSV_PATH,
            output_path=DEDUP_INPUT_FEATURE_CSV_PATH,
        )

        df_logits = run_feature_logit_ablation_pipeline(
            model_name=MODEL_NAME,
            sae_size=SAE_SIZE,
            input_feature_csv_path=dedup_input_path,
            output_csv_path=LOGIT_OUTPUT_CSV_PATH,
            dataset_csv_path=FILTERED_CSV_PATH,
            emotions=EMOTIONS,
            text_col=TEXT_COL,
            label_col=LABEL_COL,
            device=device,
            dtype=torch.bfloat16,
            # cache_dir=CACHE_DIR,
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
        print("Saved to:", LOGIT_OUTPUT_CSV_PATH)

        del df_logits
        cleanup_memory()

    if RUN_ENTROPY_CLASSIFICATION:
        entropy_df, threshold = run_entropy_classification_pipeline(
            input_path=LOGIT_OUTPUT_CSV_PATH,
            output_path=ENTROPY_OUTPUT_CSV_PATH,
            entropy_threshold=ENTROPY_THRESHOLD,
            plot_dir=PLOT_DIR,
            show_plots=SHOW_ENTROPY_PLOTS,
        )

        print("Final entropy threshold:", threshold)

        del entropy_df
        cleanup_memory()

    if RUN_TOPK_ABLATION_AND_SELECTION:
        joint_df = run_topk_joint_ablation_pipeline(
            model_name=MODEL_NAME,
            feature_csv_path=ENTROPY_OUTPUT_CSV_PATH,
            dataset_csv_path=FILTERED_CSV_PATH,
            output_csv_path=JOINT_OUT_PATH,
            emotions=EMOTIONS,
            top_k_list=TOP_K_LIST,
            sae_size=SAE_SIZE,
            # cache_dir=CACHE_DIR,
            sample_size_per_emotion=SAMPLE_SIZE_PER_EMOTION,
            batch_size=ABLATION_BATCH_SIZE,
            apply_prompt_template=APPLY_PROMPT_TEMPLATE,
            prompt_fn=PROMPT_FN,
            use_abs_ranking=False,
            device=device,
            dtype=torch.bfloat16,
        )

        selected_df, unique_df = run_topk_selection_pipeline(
            feature_csv_path=ENTROPY_OUTPUT_CSV_PATH,
            joint_ablation_csv_path=JOINT_OUT_PATH,
            selected_output_csv_path=SELECTED_OUT_PATH,
            unique_output_csv_path=UNIQUE_OUT_PATH,
            emotions=EMOTIONS,
            min_signed_drop=MIN_SIGNED_DROP,
            coverage_ratio=COVERAGE_RATIO,
        )

        print("Joint ablation shape:", joint_df.shape)
        print("Selected feature table shape:", selected_df.shape)
        print("Unique feature table shape:", unique_df.shape)


if __name__ == "__main__":
    main()