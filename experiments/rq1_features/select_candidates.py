import numpy as np
import pandas as pd

from sae_emotion.feature_utils import (
    extract_nonzero_activated_features,
    save_feature_tables,
    enrich_with_descriptions,
    model_safe_name,
)


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
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]

MODEL_NAME = MODELS[1]
SAE_SIZE = "small"  # Used for Gemma-3; match the activation extraction run.

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

FEATURE_SUFFIX = f"-{SAE_SIZE}" if "gemma-3" in MODEL_NAME else ""
ACTIVATION_SUFFIX = f"_{SAE_SIZE}" if "gemma-3" in MODEL_NAME else ""
OUTPUT_DIR = f"data/{model_safe_name(MODEL_NAME)}-feature-act{FEATURE_SUFFIX}"

SAE_ACTS_PATH = (
    f"data/sae_acts_combined_zeroshot_{model_safe_name(MODEL_NAME)}{ACTIVATION_SUFFIX}.npy"
)

FILTERED_CSV_PATH = (
    f"data/combined_zeroshot_filtered_{model_safe_name(MODEL_NAME)}.csv"
)

ACTIVATION_RATE_MIN = 0.01
TOP_PERCENT = 0.1
RANKING_METRIC = "avg_activation"

FETCH_DESCRIPTIONS_AFTER_FILTERING = True


# ============================================================
# Main
# ============================================================

def main():
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
        output_dir=OUTPUT_DIR,
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


if __name__ == "__main__":
    main()
