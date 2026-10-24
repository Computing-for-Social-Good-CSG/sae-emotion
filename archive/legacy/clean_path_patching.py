import os
import torch

from clean_path_patching_utils import run_full_path_patching_pipeline

# ============================================================
# User config
# ============================================================

MODEL_NAME = "google/gemma-2-2b"
SAE_SIZE = "big"  # "big", "medium", or "small" depending on model/SAE availability

PRED_PATH = "data/combined_zeroshot_preds_google-gemma-2-2b.csv"
EDGE_PATH = (
    "data/google-gemma-2-2b-feature-act/"
    "all_emotion_route_patching/"
    "all_emotion_layer_to_layer_route_edges.csv"
)

OUTPUT_ROOT = (
    "data/google-gemma-2-2b-feature-act/"
    "path_patching/clean_anchor_mid_final_pipeline"
)

PATHS = {
    "fear": {
        "source_layer": 18,
        "source_feature": 9484,
        "end_layer": 25,
        "end_feature": 3410,
    },
    "sadness": {
        "source_layer": 17,
        "source_feature": 7356,
        "end_layer": 25,
        "end_feature": 66,
    },
    "anger": {
        "source_layer": 17,
        "source_feature": 2792,
        "end_layer": 25,
        "end_feature": 8776,
    },
    "disgust": {
        "source_layer": 16,
        "source_feature": 15100,
        "end_layer": 25,
        "end_feature": 13837,
    },
    "surprise": {
        "source_layer": 16,
        "source_feature": 4462,
        "end_layer": 25,
        "end_feature": 1340,
    },
    "joy": {
        "source_layer": 16,
        "source_feature": 12800,
        "end_layer": 25,
        "end_feature": 11927,
    },
}

# ============================================================
# Run everything
# ============================================================

outputs = run_full_path_patching_pipeline(
    model_name=MODEL_NAME,
    pred_path=PRED_PATH,
    edge_path=EDGE_PATH,
    paths=PATHS,
    output_root=OUTPUT_ROOT,
    sae_size=SAE_SIZE,

    # Options: "single", "group", "both"
    patch_mode="both",

    # Mid-node selection
    min_edge_effect=0.03,
    top_k_mids_per_emotion=10,
    rank_col="bridge_score_product",

    # Group restore top-k nodes per layer
    group_top_k_list=[1, 2, 3, 5, 10],

    # Evaluation
    sample_size=200,
    batch_size=16,
    device="cuda",
    dtype=torch.bfloat16,
)

print("Done.")
print("Available output keys:", list(outputs.keys()))
