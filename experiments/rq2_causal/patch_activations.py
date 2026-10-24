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
import os
import traceback
import torch

from sae_emotion.activation_patching_utils import (
    run_cross_emotion_layer_feature_swap_pipeline,
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
    "google/gemma-3-1b-it",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
]

SKIP_MODELS = {
    "google/gemma-3-1b-pt",
    "google/gemma-3-1b-it",
}

DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH_SIZE = 128

BASE_DATA_DIR = "data"

CONTINUE_ON_ERROR = True

# ============================================================
# Helpers
# ============================================================

def get_sae_sizes_for_model(model_name: str):
    """
    Gemma-2 and Llama models do not use small/big SAE size.
    Gemma-3 4B/12B models use both small and big.
    """
    if model_name.startswith("google/gemma-2"):
        return [None]

    if model_name.startswith("meta-llama/"):
        return [None]

    if model_name.startswith("google/gemma-3"):
        return ["small", "big"]

    return [None]


def get_feature_dir(model_name: str, sae_size):
    model_slug = model_name.replace("/", "-")

    if sae_size is None:
        return f"{BASE_DATA_DIR}/{model_slug}-feature-act"

    return f"{BASE_DATA_DIR}/{model_slug}-feature-act-{sae_size}"


def get_paths(model_name: str, sae_size):
    model_slug = model_name.replace("/", "-")
    feature_dir = get_feature_dir(model_name, sae_size)

    pred_path = f"{BASE_DATA_DIR}/combined_zeroshot_preds_{model_slug}.csv"
    feature_path = f"{feature_dir}/auto_selected_unique_features_with_affected_emotions.csv"
    output_dir = f"{feature_dir}/cross_emotion_layer_feature_swap"

    return {
        "model_slug": model_slug,
        "feature_dir": feature_dir,
        "pred_path": pred_path,
        "feature_path": feature_path,
        "output_dir": output_dir,
    }


def check_required_files(paths):
    missing = []

    if not os.path.exists(paths["pred_path"]):
        missing.append(paths["pred_path"])

    if not os.path.exists(paths["feature_path"]):
        missing.append(paths["feature_path"])

    return missing


# ============================================================
# Main
# ============================================================

all_runs = []

for model_name in MODELS:
    if model_name in SKIP_MODELS:
        print("\n===================================================")
        print(f"Skipping model: {model_name}")
        print("===================================================")
        continue

    sae_sizes = get_sae_sizes_for_model(model_name)

    for sae_size in sae_sizes:
        paths = get_paths(model_name, sae_size)

        print("\n===================================================")
        print("Starting cross-emotion layer feature swap run")
        print("===================================================")
        print("MODEL_NAME:", model_name)
        print("SAE_SIZE:", sae_size)
        print("pred_path:", paths["pred_path"])
        print("feature_path:", paths["feature_path"])
        print("output_dir:", paths["output_dir"])

        missing = check_required_files(paths)
        if len(missing) > 0:
            print("\nSkipping because required files are missing:")
            for p in missing:
                print("  -", p)

            all_runs.append({
                "model_name": model_name,
                "sae_size": sae_size,
                "status": "skipped_missing_files",
                "error": "; ".join(missing),
            })
            continue

        try:
            feature_mean_df, instance_df, summary_df = (
                run_cross_emotion_layer_feature_swap_pipeline(
                    model_name=model_name,
                    pred_path=paths["pred_path"],
                    feature_path=paths["feature_path"],
                    output_dir=paths["output_dir"],
                    device=DEVICE,
                    dtype=DTYPE,
                    batch_size=BATCH_SIZE,
                    resume=True,
                    sae_size=sae_size,
                )
            )

            print("\nFinished cross-emotion layer feature swap run")
            print("MODEL_NAME:", model_name)
            print("SAE_SIZE:", sae_size)
            print("Feature mean rows:", len(feature_mean_df))
            print("Instance rows:", len(instance_df))
            print("Summary rows:", len(summary_df))

            all_runs.append({
                "model_name": model_name,
                "sae_size": sae_size,
                "status": "done",
                "error": "",
            })

        except Exception as e:
            print("\nRun failed")
            print("MODEL_NAME:", model_name)
            print("SAE_SIZE:", sae_size)
            print("Error:", repr(e))
            traceback.print_exc()

            all_runs.append({
                "model_name": model_name,
                "sae_size": sae_size,
                "status": "failed",
                "error": repr(e),
            })

            if not CONTINUE_ON_ERROR:
                raise

print("\n===================================================")
print("All requested cross-emotion swap runs finished or skipped.")
print("===================================================")

for r in all_runs:
    print(r)