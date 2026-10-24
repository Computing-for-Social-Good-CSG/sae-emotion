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


import torch

from sae_emotion.utils import (
    run_hidden_state_pipeline,
    run_sae_activation_pipeline,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
]

MODEL_NAME = MODELS[3]
SAE_SIZE = "small"

INPUT_CSV = "data/combined_emotions.csv"
OUTPUT_DIR = "data"

TEXT_COL = "text"
LABEL_COL = "emotion"

HIDDEN_BATCH_SIZE = 16
SAE_BATCH_SIZE = 4096

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]


# ============================================================
# Main
# ============================================================

def main():
    setup_hf_cache()

    device = get_device()
    print(f"Using device: {device}")

    device_map = device

    df_out, activations = run_hidden_state_pipeline(
        model_name=MODEL_NAME,
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        emotions=EMOTIONS,
        batch_size=HIDDEN_BATCH_SIZE,
        device_map=device_map,
        dtype=torch.bfloat16,
    )

    print("Finished hidden-state extraction.")
    # print("Hidden states shape:", activations.shape)

    sae_activations = run_sae_activation_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        batch_size=SAE_BATCH_SIZE,
        device="cuda" if torch.cuda.is_available() else device,
        sae_size=SAE_SIZE,
    )

    if sae_activations is not None:
        print("Finished SAE activation extraction.")
        print("SAE activations shape:", sae_activations.shape)


if __name__ == "__main__":
    main()