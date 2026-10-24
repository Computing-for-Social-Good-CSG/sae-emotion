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
import os
import traceback

import torch

from sae_emotion.feature_steering_utils import (
    FeatureSteeringConfig,
    TestDatasetSpec,
    run_feature_steering_experiment,
)

# ============================================================
# User-level run controls
# ============================================================

MODELS = [
    "google/gemma-3-4b-it",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]

RUN_EPOCHS = 24
LAMBDA_L1_VALUES = [0.0]
LR = 1e-2

CACHE_DIR = None
WIKITEXT_CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
TRAIN_CSV_PATH = "data/train_df.csv"
VAL_CSV_PATH = "data/val_df.csv"
MAIN_CSV_PATH = "data/self_reported_ekman_zeroshot_preds.csv"
FEATURE_CSV_NAME = "auto_selected_unique_features_with_affected_emotions.csv"

GEMMA3_SMALL_MODELS = {
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
}

TEST_DATASETS = [
    TestDatasetSpec(
        name="xed_en",
        path="data/xed-en.csv",
        text_col="text",
        label_col="emotion",
        apply_prompt_template=True,
    ),
    TestDatasetSpec(
        name="emoevent_en",
        path="data/emoevent_en.csv",
        text_col="tweet",
        label_col="emotion",
        sep="\t",
        apply_prompt_template=True,
        drop_labels=["others"],
    ),
]


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def get_sae_size(model_name: str):
    if model_name in GEMMA3_SMALL_MODELS:
        return "small"
    return None


def get_feature_dir(model_name: str, sae_size):
    slug = model_safe_name(model_name)
    if sae_size == "small":
        return f"data/{slug}-feature-act-small"
    return f"data/{slug}-feature-act"


def get_output_dir(model_name: str, sae_size, lambda_l1: float):
    feature_dir = get_feature_dir(model_name, sae_size)
    l1_tag = str(lambda_l1).replace(".", "p")
    return os.path.join(feature_dir, "global_feature_steering", f"lr_5e-3_l1_{l1_tag}")


def cleanup_after_run():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def already_finished(output_dir: str, expected_epochs: int) -> bool:
    summary_path = os.path.join(output_dir, "summary.json")
    metrics_path = os.path.join(output_dir, "epoch_metrics_long.csv")
    final_mult_path = os.path.join(output_dir, "learned_feature_multipliers.csv")
    final_epoch_mult_path = os.path.join(
        output_dir,
        "epoch_multipliers",
        f"epoch_{expected_epochs:03d}_multipliers.csv",
    )
    return all(os.path.exists(p) for p in [summary_path, metrics_path, final_mult_path, final_epoch_mult_path])


def run_one(model_name: str, lambda_l1: float):
    sae_size = get_sae_size(model_name)
    feature_dir = get_feature_dir(model_name, sae_size)
    feature_csv_path = os.path.join(feature_dir, FEATURE_CSV_NAME)
    output_dir = get_output_dir(model_name, sae_size, lambda_l1)

    if not os.path.exists(feature_csv_path):
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv_path}")

    if already_finished(output_dir, RUN_EPOCHS):
        print(f"Skipping finished run: model={model_name}, sae_size={sae_size}, lambda_l1={lambda_l1}")
        return None, None

    print("=" * 100)
    print(f"Running model={model_name}")
    print(f"sae_size={sae_size}")
    print(f"lambda_l1={lambda_l1}")
    print(f"feature_csv_path={feature_csv_path}")
    print(f"output_dir={output_dir}")
    print("=" * 100)

    config = FeatureSteeringConfig(
        model_name=model_name,
        sae_size=sae_size,
        feature_csv_path=feature_csv_path,
        train_csv_path=TRAIN_CSV_PATH,
        val_csv_path=VAL_CSV_PATH,
        main_csv_path=MAIN_CSV_PATH,
        output_dir=output_dir,

        train_text_col="input_text",
        train_label_col="emotion",
        train_apply_prompt_template=False,

        max_per_class=400,
        num_epochs=RUN_EPOCHS,
        batch_size=16,
        eval_batch_size=16,
        ppl_batch_size=8,

        lambda_kl=0.000,
        lambda_l1=lambda_l1,
        lr=LR,

        cache_dir=CACHE_DIR,
        wikitext_cache_dir=WIKITEXT_CACHE_DIR,

        # Saves epoch 0 and every epoch for the main dataset and all external datasets.
        eval_main_full=True,
        eval_main_val=True,
        eval_general_ppl=True,
        eval_dataset_ppl=True,
        max_eval_ppl_texts=500,
        save_epoch_predictions=False,

        test_datasets=TEST_DATASETS,
    )

    finetuner, outputs = run_feature_steering_experiment(config)
    return finetuner, outputs


def main():
    failures = []

    for model_name in MODELS:
        for lambda_l1 in LAMBDA_L1_VALUES:
            finetuner = None
            outputs = None
            try:
                finetuner, outputs = run_one(model_name, lambda_l1)
            except Exception as exc:
                error_msg = f"FAILED model={model_name}, lambda_l1={lambda_l1}: {repr(exc)}"
                print(error_msg)
                traceback.print_exc()
                failures.append(error_msg)
            finally:
                if finetuner is not None:
                    del finetuner
                if outputs is not None:
                    del outputs
                cleanup_after_run()
                print("Released model, SAE cache, outputs, and CUDA cache.")

    if failures:
        print("\nSome runs failed:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("\nAll runs finished successfully.")


if __name__ == "__main__":
    main()
