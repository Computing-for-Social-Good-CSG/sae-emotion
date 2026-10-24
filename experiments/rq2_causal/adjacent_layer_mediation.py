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
import traceback
import torch

from sae_emotion.adjacent_group_path_patching_utils import run_adjacent_group_path_patching_pipeline

# ============================================================
# Config
# ============================================================

MODEL_RUNS = [

    # Exclude gemma-3-1b models.
    # ("google/gemma-3-1b-pt", None),
    # ("google/gemma-3-1b-it", None),

    # Run both big and small for Gemma-3 4B.
    ("google/gemma-3-4b-pt", "big"),
    ("google/gemma-3-4b-pt", "small"),
    ("google/gemma-3-4b-it", "big"),
    ("google/gemma-3-4b-it", "small"),

    # Llama uses None.
    ("meta-llama/Llama-3.1-8B", None),

    # 12B models use small, not big.
    ("google/gemma-3-12b-pt", "small"),
    ("google/gemma-3-12b-it", "small"),
]

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

SAMPLE_SIZE = 200
BATCH_SIZE = 64
DEVICE = "cuda"
DTYPE = torch.bfloat16
CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

MIN_LAYER = None
MAX_LAYER = None
MIN_NODE_SCORE = None
EXCLUDE_LAYERS = None
MAX_FEATURES_PER_LAYER = None


def build_feature_dir(model_slug: str, sae_size):
    if sae_size is None:
        return f"data/{model_slug}-feature-act"
    return f"data/{model_slug}-feature-act-{sae_size}"


all_outputs = {}
failed_models = []

for model_name, sae_size in MODEL_RUNS:
    model_slug = model_name.replace("/", "-")
    feature_dir = build_feature_dir(model_slug, sae_size)

    pred_path = f"data/combined_zeroshot_filtered_{model_slug}.csv"

    edge_path = os.path.join(
        feature_dir,
        "emotion_feature_edges",
        "all_affected_feature_edges.csv",
    )

    output_root = os.path.join(
        feature_dir,
        "adjacent_group_route_mediation",
    )

    run_key = f"{model_slug}__sae_{sae_size if sae_size is not None else 'none'}"

    print("\n" + "=" * 90)
    print(f"Running model: {model_name}")
    print(f"Model slug:    {model_slug}")
    print(f"SAE size:      {sae_size}")
    print(f"Feature dir:   {feature_dir}")
    print(f"Pred path:     {pred_path}")
    print(f"Edge path:     {edge_path}")
    print(f"Output root:   {output_root}")
    print("=" * 90)

    if not os.path.exists(pred_path):
        print(f"[SKIP] Missing pred file: {pred_path}")
        failed_models.append((run_key, "missing pred_path"))
        continue

    if not os.path.exists(edge_path):
        print(f"[SKIP] Missing edge file: {edge_path}")
        failed_models.append((run_key, "missing edge_path"))
        continue

    try:
        outputs = run_adjacent_group_path_patching_pipeline(
            model_name=model_name,
            pred_path=pred_path,
            edge_path=edge_path,
            output_root=output_root,
            emotions=EMOTIONS,

            min_layer=MIN_LAYER,
            max_layer=MAX_LAYER,
            min_node_score=MIN_NODE_SCORE,
            exclude_layers=EXCLUDE_LAYERS,

            min_features_per_layer=1,
            max_features_per_layer=MAX_FEATURES_PER_LAYER,

            sample_size=SAMPLE_SIZE,
            random_state=42,
            batch_size=BATCH_SIZE,
            sae_size=sae_size,
            device=DEVICE,
            dtype=DTYPE,
            cache_dir=CACHE_DIR,
            resume=True,
        )

        all_outputs[run_key] = outputs

        print("\nFinished:", run_key)
        print("Selected nodes:", outputs["selected_nodes_path"])
        print("Specs:", outputs["specs_path"])
        print("Instance results:", outputs["all_instances_path"])
        print("Pair summary:", outputs["pair_summary_path"])
        print("Emotion summary:", outputs["emotion_summary_path"])

    except Exception as e:
        print(f"\n[FAILED] {run_key}")
        print(type(e).__name__, str(e))
        traceback.print_exc()
        failed_models.append((run_key, str(e)))
        continue

print("\n" + "=" * 90)
print("Sweep finished.")
print("Successful runs:", len(all_outputs))
for run_key in all_outputs:
    print("  [OK]", run_key)

print("\nFailed/skipped runs:", len(failed_models))
for run_key, reason in failed_models:
    print("  [FAILED/SKIPPED]", run_key, ":", reason)
print("=" * 90)