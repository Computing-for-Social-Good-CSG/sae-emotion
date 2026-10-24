
"""
subtype_utils.py

Utilities for distributed emotion subroute / subtype analysis.

This module supports three stages:

1. Discover feature clusters and instance subtype clusters.
2. Optionally run subtype-specific causal ablation.
3. Plot and summarize subtype analysis outputs, even when ablation outputs are missing.

Typical usage:

from sae_emotion.subtype_utils import SubtypeAnalysisConfig, run_subtype_discovery, plot_subtype_analysis

config = SubtypeAnalysisConfig(
    model_name="google/gemma-2-9b",
    pred_path="data/combined_zeroshot_preds_google-gemma-2-9b.csv",
    act_path="../data/sae_acts_combined_zeroshot_google-gemma-2-9b.npy",
    feature_path=(
        "data/google-gemma-2-9b-feature-act/"
        "auto_selected_unique_features_with_affected_emotions.csv"
    ),
    output_dir=(
        "data/google-gemma-2-9b-feature-act/"
        "distributed_emotion_subroute_analysis"
    ),
    target_emotion="disgust",
    run_causal_ablation=False,
)

run_subtype_discovery(config)
plot_subtype_analysis(config)
"""

import os
import gc
import textwrap
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# Config
# ============================================================

@dataclass
class SubtypeAnalysisConfig:
    model_name: str
    pred_path: str
    act_path: str
    feature_path: str
    output_dir: str
    target_emotion: str
    sae_size: Optional[str] = None  # None, "small", or "big"

    emotions: List[str] = field(
        default_factory=lambda: [
            "joy",
            "surprise",
            "disgust",
            "anger",
            "sadness",
            "fear",
        ]
    )

    # Clustering settings
    min_feature_clusters: int = 1
    max_feature_clusters: int = 10
    feature_cluster_method: str = "agglomerative"  # "agglomerative" or "kmeans"
    instance_cluster_method: str = "kmeans"
    max_instance_clusters: int = 8

    # Activation threshold for active features
    active_threshold: float = 0.0

    # Top prompts to save per feature cluster
    top_prompts_per_cluster: int = 50
    readable_top_prompts_per_cluster: int = 20

    # Optional causal ablation settings
    run_causal_ablation: bool = False
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 8

    # Plot settings
    show_plots: bool = True
    dpi: int = 300

    @property
    def feature_cluster_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_feature_clusters.csv",
        )

    @property
    def instance_cluster_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_instance_subtype_clusters.csv",
        )

    @property
    def top_prompts_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_top_prompts_by_feature_cluster.csv",
        )

    @property
    def instance_cluster_feature_matrix_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_instance_cluster_feature_matrix.csv",
        )

    @property
    def feature_cluster_k_selection_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_feature_cluster_k_selection.csv",
        )

    @property
    def instance_subtype_k_selection_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_instance_subtype_k_selection.csv",
        )

    @property
    def cluster_ablation_instance_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_cluster_ablation_instance_level.csv",
        )

    @property
    def cluster_ablation_summary_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_cluster_ablation_summary.csv",
        )

    @property
    def causal_matrix_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.target_emotion}_cluster_x_subtype_causal_effect_matrix.csv",
        )

    @property
    def plot_dir(self) -> str:
        return os.path.join(self.output_dir, "plots")


# ============================================================
# General helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)
    print("Saved:", path)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def zero_shot_template(text: str) -> str:
    return f"In this text: {text}, the emotion implied is:"


def feature_affects_emotion(affected_emotions: str, emotion: str) -> bool:
    parts = [x.strip() for x in str(affected_emotions).split(",")]
    return emotion in parts


def file_exists(path: Optional[str]) -> bool:
    return path is not None and os.path.exists(path)


def safe_read_csv(
    path: str,
    name: str,
    index_col: Optional[int] = None,
    required: bool = True,
) -> Optional[pd.DataFrame]:
    if file_exists(path):
        print(f"Loading {name}: {path}")
        return pd.read_csv(path, index_col=index_col)

    msg = f"Missing {name}: {path}"

    if required:
        raise FileNotFoundError(msg)

    print(f"[SKIP] {msg}")
    return None


def print_file_info(name: str, df: Optional[pd.DataFrame]) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    if df is None:
        print("Skipped: file not available")
        return

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())


def clean_numeric(df: Optional[pd.DataFrame], cols: Sequence[str]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def require_columns(df: pd.DataFrame, cols: Sequence[str], df_name: str) -> None:
    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def savefig(config: SubtypeAnalysisConfig, name: str) -> None:
    ensure_dir(config.plot_dir)
    path = os.path.join(config.plot_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi, bbox_inches="tight")
    print("Saved:", path)

    if config.show_plots:
        plt.show()
    else:
        plt.close()


# ============================================================
# Data loading
# ============================================================

def load_correct_prediction_df(config: SubtypeAnalysisConfig) -> pd.DataFrame:
    df = pd.read_csv(config.pred_path)

    required = ["input_text", "emotion", "pred_emotion"]
    require_columns(df, required, "prediction file")

    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
    df["pred_emotion"] = df["pred_emotion"].astype(str).str.lower().str.strip()

    df = df[df["emotion"].isin(config.emotions)].copy()
    df = df[df["pred_emotion"].isin(config.emotions)].copy()

    df_correct = df[df["emotion"] == df["pred_emotion"]].copy()
    df_correct = df_correct.reset_index(drop=False).rename(
        columns={"index": "original_instance_idx"}
    )

    df_correct["prompt"] = df_correct["input_text"].apply(zero_shot_template)

    return df_correct


def load_target_features(config: SubtypeAnalysisConfig) -> pd.DataFrame:
    feature_df = pd.read_csv(config.feature_path)

    required = ["layer_idx", "feature_idx", "affected_emotions"]
    require_columns(feature_df, required, "feature file")

    feature_df["layer_idx"] = feature_df["layer_idx"].astype(int)
    feature_df["feature_idx"] = feature_df["feature_idx"].astype(int)
    feature_df["affected_emotions"] = (
        feature_df["affected_emotions"].fillna("").astype(str)
    )

    feature_df = feature_df.drop_duplicates(
        subset=["layer_idx", "feature_idx"]
    ).copy()

    target_features = feature_df[
        feature_df["affected_emotions"].apply(
            lambda x: feature_affects_emotion(x, config.target_emotion)
        )
    ].copy()

    target_features = target_features.sort_values(
        ["layer_idx", "feature_idx"]
    ).reset_index(drop=True)

    print(f"Target emotion: {config.target_emotion}")
    print("Target features:", target_features.shape)
    print(target_features.groupby("layer_idx")["feature_idx"].nunique())

    if len(target_features) == 0:
        raise ValueError(
            f"No features found for target emotion: {config.target_emotion}"
        )

    return target_features


def infer_sae_size_from_paths(config: SubtypeAnalysisConfig) -> Optional[str]:
    """
    Infer SAE size from config.sae_size, feature_path, output_dir, or act_path.

    Supported values are None, "small", and "big".
    """
    if config.sae_size in {"small", "big"}:
        return config.sae_size

    joined = " ".join(
        [
            str(config.feature_path),
            str(config.output_dir),
            str(config.act_path),
        ]
    )

    if "-small-feature-act" in joined or "_small.npy" in joined:
        return "small"

    if "-big-feature-act" in joined or "_big.npy" in joined:
        return "big"

    return None


def candidate_activation_paths(config: SubtypeAnalysisConfig) -> List[str]:
    """
    Return possible activation paths, including SAE-size suffix variants.

    Example:
    ../data/sae_acts_combined_zeroshot_google-gemma-3-4b-pt.npy
    -> ../data/sae_acts_combined_zeroshot_google-gemma-3-4b-pt_big.npy
    -> ../data/sae_acts_combined_zeroshot_google-gemma-3-4b-pt_small.npy
    """
    base_path = str(config.act_path)
    root, ext = os.path.splitext(base_path)

    candidates = []

    def add(path: str) -> None:
        if path not in candidates:
            candidates.append(path)

    sae_size = infer_sae_size_from_paths(config)

    # Prefer explicitly requested size first.
    if sae_size in {"small", "big"}:
        if root.endswith(f"_{sae_size}"):
            add(base_path)
        else:
            add(f"{root}_{sae_size}{ext}")

    # Then try the original path.
    add(base_path)

    # Then try both suffixes as fallback.
    for size in ["big", "small"]:
        if not root.endswith(f"_{size}"):
            add(f"{root}_{size}{ext}")

    return candidates


def resolve_activation_path(config: SubtypeAnalysisConfig) -> str:
    candidates = candidate_activation_paths(config)

    for path in candidates:
        if os.path.exists(path):
            if path != config.act_path:
                print(f"Resolved activation path: {path}")
            return path

    raise FileNotFoundError(
        "Could not find activation .npy file. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def load_activation_array(config: SubtypeAnalysisConfig):
    act_path = resolve_activation_path(config)
    acts = np.load(act_path, mmap_mode="r")
    print("Activation path:", act_path)
    print("Raw activation shape:", acts.shape)

    if acts.shape[0] < 100:
        print("Assuming activation shape = [layers, instances, features]")
    else:
        raise ValueError(
            "Unexpected activation shape. Expected [layers, instances, features]."
        )

    return acts


# ============================================================
# Build feature × instance activation matrix
# ============================================================

def build_target_activation_matrix(
    config: SubtypeAnalysisConfig,
    acts,
    df_correct: pd.DataFrame,
    target_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[int, int]], List[str]]:
    df_target = df_correct[df_correct["emotion"] == config.target_emotion].copy()
    df_target = df_target.reset_index(drop=True)

    if len(df_target) == 0:
        raise ValueError(
            f"No correctly predicted target examples for {config.target_emotion}"
        )

    instance_indices = df_target["original_instance_idx"].astype(int).to_numpy()

    feature_labels = []
    feature_keys = []
    feature_acts = []

    for _, row in target_features.iterrows():
        layer_idx = int(row["layer_idx"])
        feature_idx = int(row["feature_idx"])

        vec = np.asarray(
            acts[layer_idx, instance_indices, feature_idx],
            dtype=np.float32,
        )

        feature_acts.append(vec)
        feature_keys.append((layer_idx, feature_idx))
        feature_labels.append(f"L{layer_idx}-F{feature_idx}")

    X_feature_instance = np.stack(feature_acts, axis=0)

    print("Target examples:", len(df_target))
    print("Feature × instance matrix:", X_feature_instance.shape)

    return df_target, X_feature_instance, feature_keys, feature_labels


# ============================================================
# Clustering
# ============================================================

def _fit_cluster_model(
    X: np.ndarray,
    k: int,
    method: str,
    random_state: int = 42,
) -> np.ndarray:
    if k == 1:
        return np.zeros(X.shape[0], dtype=int)

    if method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(X)

    if method == "kmeans":
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )
        return model.fit_predict(X)

    raise ValueError(f"Unknown clustering method: {method}")


def choose_best_k(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 8,
    method: str = "agglomerative",
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    n_items = X.shape[0]
    max_k = min(max_k, n_items - 1)

    if min_k <= 1:
        candidate_min_k = 2
        allow_one_cluster = True
    else:
        candidate_min_k = min_k
        allow_one_cluster = False

    if max_k < candidate_min_k:
        return 1, pd.DataFrame()

    rows = []

    for k in range(candidate_min_k, max_k + 1):
        labels = _fit_cluster_model(
            X=X,
            k=k,
            method=method,
            random_state=random_state,
        )

        if len(set(labels)) <= 1:
            score = np.nan
        else:
            score = silhouette_score(X, labels, metric="cosine")

        rows.append(
            {
                "k": k,
                "silhouette": float(score),
            }
        )

    score_df = pd.DataFrame(rows)

    if len(score_df) == 0:
        return 1, score_df

    best_row = score_df.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])

    if allow_one_cluster and np.isnan(best_row["silhouette"]):
        best_k = 1

    return best_k, score_df


def cluster_features_by_activation(
    config: SubtypeAnalysisConfig,
    X_feature_instance: np.ndarray,
    feature_labels: List[str],
    target_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(
        X_feature_instance
    )

    best_k, score_df = choose_best_k(
        X,
        min_k=config.min_feature_clusters,
        max_k=config.max_feature_clusters,
        method=config.feature_cluster_method,
    )

    safe_save_csv(score_df, config.feature_cluster_k_selection_path)
    print("Best feature cluster k:", best_k)

    labels = _fit_cluster_model(
        X=X,
        k=best_k,
        method=config.feature_cluster_method,
    )

    cluster_df = target_features.copy()
    cluster_df["feature_label"] = feature_labels
    cluster_df["feature_cluster"] = labels.astype(int)

    cluster_df["mean_activation_on_target"] = X_feature_instance.mean(axis=1)
    cluster_df["activation_rate_on_target"] = (
        X_feature_instance > config.active_threshold
    ).mean(axis=1)

    cluster_df = cluster_df.sort_values(
        ["feature_cluster", "layer_idx", "feature_idx"]
    ).reset_index(drop=True)

    safe_save_csv(cluster_df, config.feature_cluster_path)

    print("\nFeature cluster sizes:")
    print(cluster_df["feature_cluster"].value_counts().sort_index())

    return cluster_df, labels


def build_instance_cluster_features(
    config: SubtypeAnalysisConfig,
    X_feature_instance: np.ndarray,
    feature_cluster_labels: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    cluster_ids = sorted(set(feature_cluster_labels.tolist()))

    rows = []

    for cluster_id in cluster_ids:
        feature_mask = feature_cluster_labels == cluster_id

        cluster_act_mean = X_feature_instance[feature_mask].mean(axis=0)
        cluster_act_max = X_feature_instance[feature_mask].max(axis=0)
        cluster_active_count = (
            X_feature_instance[feature_mask] > config.active_threshold
        ).sum(axis=0)

        rows.append(cluster_act_mean)
        rows.append(cluster_act_max)
        rows.append(cluster_active_count)

    X_instance = np.stack(rows, axis=1)

    col_names = []

    for cluster_id in cluster_ids:
        col_names.extend(
            [
                f"cluster_{cluster_id}_mean",
                f"cluster_{cluster_id}_max",
                f"cluster_{cluster_id}_active_count",
            ]
        )

    return X_instance, col_names


def cluster_instances_by_subroute(
    config: SubtypeAnalysisConfig,
    df_target: pd.DataFrame,
    X_instance: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X_instance)

    best_k, score_df = choose_best_k(
        X,
        min_k=2,
        max_k=min(config.max_instance_clusters, len(df_target) - 1),
        method=config.instance_cluster_method,
    )

    safe_save_csv(score_df, config.instance_subtype_k_selection_path)
    print("Best instance subtype k:", best_k)

    labels = _fit_cluster_model(
        X=X,
        k=best_k,
        method=config.instance_cluster_method,
    )

    out = df_target.copy()
    out["instance_subtype_cluster"] = labels.astype(int)

    safe_save_csv(out, config.instance_cluster_path)

    print("\nInstance subtype sizes:")
    print(out["instance_subtype_cluster"].value_counts().sort_index())

    return out, labels


def extract_top_prompts_by_feature_cluster(
    config: SubtypeAnalysisConfig,
    df_target: pd.DataFrame,
    X_feature_instance: np.ndarray,
    feature_cluster_labels: np.ndarray,
) -> pd.DataFrame:
    cluster_ids = sorted(set(feature_cluster_labels.tolist()))
    rows = []

    for cluster_id in cluster_ids:
        feature_mask = feature_cluster_labels == cluster_id
        cluster_score = X_feature_instance[feature_mask].mean(axis=0)
        top_idx = np.argsort(cluster_score)[
            -config.top_prompts_per_cluster:
        ][::-1]

        for rank, local_idx in enumerate(top_idx, start=1):
            row = df_target.iloc[int(local_idx)]

            rows.append(
                {
                    "target_emotion": config.target_emotion,
                    "feature_cluster": int(cluster_id),
                    "rank": int(rank),
                    "cluster_activation_score": float(cluster_score[local_idx]),
                    "original_instance_idx": int(row["original_instance_idx"]),
                    "input_text": row["input_text"],
                    "prompt": row["prompt"],
                }
            )

    top_df = pd.DataFrame(rows)
    safe_save_csv(top_df, config.top_prompts_path)

    return top_df


# ============================================================
# Main subtype discovery pipeline
# ============================================================

def run_subtype_discovery(config: SubtypeAnalysisConfig) -> Dict[str, object]:
    ensure_dir(config.output_dir)

    print("Loading data...")

    df_correct = load_correct_prediction_df(config)
    target_features = load_target_features(config)
    acts = load_activation_array(config)

    df_target, X_feature_instance, feature_keys, feature_labels = (
        build_target_activation_matrix(
            config=config,
            acts=acts,
            df_correct=df_correct,
            target_features=target_features,
        )
    )

    cluster_df, feature_cluster_labels = cluster_features_by_activation(
        config=config,
        X_feature_instance=X_feature_instance,
        feature_labels=feature_labels,
        target_features=target_features,
    )

    X_instance, instance_feature_names = build_instance_cluster_features(
        config=config,
        X_feature_instance=X_feature_instance,
        feature_cluster_labels=feature_cluster_labels,
    )

    df_target_with_subtypes, instance_subtype_labels = cluster_instances_by_subroute(
        config=config,
        df_target=df_target,
        X_instance=X_instance,
    )

    top_prompt_df = extract_top_prompts_by_feature_cluster(
        config=config,
        df_target=df_target,
        X_feature_instance=X_feature_instance,
        feature_cluster_labels=feature_cluster_labels,
    )

    instance_feature_df = pd.DataFrame(
        X_instance,
        columns=instance_feature_names,
    )
    instance_feature_df.insert(
        0,
        "original_instance_idx",
        df_target["original_instance_idx"].astype(int).values,
    )
    instance_feature_df["instance_subtype_cluster"] = instance_subtype_labels

    safe_save_csv(instance_feature_df, config.instance_cluster_feature_matrix_path)

    print("\nSubtype discovery done.")
    print("Outputs saved to:", config.output_dir)

    return {
        "df_correct": df_correct,
        "target_features": target_features,
        "df_target": df_target,
        "X_feature_instance": X_feature_instance,
        "feature_keys": feature_keys,
        "feature_labels": feature_labels,
        "cluster_df": cluster_df,
        "feature_cluster_labels": feature_cluster_labels,
        "X_instance": X_instance,
        "instance_feature_names": instance_feature_names,
        "df_target_with_subtypes": df_target_with_subtypes,
        "instance_subtype_labels": instance_subtype_labels,
        "top_prompt_df": top_prompt_df,
    }


# ============================================================
# Optional causal ablation
# ============================================================

def get_emotion_logits(logits, emotion_to_token: Dict[str, int], emotions: Sequence[str]):
    return torch.stack(
        [logits[:, emotion_to_token[e]] for e in emotions],
        dim=1,
    )


def get_emotion_preds(emotion_logits, emotions: Sequence[str]) -> List[str]:
    pred_idx = torch.argmax(emotion_logits, dim=1)
    return [emotions[i] for i in pred_idx.detach().cpu().tolist()]


def build_emotion_token_map(llm, emotions: Sequence[str]) -> Dict[str, int]:
    emotion_to_token = {}

    tokenizer = llm.tokenizer

    for emotion in emotions:
        token_ids = tokenizer.encode(" " + emotion, add_special_tokens=False)

        if len(token_ids) == 0:
            token_ids = tokenizer.encode(emotion, add_special_tokens=False)

        if len(token_ids) == 0:
            raise ValueError(f"Could not tokenize emotion: {emotion}")

        emotion_to_token[emotion] = int(token_ids[-1])

    print("Emotion token map:", emotion_to_token)

    return emotion_to_token


def run_cluster_ablation_for_layer(
    config: SubtypeAnalysisConfig,
    llm,
    sae,
    df_examples: pd.DataFrame,
    layer_idx: int,
    feature_indices: Sequence[int],
    cluster_id: int,
    instance_subtype_col: str,
    emotion_to_token: Dict[str, int],
) -> pd.DataFrame:
    rows = []

    prompts = df_examples["prompt"].tolist()
    raw_texts = df_examples["input_text"].tolist()
    original_indices = df_examples["original_instance_idx"].tolist()

    if instance_subtype_col in df_examples.columns:
        subtype_labels = df_examples[instance_subtype_col].tolist()
    else:
        subtype_labels = [-1 for _ in range(len(df_examples))]

    for start in tqdm(
        range(0, len(prompts), config.batch_size),
        desc=f"Ablate cluster {cluster_id} L{layer_idx}",
        leave=False,
    ):
        batch_prompts = prompts[start:start + config.batch_size]
        batch_raw_texts = raw_texts[start:start + config.batch_size]
        batch_original_indices = original_indices[start:start + config.batch_size]
        batch_subtypes = subtype_labels[start:start + config.batch_size]

        with torch.no_grad():
            with llm.trace(batch_prompts):
                source_hidden = (
                    llm.model.layers[int(layer_idx)]
                    .output[:, -1, :]
                    .save()
                )
                clean_logits = llm.output.logits[:, -1, :].save()

        source_sae_features = sae.encode(source_hidden)
        source_hidden_decoded = sae.decode(source_sae_features)
        residual = source_hidden - source_hidden_decoded

        patched_sae_features = source_sae_features.clone()
        patched_sae_features[:, [int(f) for f in feature_indices]] = 0.0
        patched_hidden = sae.decode(patched_sae_features) + residual

        with torch.no_grad():
            with llm.trace(batch_prompts):
                llm.model.layers[int(layer_idx)].output[:, -1, :] = patched_hidden
                ablated_logits = llm.output.logits[:, -1, :].save()

        clean_emotion_logits = get_emotion_logits(
            clean_logits,
            emotion_to_token,
            config.emotions,
        )
        ablated_emotion_logits = get_emotion_logits(
            ablated_logits,
            emotion_to_token,
            config.emotions,
        )

        clean_preds = get_emotion_preds(clean_emotion_logits, config.emotions)
        ablated_preds = get_emotion_preds(ablated_emotion_logits, config.emotions)

        clean_np = clean_emotion_logits.detach().float().cpu().numpy()
        ablated_np = ablated_emotion_logits.detach().float().cpu().numpy()

        for i in range(len(batch_prompts)):
            row = {
                "target_emotion": config.target_emotion,
                "feature_cluster": int(cluster_id),
                "instance_subtype_cluster": int(batch_subtypes[i]),
                "layer_idx": int(layer_idx),
                "n_ablated_features": int(len(feature_indices)),
                "ablated_features": ",".join(map(str, feature_indices)),
                "original_instance_idx": int(batch_original_indices[i]),
                "input_text": batch_raw_texts[i],
                "prompt": batch_prompts[i],
                "clean_pred": clean_preds[i],
                "ablated_pred": ablated_preds[i],
                "clean_matches_target": clean_preds[i] == config.target_emotion,
                "prediction_flipped": clean_preds[i] != ablated_preds[i],
                "flipped_away_from_target": (
                    clean_preds[i] == config.target_emotion
                    and ablated_preds[i] != config.target_emotion
                ),
            }

            for j, emotion in enumerate(config.emotions):
                clean_logit = clean_np[i, j]
                ablated_logit = ablated_np[i, j]

                row[f"clean_logit_{emotion}"] = float(clean_logit)
                row[f"ablated_logit_{emotion}"] = float(ablated_logit)
                row[f"logit_drop_{emotion}"] = float(clean_logit - ablated_logit)

            row["target_logit_drop"] = row[f"logit_drop_{config.target_emotion}"]

            rows.append(row)

        del source_hidden
        del clean_logits
        del source_sae_features
        del source_hidden_decoded
        del residual
        del patched_sae_features
        del patched_hidden
        del ablated_logits

        cleanup_memory()

    return pd.DataFrame(rows)


def summarize_cluster_ablation(
    config: SubtypeAnalysisConfig,
    instance_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "feature_cluster",
        "instance_subtype_cluster",
        "layer_idx",
    ]

    for keys, sub in instance_df.groupby(group_cols):
        feature_cluster, instance_subtype, layer_idx = keys
        clean_sub = sub[sub["clean_matches_target"] == True].copy()

        row = {
            "target_emotion": config.target_emotion,
            "feature_cluster": int(feature_cluster),
            "instance_subtype_cluster": int(instance_subtype),
            "layer_idx": int(layer_idx),
            "n_instances": int(len(sub)),
            "n_clean_target_instances": int(len(clean_sub)),
            "n_ablated_features": int(sub["n_ablated_features"].iloc[0]),
            "flip_rate_all": float(sub["prediction_flipped"].mean()),
            "flip_away_from_target_rate_all": float(
                sub["flipped_away_from_target"].mean()
            ),
            "avg_target_logit_drop_all": float(sub["target_logit_drop"].mean()),
        }

        if len(clean_sub) > 0:
            row["flip_rate_clean_target_only"] = float(
                clean_sub["prediction_flipped"].mean()
            )
            row["flip_away_from_target_rate_clean_target_only"] = float(
                clean_sub["flipped_away_from_target"].mean()
            )
            row["avg_target_logit_drop_clean_target_only"] = float(
                clean_sub["target_logit_drop"].mean()
            )
        else:
            row["flip_rate_clean_target_only"] = np.nan
            row["flip_away_from_target_rate_clean_target_only"] = np.nan
            row["avg_target_logit_drop_clean_target_only"] = np.nan

        for emotion in config.emotions:
            row[f"avg_logit_drop_{emotion}"] = float(
                sub[f"logit_drop_{emotion}"].mean()
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["feature_cluster", "instance_subtype_cluster", "layer_idx"]
    ).reset_index(drop=True)

    return summary


def compute_cluster_subtype_specificity(
    config: SubtypeAnalysisConfig,
    summary_df: pd.DataFrame,
    metric: str = "avg_target_logit_drop_clean_target_only",
) -> pd.DataFrame:
    if metric not in summary_df.columns:
        metric = "avg_target_logit_drop_all"

    if metric not in summary_df.columns:
        raise ValueError("No target logit drop metric found in summary_df.")

    matrix = summary_df.pivot_table(
        index="feature_cluster",
        columns="instance_subtype_cluster",
        values=metric,
        aggfunc="mean",
    )

    matrix.to_csv(config.causal_matrix_path)
    print("Saved cluster × subtype matrix:", config.causal_matrix_path)
    print(matrix)

    return matrix


def run_subtype_specific_causal_ablation(
    config: SubtypeAnalysisConfig,
    cluster_df: Optional[pd.DataFrame] = None,
    df_target_with_subtypes: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run optional causal ablation after subtype discovery.

    This function expects:
    - feature cluster file
    - instance subtype file

    If cluster_df or df_target_with_subtypes are not provided, they are loaded from disk.
    """
    if cluster_df is None:
        cluster_df = pd.read_csv(config.feature_cluster_path)

    if df_target_with_subtypes is None:
        df_target_with_subtypes = pd.read_csv(config.instance_cluster_path)

    print("\nLoading model for causal ablation...")

    from sae_emotion.logit_ablation_utils import load_llm, SAECache

    llm = load_llm(
        model_name=config.model_name,
        device=config.device,
        dtype=config.dtype,
    )

    emotion_to_token = build_emotion_token_map(
        llm=llm,
        emotions=config.emotions,
    )

    sae_cache = SAECache(
        model_name=config.model_name,
        device=config.device,
    )

    all_results = []
    grouped = cluster_df.groupby(["feature_cluster", "layer_idx"])

    for (cluster_id, layer_idx), sub in grouped:
        layer_idx = int(layer_idx)
        feature_indices = sorted(sub["feature_idx"].astype(int).tolist())

        if len(feature_indices) == 0:
            continue

        sae = sae_cache.load(layer_idx)

        result_df = run_cluster_ablation_for_layer(
            config=config,
            llm=llm,
            sae=sae,
            df_examples=df_target_with_subtypes,
            layer_idx=layer_idx,
            feature_indices=feature_indices,
            cluster_id=int(cluster_id),
            instance_subtype_col="instance_subtype_cluster",
            emotion_to_token=emotion_to_token,
        )

        all_results.append(result_df)

        partial = pd.concat(all_results, ignore_index=True)
        safe_save_csv(partial, config.cluster_ablation_instance_path)

        summary = summarize_cluster_ablation(config, partial)
        safe_save_csv(summary, config.cluster_ablation_summary_path)

    if len(all_results) == 0:
        raise ValueError("No ablation results were produced.")

    instance_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_cluster_ablation(config, instance_df)

    safe_save_csv(instance_df, config.cluster_ablation_instance_path)
    safe_save_csv(summary_df, config.cluster_ablation_summary_path)

    compute_cluster_subtype_specificity(config, summary_df)

    print("Saved cluster ablation instance file:", config.cluster_ablation_instance_path)
    print("Saved cluster ablation summary file:", config.cluster_ablation_summary_path)

    return instance_df, summary_df


# ============================================================
# Plotting utilities
# ============================================================

def load_subtype_outputs_for_plotting(
    config: SubtypeAnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    feature_clusters = safe_read_csv(
        config.feature_cluster_path,
        "feature_clusters",
        required=True,
    )
    top_prompts = safe_read_csv(
        config.top_prompts_path,
        "top_prompts",
        required=True,
    )
    instance_subtypes = safe_read_csv(
        config.instance_cluster_path,
        "instance_subtypes",
        required=True,
    )
    ablation_summary = safe_read_csv(
        config.cluster_ablation_summary_path,
        "ablation_summary",
        required=False,
    )
    causal_matrix = safe_read_csv(
        config.causal_matrix_path,
        "causal_matrix",
        index_col=0,
        required=False,
    )

    print_file_info("feature_clusters", feature_clusters)
    print_file_info("top_prompts", top_prompts)
    print_file_info("instance_subtypes", instance_subtypes)
    print_file_info("ablation_summary", ablation_summary)
    print_file_info("causal_matrix", causal_matrix)

    return (
        feature_clusters,
        top_prompts,
        instance_subtypes,
        ablation_summary,
        causal_matrix,
    )


def normalize_plotting_dtypes(
    feature_clusters: pd.DataFrame,
    top_prompts: pd.DataFrame,
    instance_subtypes: pd.DataFrame,
    ablation_summary: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    require_columns(
        feature_clusters,
        ["feature_cluster", "layer_idx", "feature_idx"],
        "feature_clusters",
    )
    require_columns(
        top_prompts,
        ["feature_cluster", "rank", "input_text", "original_instance_idx"],
        "top_prompts",
    )
    require_columns(
        instance_subtypes,
        ["instance_subtype_cluster", "original_instance_idx"],
        "instance_subtypes",
    )

    feature_clusters = feature_clusters.copy()
    top_prompts = top_prompts.copy()
    instance_subtypes = instance_subtypes.copy()

    feature_clusters["feature_cluster"] = feature_clusters["feature_cluster"].astype(int)
    feature_clusters["layer_idx"] = feature_clusters["layer_idx"].astype(int)
    feature_clusters["feature_idx"] = feature_clusters["feature_idx"].astype(int)

    top_prompts["feature_cluster"] = top_prompts["feature_cluster"].astype(int)

    instance_subtypes["instance_subtype_cluster"] = instance_subtypes[
        "instance_subtype_cluster"
    ].astype(int)

    if ablation_summary is not None:
        ablation_summary = ablation_summary.copy()
        require_columns(
            ablation_summary,
            ["feature_cluster", "instance_subtype_cluster", "layer_idx"],
            "ablation_summary",
        )
        ablation_summary["feature_cluster"] = ablation_summary[
            "feature_cluster"
        ].astype(int)
        ablation_summary["instance_subtype_cluster"] = ablation_summary[
            "instance_subtype_cluster"
        ].astype(int)
        ablation_summary["layer_idx"] = ablation_summary["layer_idx"].astype(int)

    metric_cols = [
        "mean_activation_on_target",
        "activation_rate_on_target",
        "avg_target_logit_drop_clean_target_only",
        "flip_away_from_target_rate_clean_target_only",
        "avg_target_logit_drop_all",
        "flip_away_from_target_rate_all",
    ]

    feature_clusters = clean_numeric(feature_clusters, metric_cols)
    ablation_summary = clean_numeric(ablation_summary, metric_cols)

    return feature_clusters, top_prompts, instance_subtypes, ablation_summary


def plot_feature_cluster_sizes(
    config: SubtypeAnalysisConfig,
    feature_clusters: pd.DataFrame,
) -> pd.DataFrame:
    agg_dict = {
        "n_features": ("feature_idx", "nunique"),
        "n_layers": ("layer_idx", "nunique"),
    }

    if "mean_activation_on_target" in feature_clusters.columns:
        agg_dict["mean_activation"] = ("mean_activation_on_target", "mean")

    if "activation_rate_on_target" in feature_clusters.columns:
        agg_dict["mean_activation_rate"] = ("activation_rate_on_target", "mean")

    cluster_size = (
        feature_clusters
        .groupby("feature_cluster", as_index=False)
        .agg(**agg_dict)
        .sort_values("feature_cluster")
    )

    out_path = os.path.join(
        config.plot_dir,
        f"{config.target_emotion}_feature_cluster_size_summary.csv",
    )
    safe_save_csv(cluster_size, out_path)

    plt.figure(figsize=(8, 5))
    plt.bar(
        cluster_size["feature_cluster"].astype(str),
        cluster_size["n_features"],
    )
    plt.xlabel("Feature Cluster")
    plt.ylabel("Number of Features")
    plt.title(f"{config.target_emotion}: Feature Cluster Sizes")
    plt.grid(axis="y", alpha=0.3)
    savefig(config, f"{config.target_emotion}_feature_cluster_sizes.png")

    return cluster_size


def plot_feature_clusters_across_layers(
    config: SubtypeAnalysisConfig,
    feature_clusters: pd.DataFrame,
) -> pd.DataFrame:
    layer_cluster_counts = (
        feature_clusters
        .groupby(["layer_idx", "feature_cluster"], as_index=False)
        .agg(n_features=("feature_idx", "nunique"))
    )

    pivot_layer_cluster = (
        layer_cluster_counts
        .pivot(index="layer_idx", columns="feature_cluster", values="n_features")
        .fillna(0)
        .sort_index()
    )

    plt.figure(figsize=(10, 6))

    bottom = np.zeros(len(pivot_layer_cluster))

    for cluster_id in pivot_layer_cluster.columns:
        vals = pivot_layer_cluster[cluster_id].values
        plt.bar(
            pivot_layer_cluster.index.astype(str),
            vals,
            bottom=bottom,
            label=f"cluster {cluster_id}",
        )
        bottom += vals

    plt.xlabel("Layer")
    plt.ylabel("Number of Features")
    plt.title(f"{config.target_emotion}: Feature Clusters Across Layers")
    plt.legend(title="Feature cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    savefig(config, f"{config.target_emotion}_feature_clusters_across_layers.png")

    return pivot_layer_cluster


def plot_cluster_activation_strength_frequency(
    config: SubtypeAnalysisConfig,
    cluster_size: pd.DataFrame,
) -> None:
    if (
        "mean_activation" not in cluster_size.columns
        or "mean_activation_rate" not in cluster_size.columns
    ):
        print(
            "[SKIP] Activation strength/frequency plot: "
            "mean_activation_on_target or activation_rate_on_target missing."
        )
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(cluster_size))
    labels = cluster_size["feature_cluster"].astype(str).tolist()

    ax1.bar(
        x - 0.2,
        cluster_size["mean_activation"],
        width=0.4,
        label="mean activation",
    )

    ax2 = ax1.twinx()

    ax2.bar(
        x + 0.2,
        cluster_size["mean_activation_rate"],
        width=0.4,
        alpha=0.6,
        label="activation rate",
    )

    ax1.set_xlabel("Feature Cluster")
    ax1.set_ylabel("Mean Activation")
    ax2.set_ylabel("Activation Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(
        f"{config.target_emotion}: Cluster Activation Strength and Frequency"
    )
    ax1.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    savefig(
        config,
        f"{config.target_emotion}_cluster_activation_strength_frequency.png",
    )


def plot_instance_subtype_sizes(
    config: SubtypeAnalysisConfig,
    instance_subtypes: pd.DataFrame,
) -> pd.DataFrame:
    subtype_size = (
        instance_subtypes
        .groupby("instance_subtype_cluster", as_index=False)
        .agg(n_instances=("original_instance_idx", "nunique"))
        .sort_values("instance_subtype_cluster")
    )

    out_path = os.path.join(
        config.plot_dir,
        f"{config.target_emotion}_instance_subtype_size_summary.csv",
    )
    safe_save_csv(subtype_size, out_path)

    plt.figure(figsize=(8, 5))
    plt.bar(
        subtype_size["instance_subtype_cluster"].astype(str),
        subtype_size["n_instances"],
    )
    plt.xlabel("Instance Subtype Cluster")
    plt.ylabel("Number of Instances")
    plt.title(f"{config.target_emotion}: Instance Subtype Sizes")
    plt.grid(axis="y", alpha=0.3)
    savefig(config, f"{config.target_emotion}_instance_subtype_sizes.png")

    return subtype_size


def plot_causal_effect_matrix(
    config: SubtypeAnalysisConfig,
    causal_matrix: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if causal_matrix is None:
        print("[SKIP] Causal effect matrix: causal_matrix file not found.")
        return None

    causal_matrix = causal_matrix.apply(pd.to_numeric, errors="coerce")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(causal_matrix.values, aspect="auto")
    plt.colorbar(im, label="Avg Target Logit Drop")

    plt.xticks(
        np.arange(len(causal_matrix.columns)),
        causal_matrix.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        np.arange(len(causal_matrix.index)),
        causal_matrix.index,
    )

    plt.xlabel("Instance Subtype Cluster")
    plt.ylabel("Feature Cluster")
    plt.title(f"{config.target_emotion}: Cluster × Subtype Causal Effect Matrix")

    for i in range(causal_matrix.shape[0]):
        for j in range(causal_matrix.shape[1]):
            val = causal_matrix.values[i, j]
            if not np.isnan(val):
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    savefig(
        config,
        f"{config.target_emotion}_cluster_x_subtype_causal_effect_matrix.png",
    )

    return causal_matrix


def plot_relative_subtype_specificity_matrix(
    config: SubtypeAnalysisConfig,
    causal_matrix: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if causal_matrix is None:
        print("[SKIP] Relative subtype specificity: causal_matrix file not found.")
        return None

    causal_matrix = causal_matrix.apply(pd.to_numeric, errors="coerce")

    row_norm = causal_matrix.div(
        causal_matrix.max(axis=1).replace(0, np.nan),
        axis=0,
    )

    plt.figure(figsize=(8, 6))
    im = plt.imshow(row_norm.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Row-normalized Effect")

    plt.xticks(
        np.arange(len(row_norm.columns)),
        row_norm.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        np.arange(len(row_norm.index)),
        row_norm.index,
    )

    plt.xlabel("Instance Subtype Cluster")
    plt.ylabel("Feature Cluster")
    plt.title(f"{config.target_emotion}: Relative Subtype Specificity")

    for i in range(row_norm.shape[0]):
        for j in range(row_norm.shape[1]):
            val = row_norm.values[i, j]
            if not np.isnan(val):
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    savefig(
        config,
        f"{config.target_emotion}_relative_subtype_specificity_matrix.png",
    )

    return row_norm


def plot_layerwise_causal_effect(
    config: SubtypeAnalysisConfig,
    ablation_summary: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if ablation_summary is None:
        print("[SKIP] Layer-wise causal effect: ablation_summary file not found.")
        return None

    effect_col = "avg_target_logit_drop_clean_target_only"

    if effect_col not in ablation_summary.columns:
        effect_col = "avg_target_logit_drop_all"

    if effect_col not in ablation_summary.columns:
        print("[SKIP] Layer-wise causal effect: no causal effect column found.")
        return None

    layer_effect = (
        ablation_summary
        .groupby(["feature_cluster", "layer_idx"], as_index=False)
        .agg(
            avg_effect=(effect_col, "mean"),
            n_rows=(effect_col, "size"),
        )
    )

    plt.figure(figsize=(10, 6))

    for cluster_id in sorted(layer_effect["feature_cluster"].unique()):
        sub = layer_effect[layer_effect["feature_cluster"] == cluster_id].copy()
        sub = sub.sort_values("layer_idx")

        plt.plot(
            sub["layer_idx"],
            sub["avg_effect"],
            marker="o",
            linewidth=2,
            label=f"cluster {cluster_id}",
        )

    plt.xlabel("Layer")
    plt.ylabel("Average Target Logit Drop")
    plt.title(f"{config.target_emotion}: Layer-wise Causal Effect by Feature Cluster")
    plt.grid(alpha=0.3)
    plt.legend(title="Feature cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(
        config,
        f"{config.target_emotion}_layerwise_causal_effect_by_feature_cluster.png",
    )

    return layer_effect


def plot_flip_away_matrix(
    config: SubtypeAnalysisConfig,
    ablation_summary: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if ablation_summary is None:
        print("[SKIP] Flip-away matrix: ablation_summary file not found.")
        return None

    flip_col = "flip_away_from_target_rate_clean_target_only"

    if flip_col not in ablation_summary.columns:
        flip_col = "flip_away_from_target_rate_all"

    if flip_col not in ablation_summary.columns:
        print("[SKIP] Flip-away matrix: no flip-rate column found.")
        return None

    flip_matrix = ablation_summary.pivot_table(
        index="feature_cluster",
        columns="instance_subtype_cluster",
        values=flip_col,
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 6))
    im = plt.imshow(flip_matrix.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Flip-away Rate")

    plt.xticks(
        np.arange(len(flip_matrix.columns)),
        flip_matrix.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        np.arange(len(flip_matrix.index)),
        flip_matrix.index,
    )

    plt.xlabel("Instance Subtype Cluster")
    plt.ylabel("Feature Cluster")
    plt.title(f"{config.target_emotion}: Flip-away Matrix")

    for i in range(flip_matrix.shape[0]):
        for j in range(flip_matrix.shape[1]):
            val = flip_matrix.values[i, j]
            if not np.isnan(val):
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    savefig(config, f"{config.target_emotion}_cluster_x_subtype_flip_matrix.png")

    return flip_matrix


def save_best_subtype_per_feature_cluster(
    config: SubtypeAnalysisConfig,
    causal_matrix: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if causal_matrix is None:
        print("[SKIP] Best subtype per feature cluster: causal_matrix file not found.")
        return None

    causal_matrix = causal_matrix.apply(pd.to_numeric, errors="coerce")

    best_subtype_rows = []

    for cluster_id in causal_matrix.index:
        row = causal_matrix.loc[cluster_id].dropna()

        if len(row) == 0:
            continue

        best_subtype = row.idxmax()
        best_effect = row.max()
        mean_effect = row.mean()
        specificity_ratio = best_effect / (mean_effect + 1e-8)

        best_subtype_rows.append(
            {
                "feature_cluster": int(cluster_id),
                "best_instance_subtype": int(best_subtype),
                "best_effect": float(best_effect),
                "mean_effect": float(mean_effect),
                "specificity_ratio": float(specificity_ratio),
            }
        )

    best_subtype_df = pd.DataFrame(best_subtype_rows)

    best_subtype_path = os.path.join(
        config.plot_dir,
        f"{config.target_emotion}_best_subtype_per_feature_cluster.csv",
    )
    safe_save_csv(best_subtype_df, best_subtype_path)
    print(best_subtype_df)

    if len(best_subtype_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.bar(
            best_subtype_df["feature_cluster"].astype(str),
            best_subtype_df["specificity_ratio"],
        )
        plt.xlabel("Feature Cluster")
        plt.ylabel("Best Subtype Effect / Mean Effect")
        plt.title(f"{config.target_emotion}: Causal Subtype Specificity Ratio")
        plt.grid(axis="y", alpha=0.3)
        savefig(
            config,
            f"{config.target_emotion}_causal_subtype_specificity_ratio.png",
        )
    else:
        print("[SKIP] Causal subtype specificity ratio: no valid rows.")

    return best_subtype_df


def save_readable_top_prompts(
    config: SubtypeAnalysisConfig,
    top_prompts: pd.DataFrame,
) -> pd.DataFrame:
    readable_rows = []

    for cluster_id in sorted(top_prompts["feature_cluster"].unique()):
        sub = top_prompts[top_prompts["feature_cluster"] == cluster_id].copy()

        if "rank" in sub.columns:
            sub = sub.sort_values("rank")

        sub = sub.head(config.readable_top_prompts_per_cluster)

        for local_rank, (_, row) in enumerate(sub.iterrows(), start=1):
            text = str(row["input_text"])
            text_short = textwrap.shorten(text, width=220, placeholder="...")

            rank = row["rank"] if "rank" in row.index else local_rank

            readable_rows.append(
                {
                    "feature_cluster": int(cluster_id),
                    "rank": int(rank),
                    "cluster_activation_score": row.get(
                        "cluster_activation_score",
                        np.nan,
                    ),
                    "original_instance_idx": int(row["original_instance_idx"]),
                    "input_text_short": text_short,
                    "input_text": text,
                }
            )

    readable_top_prompts = pd.DataFrame(readable_rows)

    readable_path = os.path.join(
        config.plot_dir,
        f"{config.target_emotion}_readable_top20_prompts_by_feature_cluster.csv",
    )
    safe_save_csv(readable_top_prompts, readable_path)

    for cluster_id in sorted(readable_top_prompts["feature_cluster"].unique()):
        print("\n" + "=" * 80)
        print(f"Feature Cluster {cluster_id}: Top Prompts")
        print("=" * 80)

        sub = readable_top_prompts[
            readable_top_prompts["feature_cluster"] == cluster_id
        ].head(10)

        for _, row in sub.iterrows():
            print(f"[{row['rank']}] {row['input_text_short']}")

    return readable_top_prompts


def plot_subtype_analysis(config: SubtypeAnalysisConfig) -> Dict[str, object]:
    ensure_dir(config.plot_dir)

    (
        feature_clusters,
        top_prompts,
        instance_subtypes,
        ablation_summary,
        causal_matrix,
    ) = load_subtype_outputs_for_plotting(config)

    (
        feature_clusters,
        top_prompts,
        instance_subtypes,
        ablation_summary,
    ) = normalize_plotting_dtypes(
        feature_clusters=feature_clusters,
        top_prompts=top_prompts,
        instance_subtypes=instance_subtypes,
        ablation_summary=ablation_summary,
    )

    cluster_size = plot_feature_cluster_sizes(config, feature_clusters)
    pivot_layer_cluster = plot_feature_clusters_across_layers(
        config,
        feature_clusters,
    )
    plot_cluster_activation_strength_frequency(config, cluster_size)
    subtype_size = plot_instance_subtype_sizes(config, instance_subtypes)

    causal_matrix = plot_causal_effect_matrix(config, causal_matrix)
    row_norm = plot_relative_subtype_specificity_matrix(config, causal_matrix)

    layer_effect = plot_layerwise_causal_effect(config, ablation_summary)
    flip_matrix = plot_flip_away_matrix(config, ablation_summary)
    best_subtype_df = save_best_subtype_per_feature_cluster(config, causal_matrix)
    readable_top_prompts = save_readable_top_prompts(config, top_prompts)

    print("\nPlotting done.")
    print("Plots saved to:", config.plot_dir)

    return {
        "cluster_size": cluster_size,
        "pivot_layer_cluster": pivot_layer_cluster,
        "subtype_size": subtype_size,
        "causal_matrix": causal_matrix,
        "row_norm": row_norm,
        "layer_effect": layer_effect,
        "flip_matrix": flip_matrix,
        "best_subtype_df": best_subtype_df,
        "readable_top_prompts": readable_top_prompts,
    }


def build_subtype_config_from_model_name(
    model_name: str,
    target_emotion: str,
    sae_size: Optional[str] = None,
    pred_dir: str = "data",
    act_dir: str = "../data",
    output_root: str = "data",
    run_causal_ablation: bool = False,
    **kwargs,
) -> SubtypeAnalysisConfig:
    """
    Convenience builder when you only want to specify model_name and sae_size.

    sae_size controls both:
    1. feature folder:
       data/{model_slug}-feature-act/
       data/{model_slug}-small-feature-act/
       data/{model_slug}-big-feature-act/

    2. activation file:
       ../data/sae_acts_combined_zeroshot_{model_slug}.npy
       ../data/sae_acts_combined_zeroshot_{model_slug}_small.npy
       ../data/sae_acts_combined_zeroshot_{model_slug}_big.npy
    """
    model_slug = model_name.replace("/", "-")

    if sae_size is None:
        feature_act_dir_name = f"{model_slug}-feature-act"
        act_filename = f"sae_acts_combined_zeroshot_{model_slug}.npy"
    else:
        feature_act_dir_name = f"{model_slug}-{sae_size}-feature-act"
        act_filename = f"sae_acts_combined_zeroshot_{model_slug}_{sae_size}.npy"

    feature_act_dir = os.path.join(output_root, feature_act_dir_name)

    return SubtypeAnalysisConfig(
        model_name=model_name,
        pred_path=os.path.join(
            pred_dir,
            f"combined_zeroshot_preds_{model_slug}.csv",
        ),
        act_path=os.path.join(act_dir, act_filename),
        feature_path=os.path.join(
            feature_act_dir,
            "auto_selected_unique_features_with_affected_emotions.csv",
        ),
        output_dir=os.path.join(
            feature_act_dir,
            "distributed_emotion_subroute_analysis",
        ),
        target_emotion=target_emotion,
        sae_size=sae_size,
        run_causal_ablation=run_causal_ablation,
        **kwargs,
    )


def run_full_subtype_pipeline(config: SubtypeAnalysisConfig) -> Dict[str, object]:
    discovery_outputs = run_subtype_discovery(config)

    ablation_outputs = None

    if config.run_causal_ablation:
        instance_df, summary_df = run_subtype_specific_causal_ablation(
            config=config,
            cluster_df=discovery_outputs["cluster_df"],
            df_target_with_subtypes=discovery_outputs["df_target_with_subtypes"],
        )
        ablation_outputs = {
            "instance_df": instance_df,
            "summary_df": summary_df,
        }

    plot_outputs = plot_subtype_analysis(config)

    return {
        "discovery_outputs": discovery_outputs,
        "ablation_outputs": ablation_outputs,
        "plot_outputs": plot_outputs,
    }
