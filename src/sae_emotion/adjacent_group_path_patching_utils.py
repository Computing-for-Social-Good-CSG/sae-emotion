"""
Adjacent-layer group path-patching utilities for sparse emotion routes.

Core question:
    Do sparse emotion features in layer l pass emotion evidence to sparse emotion
    features in the next selected downstream layer?

Core intervention for each target emotion e and adjacent selected layer pair l -> m:
    1. clean run
    2. ablate all selected emotion graph features X=G_l at layer l
    3. measure whether X ablation changes downstream mediator activations M=G_m
    4. ablate M alone and measure target-emotion logit loss
    5. ablate X + restore M to clean activation
    6. measure target-emotion logit recovery

This gives the three pieces needed for a route claim:
    X -> M: ablate X changes M activation
    M -> Y: ablate M changes target emotion logit Y
    X -> M -> Y: restoring M after ablating X recovers Y

This file intentionally removes:
    - single-node restoration
    - bridge/source/end edge-path selection
    - explicit start/end node path configs

Input graph:
    emotion_feature_edges/all_affected_feature_edges.csv
or:
    emotion_feature_edges/{emotion}_affected_feature_edges.csv

Required graph columns:
    conditioning_emotion, source_layer, source_feature, target_layer, target_feature

Optional graph columns used only for diagnostics:
    mean_effect, mean_abs_effect, normalized_effect
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from sae_emotion.logit_ablation_utils import (
    load_llm,
    SAECache,
    build_emotion_token_map,
    zero_ablate_features,
    cleanup_memory,
    safe_save_csv,
    zero_shot_template,
)

DEFAULT_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


# ============================================================
# Basic helpers
# ============================================================

def resolve_sae_size(sae_size: Optional[str]) -> str:
    """Normalize SAE size for project utilities."""
    if sae_size is None:
        return "big"
    sae_size = str(sae_size).lower().strip()
    if sae_size in {"", "none", "null"}:
        return "big"
    if sae_size not in {"big", "medium", "small"}:
        raise ValueError(
            f"Unsupported sae_size={sae_size!r}. Use 'big', 'medium', 'small', or None."
        )
    return sae_size


def get_emotion_logits(
    logits: torch.Tensor,
    emotion_to_token: Dict[str, int],
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> torch.Tensor:
    return torch.stack([logits[:, emotion_to_token[e]] for e in emotions], dim=1)


def get_preds(
    emotion_logits: torch.Tensor,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> List[str]:
    idx = torch.argmax(emotion_logits, dim=1)
    return [emotions[i] for i in idx.detach().cpu().tolist()]


def encode_with_residual(sae, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = sae.encode(hidden)
    decoded = sae.decode(feats)
    residual = hidden - decoded
    return feats, residual


def patch_features_to_clean(
    sae,
    hidden_after_ablation: torch.Tensor,
    clean_feature_values: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Patch one or more SAE latent features at one layer to their clean per-example values.

    clean_feature_values:
        feature_idx -> tensor with shape [batch]
    """
    feats = sae.encode(hidden_after_ablation)
    decoded = sae.decode(feats)
    residual = hidden_after_ablation - decoded

    patched = feats.clone()
    for feature_idx, clean_values in clean_feature_values.items():
        patched[:, int(feature_idx)] = clean_values.to(
            device=patched.device,
            dtype=patched.dtype,
        )

    return sae.decode(patched) + residual


def build_eval_df(
    pred_path: str,
    true_emotion: Optional[str] = None,
    pred_emotion: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    text_col: str = "input_text",
    true_col: str = "emotion",
    pred_col: str = "pred_emotion",
) -> pd.DataFrame:
    df = pd.read_csv(pred_path)

    required = [text_col, true_col, pred_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prediction file: {missing}")

    df[true_col] = df[true_col].astype(str).str.lower().str.strip()
    df[pred_col] = df[pred_col].astype(str).str.lower().str.strip()

    if true_emotion is not None:
        df = df[df[true_col] == str(true_emotion).lower().strip()].copy()

    if pred_emotion is not None:
        df = df[df[pred_col] == str(pred_emotion).lower().strip()].copy()

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=int(sample_size), random_state=random_state).copy()

    df = df.reset_index(drop=False).rename(columns={"index": "original_instance_idx"})
    df["prompt"] = df[text_col].apply(zero_shot_template)
    return df


# ============================================================
# Select all target-emotion graph nodes by layer
# ============================================================

def load_emotion_graph_edges(edge_path: str) -> pd.DataFrame:
    """Load all affected feature edges and normalize columns."""
    edges = pd.read_csv(edge_path)

    required = [
        "conditioning_emotion",
        "source_layer",
        "source_feature",
        "target_layer",
        "target_feature",
    ]
    missing = [c for c in required if c not in edges.columns]
    if missing:
        raise ValueError(f"Missing columns in edge file: {missing}")

    edges = edges.copy()
    edges["conditioning_emotion"] = (
        edges["conditioning_emotion"].astype(str).str.lower().str.strip()
    )

    for col in ["source_layer", "source_feature", "target_layer", "target_feature"]:
        edges[col] = pd.to_numeric(edges[col], errors="coerce").astype("Int64")

    for col in ["mean_effect", "mean_abs_effect", "normalized_effect"]:
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")
        else:
            edges[col] = np.nan

    if edges["mean_abs_effect"].isna().all():
        edges["mean_abs_effect"] = edges["mean_effect"].abs()

    edges = edges.dropna(
        subset=["conditioning_emotion", "source_layer", "source_feature", "target_layer", "target_feature"]
    ).copy()
    edges["source_layer"] = edges["source_layer"].astype(int)
    edges["source_feature"] = edges["source_feature"].astype(int)
    edges["target_layer"] = edges["target_layer"].astype(int)
    edges["target_feature"] = edges["target_feature"].astype(int)

    return edges


def select_all_nodes_by_emotion_layer(
    edge_path: str,
    output_dir: str,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    min_layer: Optional[int] = None,
    max_layer: Optional[int] = None,
    min_node_score: Optional[float] = None,
    exclude_layers: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Select every feature node that appears in the target-emotion graph.

    A node is included if it appears as either a source node or target node in edges
    whose conditioning_emotion equals the target emotion.

    The selected output has one row per unique:
        target_emotion, layer, feature
    """
    os.makedirs(output_dir, exist_ok=True)
    edges = load_emotion_graph_edges(edge_path)
    emotions = [str(e).lower().strip() for e in emotions]
    exclude_layers_set = set(int(x) for x in (exclude_layers or []))

    all_selected = []

    for emotion in emotions:
        sub = edges[edges["conditioning_emotion"] == emotion].copy()
        if len(sub) == 0:
            print(f"No graph edges found for emotion={emotion}")
            continue

        src = sub[["source_layer", "source_feature", "mean_effect", "mean_abs_effect", "normalized_effect"]].rename(
            columns={
                "source_layer": "layer",
                "source_feature": "feature",
                "mean_effect": "edge_mean_effect",
                "mean_abs_effect": "edge_mean_abs_effect",
                "normalized_effect": "edge_normalized_effect",
            }
        )
        tgt = sub[["target_layer", "target_feature", "mean_effect", "mean_abs_effect", "normalized_effect"]].rename(
            columns={
                "target_layer": "layer",
                "target_feature": "feature",
                "mean_effect": "edge_mean_effect",
                "mean_abs_effect": "edge_mean_abs_effect",
                "normalized_effect": "edge_normalized_effect",
            }
        )

        nodes = pd.concat([src, tgt], ignore_index=True)
        nodes = nodes.dropna(subset=["layer", "feature"]).copy()
        nodes["layer"] = nodes["layer"].astype(int)
        nodes["feature"] = nodes["feature"].astype(int)

        if min_layer is not None:
            nodes = nodes[nodes["layer"] >= int(min_layer)].copy()
        if max_layer is not None:
            nodes = nodes[nodes["layer"] <= int(max_layer)].copy()
        if exclude_layers_set:
            nodes = nodes[~nodes["layer"].isin(exclude_layers_set)].copy()

        selected = (
            nodes.groupby(["layer", "feature"], dropna=False)
            .agg(
                node_score=("edge_mean_abs_effect", "max"),
                node_signed_score=("edge_mean_effect", "max"),
                node_normalized_score=("edge_normalized_effect", "max"),
                n_incident_edges=("feature", "size"),
            )
            .reset_index()
        )
        selected["node_score"] = pd.to_numeric(selected["node_score"], errors="coerce")
        selected["node_signed_score"] = pd.to_numeric(selected["node_signed_score"], errors="coerce")
        selected["node_normalized_score"] = pd.to_numeric(selected["node_normalized_score"], errors="coerce")
        selected["node_score"] = selected["node_score"].fillna(selected["node_signed_score"].abs()).fillna(1.0)

        if min_node_score is not None:
            selected = selected[selected["node_score"] >= float(min_node_score)].copy()

        if len(selected) == 0:
            print(f"No selected nodes after filtering for emotion={emotion}")
            continue

        selected["target_emotion"] = emotion
        selected = selected.sort_values(["layer", "node_score"], ascending=[True, False]).copy()
        selected["rank_within_layer"] = selected.groupby("layer")["node_score"].rank(
            method="first", ascending=False
        ).astype(int)
        selected["global_rank"] = selected["node_score"].rank(method="first", ascending=False).astype(int)

        out_path = os.path.join(output_dir, f"{emotion}_all_nodes_by_layer.csv")
        selected.to_csv(out_path, index=False)

        layer_summary = (
            selected.groupby("layer")
            .agg(n_features=("feature", "nunique"), mean_node_score=("node_score", "mean"))
            .reset_index()
        )
        layer_summary_path = os.path.join(output_dir, f"{emotion}_layer_node_counts.csv")
        layer_summary.to_csv(layer_summary_path, index=False)

        print("\n" + "=" * 70)
        print(f"Selected all graph nodes for {emotion}")
        print("Total nodes:", len(selected))
        print("Layers:", selected["layer"].nunique())
        print("Saved:", out_path)
        print(layer_summary.to_string(index=False))

        all_selected.append(selected)

    all_selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    all_out_path = os.path.join(output_dir, "all_emotion_nodes_by_layer.csv")
    all_selected_df.to_csv(all_out_path, index=False)
    print("\nSaved all selected nodes:", all_out_path)
    print("Total selected nodes:", len(all_selected_df))
    return all_selected_df


def build_adjacent_layer_group_specs(
    selected_nodes_df: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    min_features_per_layer: int = 1,
    max_features_per_layer: Optional[int] = None,
    rank_col: str = "node_score",
) -> Dict[str, List[Dict]]:
    """
    Build adjacent-layer group path-patching specs.

    For each emotion and each adjacent selected layer pair l -> m:
        ablate all selected features in layer l
        restore all selected features in layer m

    If max_features_per_layer is None, all nodes in that layer are used.
    """
    specs: Dict[str, List[Dict]] = {str(e).lower().strip(): [] for e in emotions}

    if selected_nodes_df is None or len(selected_nodes_df) == 0:
        return specs

    df = selected_nodes_df.copy()
    df["target_emotion"] = df["target_emotion"].astype(str).str.lower().str.strip()
    df["layer"] = df["layer"].astype(int)
    df["feature"] = df["feature"].astype(int)

    if rank_col not in df.columns:
        df[rank_col] = 1.0

    for emotion in specs.keys():
        emo_df = df[df["target_emotion"] == emotion].copy()
        if len(emo_df) == 0:
            continue

        layer_to_features = {}
        for layer, layer_df in emo_df.groupby("layer"):
            layer_df = layer_df.sort_values(rank_col, ascending=False).copy()
            features = layer_df["feature"].astype(int).drop_duplicates().tolist()
            if max_features_per_layer is not None:
                features = features[: int(max_features_per_layer)]
            if len(features) >= int(min_features_per_layer):
                layer_to_features[int(layer)] = features

        layers = sorted(layer_to_features.keys())
        for source_layer, restore_layer in zip(layers[:-1], layers[1:]):
            source_features = layer_to_features[source_layer]
            restore_features = layer_to_features[restore_layer]

            specs[emotion].append(
                {
                    "target_emotion": emotion,
                    "source_layer": int(source_layer),
                    "source_features": source_features,
                    "restore_layer": int(restore_layer),
                    "restore_features": restore_features,
                    "source_group_name": f"G_{emotion}_L{source_layer}_all{len(source_features)}",
                    "restore_group_name": f"G_{emotion}_L{restore_layer}_all{len(restore_features)}",
                    "pair_name": f"{emotion}_L{source_layer}_to_L{restore_layer}",
                    "n_source_features": len(source_features),
                    "n_restore_features": len(restore_features),
                }
            )

    return specs


# ============================================================
# Core adjacent-layer group path patching
# ============================================================

def run_one_adjacent_group_path_patching_batch(
    llm,
    sae_cache: SAECache,
    prompts: List[str],
    source_layer: int,
    source_features: Sequence[int],
    restore_layer: int,
    restore_features: Sequence[int],
    target_emotion: str,
    emotion_to_token: Dict[str, int],
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> Dict:
    """
    One batch for adjacent sparse-feature route mediation.

    Let:
        X = source group G_l
        M = downstream group G_m
        Y = target emotion logit

    Conditions:
        clean
        ablate_X
        ablate_M
        ablate_X_restore_M

    Metrics enabled by these conditions:
        X -> M: compare M activation under clean vs ablate_X
        X -> Y: compare Y logit under clean vs ablate_X
        M -> Y: compare Y logit under clean vs ablate_M
        X -> M -> Y: compare Y logit under ablate_X vs ablate_X_restore_M
    """
    emotions = list(emotions)
    source_layer = int(source_layer)
    restore_layer = int(restore_layer)
    source_features = [int(x) for x in source_features]
    restore_features = [int(x) for x in restore_features]
    target_idx = emotions.index(target_emotion)

    if restore_layer <= source_layer:
        raise ValueError(
            f"restore_layer must be downstream of source_layer, got {source_layer}->{restore_layer}"
        )
    if len(source_features) == 0:
        raise ValueError("source_features is empty")
    if len(restore_features) == 0:
        raise ValueError("restore_features is empty")

    layers_to_save = sorted({source_layer, restore_layer})

    # 1. Clean run.
    clean_hiddens = {}
    with torch.no_grad():
        with llm.trace(prompts):
            for layer in layers_to_save:
                clean_hiddens[layer] = llm.model.layers[layer].output[:, -1, :].save()
            clean_logits = llm.output.logits[:, -1, :].save()

    clean_emotion_logits = get_emotion_logits(clean_logits, emotion_to_token, emotions)
    clean_preds = get_preds(clean_emotion_logits, emotions)

    clean_features = {}
    clean_residuals = {}
    for layer in layers_to_save:
        sae = sae_cache.load(layer)
        feats, residual = encode_with_residual(sae, clean_hiddens[layer])
        clean_features[layer] = feats
        clean_residuals[layer] = residual

    source_sae = sae_cache.load(source_layer)
    restore_sae = sae_cache.load(restore_layer)

    # Clean mediator group activations, per instance.
    clean_m_feats = clean_features[restore_layer][:, restore_features]
    clean_m_mean_activation = clean_m_feats.mean(dim=1)
    clean_m_sum_activation = clean_m_feats.sum(dim=1)
    clean_m_active_count = (clean_m_feats > 0).sum(dim=1)

    # 2. Source group ablation only: X -> Y.
    with torch.no_grad():
        with llm.trace(prompts):
            llm.model.layers[source_layer].output[:, -1, :] = zero_ablate_features(
                sae=source_sae,
                source_sae_features=clean_features[source_layer],
                feature_indices=source_features,
                residual=clean_residuals[source_layer],
            )
            ablated_x_logits = llm.output.logits[:, -1, :].save()

    ablated_x_emotion_logits = get_emotion_logits(ablated_x_logits, emotion_to_token, emotions)
    ablated_x_preds = get_preds(ablated_x_emotion_logits, emotions)

    # 3. Mediator group ablation only: M -> Y.
    with torch.no_grad():
        with llm.trace(prompts):
            llm.model.layers[restore_layer].output[:, -1, :] = zero_ablate_features(
                sae=restore_sae,
                source_sae_features=clean_features[restore_layer],
                feature_indices=restore_features,
                residual=clean_residuals[restore_layer],
            )
            ablated_m_logits = llm.output.logits[:, -1, :].save()

    ablated_m_emotion_logits = get_emotion_logits(ablated_m_logits, emotion_to_token, emotions)
    ablated_m_preds = get_preds(ablated_m_emotion_logits, emotions)

    # 4a. Capture downstream hidden after source group ablation: X -> M.
    with torch.no_grad():
        with llm.trace(prompts):
            llm.model.layers[source_layer].output[:, -1, :] = zero_ablate_features(
                sae=source_sae,
                source_sae_features=clean_features[source_layer],
                feature_indices=source_features,
                residual=clean_residuals[source_layer],
            )
            restore_hidden_after_x_ablation = llm.model.layers[restore_layer].output[:, -1, :].save()

    ablated_x_restore_feats, _ = encode_with_residual(restore_sae, restore_hidden_after_x_ablation)
    ablated_x_m_feats = ablated_x_restore_feats[:, restore_features]
    ablated_x_m_mean_activation = ablated_x_m_feats.mean(dim=1)
    ablated_x_m_sum_activation = ablated_x_m_feats.sum(dim=1)
    ablated_x_m_active_count = (ablated_x_m_feats > 0).sum(dim=1)

    clean_feature_values = {
        fid: clean_features[restore_layer][:, fid].detach()
        for fid in restore_features
    }
    patched_restore_hidden = patch_features_to_clean(
        sae=restore_sae,
        hidden_after_ablation=restore_hidden_after_x_ablation,
        clean_feature_values=clean_feature_values,
    ).detach()

    # 4b. Rerun source ablation and inject patched downstream group: mediation rescue.
    with torch.no_grad():
        with llm.trace(prompts):
            llm.model.layers[source_layer].output[:, -1, :] = zero_ablate_features(
                sae=source_sae,
                source_sae_features=clean_features[source_layer],
                feature_indices=source_features,
                residual=clean_residuals[source_layer],
            )
            llm.model.layers[restore_layer].output[:, -1, :] = patched_restore_hidden
            restored_logits = llm.output.logits[:, -1, :].save()

    restored_emotion_logits = get_emotion_logits(restored_logits, emotion_to_token, emotions)
    restored_preds = get_preds(restored_emotion_logits, emotions)

    out = {
        "target_idx": target_idx,
        "clean_logits": clean_emotion_logits.detach().float().cpu().numpy(),
        "ablated_x_logits": ablated_x_emotion_logits.detach().float().cpu().numpy(),
        "ablated_m_logits": ablated_m_emotion_logits.detach().float().cpu().numpy(),
        "restored_logits": restored_emotion_logits.detach().float().cpu().numpy(),
        "clean_preds": clean_preds,
        "ablated_x_preds": ablated_x_preds,
        "ablated_m_preds": ablated_m_preds,
        "restored_preds": restored_preds,
        "clean_m_mean_activation": clean_m_mean_activation.detach().float().cpu().numpy(),
        "ablated_x_m_mean_activation": ablated_x_m_mean_activation.detach().float().cpu().numpy(),
        "clean_m_sum_activation": clean_m_sum_activation.detach().float().cpu().numpy(),
        "ablated_x_m_sum_activation": ablated_x_m_sum_activation.detach().float().cpu().numpy(),
        "clean_m_active_count": clean_m_active_count.detach().float().cpu().numpy(),
        "ablated_x_m_active_count": ablated_x_m_active_count.detach().float().cpu().numpy(),
    }

    del clean_hiddens, clean_logits, ablated_x_logits, ablated_m_logits, restored_logits
    del clean_emotion_logits, ablated_x_emotion_logits, ablated_m_emotion_logits, restored_emotion_logits
    del restore_hidden_after_x_ablation, patched_restore_hidden
    del clean_m_feats, ablated_x_restore_feats, ablated_x_m_feats
    cleanup_memory()
    return out

def summarize_adjacent_group_batch(
    batch_result: Dict,
    batch_df: pd.DataFrame,
    spec: Dict,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    rows = []
    emotions = list(emotions)
    target_emotion = spec["target_emotion"]
    target_idx = emotions.index(target_emotion)

    clean_logits = batch_result["clean_logits"][:, target_idx]
    ablated_x_logits = batch_result["ablated_x_logits"][:, target_idx]
    ablated_m_logits = batch_result["ablated_m_logits"][:, target_idx]
    restored_logits = batch_result["restored_logits"][:, target_idx]

    for i in range(len(batch_df)):
        clean_logit = float(clean_logits[i])
        ablated_x_logit = float(ablated_x_logits[i])
        ablated_m_logit = float(ablated_m_logits[i])
        restored_logit = float(restored_logits[i])

        x_logit_loss = clean_logit - ablated_x_logit
        m_logit_loss = clean_logit - ablated_m_logit
        restored_delta = restored_logit - ablated_x_logit
        recovery = restored_delta / x_logit_loss if abs(x_logit_loss) > 1e-6 else np.nan

        clean_m_mean = float(batch_result["clean_m_mean_activation"][i])
        ablated_x_m_mean = float(batch_result["ablated_x_m_mean_activation"][i])
        clean_m_sum = float(batch_result["clean_m_sum_activation"][i])
        ablated_x_m_sum = float(batch_result["ablated_x_m_sum_activation"][i])
        clean_m_active_count = float(batch_result["clean_m_active_count"][i])
        ablated_x_m_active_count = float(batch_result["ablated_x_m_active_count"][i])

        m_mean_drop_after_x_ablation = clean_m_mean - ablated_x_m_mean
        m_sum_drop_after_x_ablation = clean_m_sum - ablated_x_m_sum
        m_relative_mean_drop_after_x_ablation = (
            m_mean_drop_after_x_ablation / abs(clean_m_mean)
            if abs(clean_m_mean) > 1e-6
            else np.nan
        )
        m_active_count_drop_after_x_ablation = clean_m_active_count - ablated_x_m_active_count

        x_to_m_positive = m_mean_drop_after_x_ablation > 0
        x_to_y_positive = x_logit_loss > 0
        m_to_y_positive = m_logit_loss > 0
        mediation_positive = restored_delta > 0
        supports_route_instance = (
            x_to_m_positive and x_to_y_positive and m_to_y_positive and mediation_positive
        )

        row = {
            "original_instance_idx": int(batch_df.iloc[i]["original_instance_idx"]),
            "text": batch_df.iloc[i].get("input_text", ""),
            "true_emotion": batch_df.iloc[i].get("emotion", ""),
            "pred_emotion": batch_df.iloc[i].get("pred_emotion", ""),
            "target_emotion": target_emotion,
            "pair_name": spec["pair_name"],
            "source_layer": int(spec["source_layer"]),
            "restore_layer": int(spec["restore_layer"]),
            "source_group_name": spec["source_group_name"],
            "restore_group_name": spec["restore_group_name"],
            "source_features": ",".join(map(str, spec["source_features"])),
            "restore_features": ",".join(map(str, spec["restore_features"])),
            "n_source_features": int(spec["n_source_features"]),
            "n_restore_features": int(spec["n_restore_features"]),

            # Predictions.
            "clean_pred": batch_result["clean_preds"][i],
            "ablated_x_pred": batch_result["ablated_x_preds"][i],
            "ablated_m_pred": batch_result["ablated_m_preds"][i],
            "restored_pred": batch_result["restored_preds"][i],

            # Logit metrics.
            "clean_target_logit": clean_logit,
            "ablated_x_target_logit": ablated_x_logit,
            "ablated_m_target_logit": ablated_m_logit,
            "restored_target_logit": restored_logit,
            "x_to_y_logit_loss": float(x_logit_loss),
            "m_to_y_logit_loss": float(m_logit_loss),
            "mediation_restored_delta": float(restored_delta),
            "mediation_recovery_ratio": float(recovery) if np.isfinite(recovery) else np.nan,

            # X -> M activation metrics.
            "clean_m_mean_activation": clean_m_mean,
            "ablated_x_m_mean_activation": ablated_x_m_mean,
            "m_mean_drop_after_x_ablation": float(m_mean_drop_after_x_ablation),
            "m_relative_mean_drop_after_x_ablation": (
                float(m_relative_mean_drop_after_x_ablation)
                if np.isfinite(m_relative_mean_drop_after_x_ablation)
                else np.nan
            ),
            "clean_m_sum_activation": clean_m_sum,
            "ablated_x_m_sum_activation": ablated_x_m_sum,
            "m_sum_drop_after_x_ablation": float(m_sum_drop_after_x_ablation),
            "clean_m_active_count": clean_m_active_count,
            "ablated_x_m_active_count": ablated_x_m_active_count,
            "m_active_count_drop_after_x_ablation": float(m_active_count_drop_after_x_ablation),

            # Boolean route evidence.
            "x_ablation_decreases_m_activation": bool(x_to_m_positive),
            "x_ablation_lowers_logit": bool(x_to_y_positive),
            "m_ablation_lowers_logit": bool(m_to_y_positive),
            "restore_increases_logit_after_x_ablation": bool(mediation_positive),
            "supports_route_instance": bool(supports_route_instance),

            # Flip metrics.
            "clean_to_x_ablate_flip": batch_result["clean_preds"][i] != batch_result["ablated_x_preds"][i],
            "clean_to_m_ablate_flip": batch_result["clean_preds"][i] != batch_result["ablated_m_preds"][i],
            "x_ablate_to_restore_flip": batch_result["ablated_x_preds"][i] != batch_result["restored_preds"][i],
            "restored_to_target": batch_result["restored_preds"][i] == target_emotion,
        }

        # Backward-compatible aliases for older plotting code.
        row["ablated_pred"] = row["ablated_x_pred"]
        row["ablated_target_logit"] = row["ablated_x_target_logit"]
        row["target_logit_loss"] = row["x_to_y_logit_loss"]
        row["target_logit_restored_delta"] = row["mediation_restored_delta"]
        row["target_logit_recovery"] = row["mediation_recovery_ratio"]
        row["source_ablation_lowers_logit"] = row["x_ablation_lowers_logit"]
        row["restore_increases_logit"] = row["restore_increases_logit_after_x_ablation"]
        row["positive_layer_mediation"] = bool(x_to_y_positive and mediation_positive)
        row["clean_to_ablate_flip"] = row["clean_to_x_ablate_flip"]
        row["ablate_to_restore_flip"] = row["x_ablate_to_restore_flip"]

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_adjacent_group_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # Backward compatibility for older partial result files.
    rename_if_missing = {
        "ablated_x_target_logit": "ablated_target_logit",
        "x_to_y_logit_loss": "target_logit_loss",
        "mediation_restored_delta": "target_logit_restored_delta",
        "mediation_recovery_ratio": "target_logit_recovery",
        "ablated_x_pred": "ablated_pred",
        "clean_to_x_ablate_flip": "clean_to_ablate_flip",
        "x_ablate_to_restore_flip": "ablate_to_restore_flip",
    }
    for new_col, old_col in rename_if_missing.items():
        if new_col not in df.columns and old_col in df.columns:
            df[new_col] = df[old_col]

    numeric_cols = [
        "source_layer",
        "restore_layer",
        "n_source_features",
        "n_restore_features",
        "clean_target_logit",
        "ablated_x_target_logit",
        "ablated_m_target_logit",
        "restored_target_logit",
        "x_to_y_logit_loss",
        "m_to_y_logit_loss",
        "mediation_restored_delta",
        "mediation_recovery_ratio",
        "clean_m_mean_activation",
        "ablated_x_m_mean_activation",
        "m_mean_drop_after_x_ablation",
        "m_relative_mean_drop_after_x_ablation",
        "clean_m_sum_activation",
        "ablated_x_m_sum_activation",
        "m_sum_drop_after_x_ablation",
        "clean_m_active_count",
        "ablated_x_m_active_count",
        "m_active_count_drop_after_x_ablation",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Recompute core metrics where possible.
    if {"clean_target_logit", "ablated_x_target_logit"}.issubset(df.columns):
        df["x_to_y_logit_loss"] = df["clean_target_logit"] - df["ablated_x_target_logit"]
    if {"clean_target_logit", "ablated_m_target_logit"}.issubset(df.columns):
        df["m_to_y_logit_loss"] = df["clean_target_logit"] - df["ablated_m_target_logit"]
    if {"restored_target_logit", "ablated_x_target_logit"}.issubset(df.columns):
        df["mediation_restored_delta"] = df["restored_target_logit"] - df["ablated_x_target_logit"]
    if {"mediation_restored_delta", "x_to_y_logit_loss"}.issubset(df.columns):
        df["mediation_recovery_ratio_recomputed"] = np.where(
            df["x_to_y_logit_loss"].abs() > 1e-6,
            df["mediation_restored_delta"] / df["x_to_y_logit_loss"],
            np.nan,
        )
    else:
        df["mediation_recovery_ratio_recomputed"] = np.nan

    if {"clean_m_mean_activation", "ablated_x_m_mean_activation"}.issubset(df.columns):
        df["m_mean_drop_after_x_ablation"] = (
            df["clean_m_mean_activation"] - df["ablated_x_m_mean_activation"]
        )
        df["m_relative_mean_drop_after_x_ablation"] = np.where(
            df["clean_m_mean_activation"].abs() > 1e-6,
            df["m_mean_drop_after_x_ablation"] / df["clean_m_mean_activation"].abs(),
            np.nan,
        )
    if {"clean_m_sum_activation", "ablated_x_m_sum_activation"}.issubset(df.columns):
        df["m_sum_drop_after_x_ablation"] = (
            df["clean_m_sum_activation"] - df["ablated_x_m_sum_activation"]
        )
    if {"clean_m_active_count", "ablated_x_m_active_count"}.issubset(df.columns):
        df["m_active_count_drop_after_x_ablation"] = (
            df["clean_m_active_count"] - df["ablated_x_m_active_count"]
        )

    df["x_ablation_decreases_m_activation"] = df.get("m_mean_drop_after_x_ablation", np.nan) > 0
    df["x_ablation_lowers_logit"] = df.get("x_to_y_logit_loss", np.nan) > 0
    df["m_ablation_lowers_logit"] = df.get("m_to_y_logit_loss", np.nan) > 0
    df["restore_increases_logit_after_x_ablation"] = df.get("mediation_restored_delta", np.nan) > 0
    df["supports_route_instance"] = (
        df["x_ablation_decreases_m_activation"]
        & df["x_ablation_lowers_logit"]
        & df["m_ablation_lowers_logit"]
        & df["restore_increases_logit_after_x_ablation"]
    )

    if {"clean_pred", "ablated_x_pred"}.issubset(df.columns):
        df["clean_to_x_ablate_flip"] = df["clean_pred"] != df["ablated_x_pred"]
    if {"clean_pred", "ablated_m_pred"}.issubset(df.columns):
        df["clean_to_m_ablate_flip"] = df["clean_pred"] != df["ablated_m_pred"]
    if {"ablated_x_pred", "restored_pred"}.issubset(df.columns):
        df["x_ablate_to_restore_flip"] = df["ablated_x_pred"] != df["restored_pred"]
    if {"restored_pred", "target_emotion"}.issubset(df.columns):
        df["restored_to_target"] = df["restored_pred"] == df["target_emotion"]

    # Backward-compatible aliases.
    df["target_logit_loss"] = df["x_to_y_logit_loss"]
    df["target_logit_restored_delta"] = df["mediation_restored_delta"]
    df["target_logit_recovery_recomputed"] = df["mediation_recovery_ratio_recomputed"]
    df["source_ablation_lowers_logit"] = df["x_ablation_lowers_logit"]
    df["restore_increases_logit"] = df["restore_increases_logit_after_x_ablation"]
    df["positive_layer_mediation"] = df["x_ablation_lowers_logit"] & df["restore_increases_logit_after_x_ablation"]
    if "clean_to_x_ablate_flip" in df.columns:
        df["clean_to_ablate_flip"] = df["clean_to_x_ablate_flip"]
    if "x_ablate_to_restore_flip" in df.columns:
        df["ablate_to_restore_flip"] = df["x_ablate_to_restore_flip"]

    group_cols = [
        "target_emotion",
        "pair_name",
        "source_layer",
        "restore_layer",
        "source_group_name",
        "restore_group_name",
        "source_features",
        "restore_features",
        "n_source_features",
        "n_restore_features",
    ]

    agg_spec = dict(
        n_instances=("original_instance_idx", "nunique"),

        # X -> M.
        mean_clean_m_mean_activation=("clean_m_mean_activation", "mean"),
        mean_ablated_x_m_mean_activation=("ablated_x_m_mean_activation", "mean"),
        mean_m_mean_drop_after_x_ablation=("m_mean_drop_after_x_ablation", "mean"),
        median_m_mean_drop_after_x_ablation=("m_mean_drop_after_x_ablation", "median"),
        mean_m_relative_mean_drop_after_x_ablation=("m_relative_mean_drop_after_x_ablation", "mean"),
        mean_m_sum_drop_after_x_ablation=("m_sum_drop_after_x_ablation", "mean"),
        mean_m_active_count_drop_after_x_ablation=("m_active_count_drop_after_x_ablation", "mean"),
        rate_x_ablation_decreases_m_activation=("x_ablation_decreases_m_activation", "mean"),

        # X -> Y.
        mean_clean_target_logit=("clean_target_logit", "mean"),
        mean_ablated_x_target_logit=("ablated_x_target_logit", "mean"),
        mean_x_to_y_logit_loss=("x_to_y_logit_loss", "mean"),
        median_x_to_y_logit_loss=("x_to_y_logit_loss", "median"),
        rate_x_ablation_lowers_logit=("x_ablation_lowers_logit", "mean"),

        # M -> Y.
        mean_ablated_m_target_logit=("ablated_m_target_logit", "mean"),
        mean_m_to_y_logit_loss=("m_to_y_logit_loss", "mean"),
        median_m_to_y_logit_loss=("m_to_y_logit_loss", "median"),
        rate_m_ablation_lowers_logit=("m_ablation_lowers_logit", "mean"),

        # X -> M -> Y rescue.
        mean_restored_target_logit=("restored_target_logit", "mean"),
        mean_mediation_restored_delta=("mediation_restored_delta", "mean"),
        median_mediation_restored_delta=("mediation_restored_delta", "median"),
        mean_mediation_recovery_ratio=("mediation_recovery_ratio_recomputed", "mean"),
        median_mediation_recovery_ratio=("mediation_recovery_ratio_recomputed", "median"),
        rate_restore_increases_logit_after_x_ablation=("restore_increases_logit_after_x_ablation", "mean"),

        # Combined route evidence.
        rate_supports_route_instance=("supports_route_instance", "mean"),

        # Flip metrics.
        flip_rate_clean_to_x_ablate=("clean_to_x_ablate_flip", "mean"),
        flip_rate_clean_to_m_ablate=("clean_to_m_ablate_flip", "mean"),
        flip_rate_x_ablate_to_restore=("x_ablate_to_restore_flip", "mean"),
        restore_to_target_rate=("restored_to_target", "mean"),
    )

    # Remove aggregations whose input columns are missing for resumed old files.
    existing_agg_spec = {
        out_col: spec_tuple
        for out_col, spec_tuple in agg_spec.items()
        if spec_tuple[0] in df.columns
    }

    summary = df.groupby(group_cols, dropna=False).agg(**existing_agg_spec).reset_index()

    if {"mean_mediation_restored_delta", "mean_x_to_y_logit_loss"}.issubset(summary.columns):
        summary["mean_restored_delta_over_x_loss"] = (
            summary["mean_mediation_restored_delta"]
            / (summary["mean_x_to_y_logit_loss"].abs() + 1e-8)
        )
    else:
        summary["mean_restored_delta_over_x_loss"] = np.nan

    summary["supports_route_pair"] = (
        (summary.get("mean_m_mean_drop_after_x_ablation", np.nan) > 0)
        & (summary.get("mean_x_to_y_logit_loss", np.nan) > 0)
        & (summary.get("mean_m_to_y_logit_loss", np.nan) > 0)
        & (summary.get("mean_mediation_restored_delta", np.nan) > 0)
    )

    # Backward-compatible aliases used by existing plotting code.
    alias_pairs = {
        "mean_target_logit_loss": "mean_x_to_y_logit_loss",
        "median_target_logit_loss": "median_x_to_y_logit_loss",
        "mean_restored_delta": "mean_mediation_restored_delta",
        "median_restored_delta": "median_mediation_restored_delta",
        "mean_recovery_ratio": "mean_mediation_recovery_ratio",
        "median_recovery_ratio": "median_mediation_recovery_ratio",
        "rate_source_ablation_lowers_logit": "rate_x_ablation_lowers_logit",
        "rate_restore_increases_logit": "rate_restore_increases_logit_after_x_ablation",
        "rate_supports_route_instance": "rate_supports_route_instance",
        "flip_rate_clean_to_ablate": "flip_rate_clean_to_x_ablate",
        "flip_rate_ablate_to_restore": "flip_rate_x_ablate_to_restore",
        "mean_restored_delta_over_loss": "mean_restored_delta_over_x_loss",
    }
    for old_name, new_name in alias_pairs.items():
        if new_name in summary.columns:
            summary[old_name] = summary[new_name]

    summary = summary.sort_values(
        ["target_emotion", "source_layer", "restore_layer"]
    ).reset_index(drop=True)
    return summary

def summarize_adjacent_group_emotion_level(pair_summary: pd.DataFrame) -> pd.DataFrame:
    if pair_summary is None or len(pair_summary) == 0:
        return pd.DataFrame()

    df = pair_summary.copy()
    emotion_summary = (
        df.groupby("target_emotion", dropna=False)
        .agg(
            n_layer_pairs=("pair_name", "nunique"),
            total_source_features_across_pairs=("n_source_features", "sum"),
            total_restore_features_across_pairs=("n_restore_features", "sum"),

            # X -> M.
            mean_x_to_m_activation_drop=("mean_m_mean_drop_after_x_ablation", "mean"),
            mean_x_to_m_relative_activation_drop=("mean_m_relative_mean_drop_after_x_ablation", "mean"),
            mean_rate_x_ablation_decreases_m_activation=("rate_x_ablation_decreases_m_activation", "mean"),

            # X -> Y.
            mean_x_to_y_logit_loss=("mean_x_to_y_logit_loss", "mean"),
            mean_rate_x_ablation_lowers_logit=("rate_x_ablation_lowers_logit", "mean"),

            # M -> Y.
            mean_m_to_y_logit_loss=("mean_m_to_y_logit_loss", "mean"),
            mean_rate_m_ablation_lowers_logit=("rate_m_ablation_lowers_logit", "mean"),

            # Rescue.
            mean_mediation_restored_delta=("mean_mediation_restored_delta", "mean"),
            mean_mediation_recovery_ratio=("mean_mediation_recovery_ratio", "mean"),
            max_mediation_recovery_ratio=("mean_mediation_recovery_ratio", "max"),
            min_mediation_recovery_ratio=("mean_mediation_recovery_ratio", "min"),
            mean_rate_restore_increases_logit_after_x_ablation=("rate_restore_increases_logit_after_x_ablation", "mean"),

            # Combined route support.
            mean_rate_supports_route_instance=("rate_supports_route_instance", "mean"),
            n_supported_layer_pairs=("supports_route_pair", "sum"),

            # Flips.
            mean_clean_to_x_ablate_flip=("flip_rate_clean_to_x_ablate", "mean"),
            mean_clean_to_m_ablate_flip=("flip_rate_clean_to_m_ablate", "mean"),
            mean_x_ablate_to_restore_flip=("flip_rate_x_ablate_to_restore", "mean"),
        )
        .reset_index()
    )
    emotion_summary["supported_pair_rate"] = (
        emotion_summary["n_supported_layer_pairs"]
        / emotion_summary["n_layer_pairs"].replace(0, np.nan)
    )

    # Backward-compatible aliases.
    emotion_summary["mean_logit_loss"] = emotion_summary["mean_x_to_y_logit_loss"]
    emotion_summary["mean_restored_delta"] = emotion_summary["mean_mediation_restored_delta"]
    emotion_summary["mean_recovery_ratio"] = emotion_summary["mean_mediation_recovery_ratio"]
    emotion_summary["mean_positive_layer_mediation"] = emotion_summary["mean_rate_supports_route_instance"]
    emotion_summary["mean_clean_to_ablate_flip"] = emotion_summary["mean_clean_to_x_ablate_flip"]
    emotion_summary["mean_ablate_to_restore_flip"] = emotion_summary["mean_x_ablate_to_restore_flip"]
    emotion_summary["max_recovery_ratio"] = emotion_summary["max_mediation_recovery_ratio"]
    emotion_summary["min_recovery_ratio"] = emotion_summary["min_mediation_recovery_ratio"]
    return emotion_summary

def run_adjacent_group_path_patching_for_spec(
    model_name: str,
    pred_path: str,
    spec: Dict,
    output_path: str,
    sample_size: Optional[int] = 200,
    random_state: int = 42,
    batch_size: int = 16,
    sae_size: Optional[str] = "big",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    resume: bool = True,
    output_summary_path: Optional[str] = None,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_summary_path is None:
        output_summary_path = output_path.replace(".csv", "_summary.csv")

    target_emotion = str(spec["target_emotion"]).lower().strip()

    df_eval = build_eval_df(
        pred_path=pred_path,
        true_emotion=target_emotion,
        pred_emotion=target_emotion,
        sample_size=sample_size,
        random_state=random_state,
    )
    print("Eval examples:", len(df_eval))

    llm = load_llm(model_name=model_name, device=device, dtype=dtype, cache_dir=cache_dir)
    emotion_to_token = build_emotion_token_map(llm, list(emotions))
    sae_cache = SAECache(model_name=model_name, device=device, sae_size=resolve_sae_size(sae_size))

    all_rows = []
    done_indices = set()

    if resume and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) > 0:
            all_rows.append(existing)
            done_indices = set(existing["original_instance_idx"].astype(int).tolist())
            print("Resuming existing rows:", len(existing))
            print("Completed instances:", len(done_indices))

    df_todo = df_eval[
        ~df_eval["original_instance_idx"].astype(int).isin(done_indices)
    ].copy()

    for start in tqdm(range(0, len(df_todo), batch_size), desc=spec["pair_name"]):
        batch_df = df_todo.iloc[start:start + batch_size].copy()
        prompts = batch_df["prompt"].tolist()

        batch_result = run_one_adjacent_group_path_patching_batch(
            llm=llm,
            sae_cache=sae_cache,
            prompts=prompts,
            source_layer=spec["source_layer"],
            source_features=spec["source_features"],
            restore_layer=spec["restore_layer"],
            restore_features=spec["restore_features"],
            target_emotion=target_emotion,
            emotion_to_token=emotion_to_token,
            emotions=emotions,
        )

        batch_summary = summarize_adjacent_group_batch(
            batch_result=batch_result,
            batch_df=batch_df,
            spec=spec,
            emotions=emotions,
        )

        all_rows.append(batch_summary)
        out_df = pd.concat(all_rows, ignore_index=True)
        safe_save_csv(out_df, output_path)

        summary_df = summarize_adjacent_group_results(out_df)
        safe_save_csv(summary_df, output_summary_path)
        cleanup_memory()

    final_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    safe_save_csv(final_df, output_path)

    summary_df = summarize_adjacent_group_results(final_df)
    safe_save_csv(summary_df, output_summary_path)

    print("Saved instance-level:", output_path)
    print("Saved summary:", output_summary_path)
    return final_df


# ============================================================
# Full pipeline
# ============================================================

def run_adjacent_group_path_patching_pipeline(
    model_name: str,
    pred_path: str,
    edge_path: str,
    output_root: str,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    min_layer: Optional[int] = None,
    max_layer: Optional[int] = None,
    min_node_score: Optional[float] = None,
    exclude_layers: Optional[Sequence[int]] = None,
    min_features_per_layer: int = 1,
    max_features_per_layer: Optional[int] = None,
    sample_size: Optional[int] = 200,
    random_state: int = 42,
    batch_size: int = 16,
    sae_size: Optional[str] = "big",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, object]:
    """
    One-call adjacent-layer group path patching.

    It does:
        1. Read all target-emotion graph nodes from edge_path.
        2. Group nodes by emotion and layer.
        3. For each adjacent selected layer pair l -> m:
            ablate all nodes in X=G_l
            measure downstream mediator activation change in M=G_m
            ablate M alone and measure target-logit loss
            restore M after ablating X and measure target-logit recovery.

    This tests the three pieces of layer-to-layer route evidence:
        X -> M, M -> logit, and X -> M -> logit rescue.
    """
    os.makedirs(output_root, exist_ok=True)
    sae_size = resolve_sae_size(sae_size)

    selected_dir = os.path.join(output_root, "selected_all_nodes_by_layer")
    pair_out_dir = os.path.join(output_root, "adjacent_group_path_patching")
    plot_out_dir = os.path.join(output_root, "plots")
    os.makedirs(pair_out_dir, exist_ok=True)
    os.makedirs(plot_out_dir, exist_ok=True)

    selected_nodes_df = select_all_nodes_by_emotion_layer(
        edge_path=edge_path,
        output_dir=selected_dir,
        emotions=emotions,
        min_layer=min_layer,
        max_layer=max_layer,
        min_node_score=min_node_score,
        exclude_layers=exclude_layers,
    )

    specs = build_adjacent_layer_group_specs(
        selected_nodes_df=selected_nodes_df,
        emotions=emotions,
        min_features_per_layer=min_features_per_layer,
        max_features_per_layer=max_features_per_layer,
        rank_col="node_score",
    )

    specs_rows = []
    for emotion, emotion_specs in specs.items():
        for spec in emotion_specs:
            row = dict(spec)
            row["source_features"] = ",".join(map(str, spec["source_features"]))
            row["restore_features"] = ",".join(map(str, spec["restore_features"]))
            specs_rows.append(row)
    specs_df = pd.DataFrame(specs_rows)
    specs_path = os.path.join(output_root, "adjacent_group_specs.csv")
    specs_df.to_csv(specs_path, index=False)
    print("Saved specs:", specs_path)

    all_results = []
    for emotion, emotion_specs in specs.items():
        for spec in emotion_specs:
            output_path = os.path.join(pair_out_dir, f"{spec['pair_name']}.csv")
            print("\n" + "=" * 70)
            print(f"Running adjacent group path patching: {spec['pair_name']}")
            print(f"Ablate layer {spec['source_layer']}: {spec['n_source_features']} features")
            print(f"Restore layer {spec['restore_layer']}: {spec['n_restore_features']} features")
            print("=" * 70)

            df = run_adjacent_group_path_patching_for_spec(
                model_name=model_name,
                pred_path=pred_path,
                spec=spec,
                output_path=output_path,
                sample_size=sample_size,
                random_state=random_state,
                batch_size=batch_size,
                sae_size=sae_size,
                device=device,
                dtype=dtype,
                cache_dir=cache_dir,
                emotions=emotions,
                resume=resume,
            )
            all_results.append(df)

    all_instances = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    all_instances_path = os.path.join(pair_out_dir, "all_adjacent_group_instances.csv")
    safe_save_csv(all_instances, all_instances_path)

    pair_summary = summarize_adjacent_group_results(all_instances)
    pair_summary_path = os.path.join(pair_out_dir, "adjacent_group_pair_summary.csv")
    safe_save_csv(pair_summary, pair_summary_path)

    emotion_summary = summarize_adjacent_group_emotion_level(pair_summary)
    emotion_summary_path = os.path.join(pair_out_dir, "adjacent_group_emotion_summary.csv")
    safe_save_csv(emotion_summary, emotion_summary_path)

    print("Saved combined instances:", all_instances_path)
    print("Saved pair summary:", pair_summary_path)
    print("Saved emotion summary:", emotion_summary_path)

    if len(pair_summary) > 0:
        plot_adjacent_group_path_patching(
            pair_summary=pair_summary,
            output_dir=plot_out_dir,
            emotion_order=emotions,
            prefix="adjacent_group",
        )

    return {
        "selected_nodes_df": selected_nodes_df,
        "specs": specs,
        "specs_df": specs_df,
        "all_instances": all_instances,
        "pair_summary": pair_summary,
        "emotion_summary": emotion_summary,
        "selected_nodes_path": os.path.join(selected_dir, "all_emotion_nodes_by_layer.csv"),
        "specs_path": specs_path,
        "all_instances_path": all_instances_path,
        "pair_summary_path": pair_summary_path,
        "emotion_summary_path": emotion_summary_path,
    }


# ============================================================
# Plotting
# ============================================================

def plot_adjacent_group_path_patching(
    pair_summary: pd.DataFrame,
    output_dir: str,
    emotion_order: Sequence[str] = DEFAULT_EMOTIONS,
    prefix: str = "adjacent_group",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    df = pair_summary.copy()

    numeric_cols = [
        "source_layer",
        "restore_layer",
        "mean_restored_delta",
        "mean_recovery_ratio",
        "mean_restored_delta_over_loss",
        "rate_supports_route_instance",
        "rate_restore_increases_logit",
        "n_source_features",
        "n_restore_features",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target_emotion"] = pd.Categorical(
        df["target_emotion"], categories=list(emotion_order), ordered=True
    )
    df = df.sort_values(["target_emotion", "source_layer", "restore_layer"])

    # Recovery curve by adjacent layer pair.
    plt.figure(figsize=(10, 6))
    for emotion in emotion_order:
        sub = df[df["target_emotion"].astype(str) == str(emotion)].copy()
        if len(sub) == 0:
            continue
        x = sub["source_layer"].astype(str) + "→" + sub["restore_layer"].astype(str)
        plt.plot(range(len(sub)), sub["mean_recovery_ratio"], marker="o", label=str(emotion))
    plt.axhline(0.0, linestyle="--")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Adjacent selected layer pair index")
    plt.ylabel("Mean recovery ratio")
    plt.title("Adjacent-Layer Sparse Feature Route Recovery")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_recovery_curve.png")
    plt.savefig(out, dpi=300)
    plt.show()
    print("Saved:", out)

    # Heatmap-style matrix as a scatter plot, one point per pair.
    plt.figure(figsize=(10, 6))
    y_labels = [str(e) for e in emotion_order]
    y_pos = {e: i for i, e in enumerate(y_labels)}
    for _, r in df.iterrows():
        emotion = str(r["target_emotion"])
        if emotion not in y_pos:
            continue
        plt.scatter(
            float(r["source_layer"]),
            y_pos[emotion],
            s=max(20.0, 180.0 * float(max(r.get("rate_supports_route_instance", 0.0), 0.0))),
        )
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel("Ablated source layer")
    plt.ylabel("Target emotion")
    plt.title("Route Support Rate by Adjacent Layer Pair")
    plt.tight_layout()
    out = os.path.join(output_dir, f"{prefix}_positive_mediation_scatter.png")
    plt.savefig(out, dpi=300)
    plt.show()
    print("Saved:", out)

    # Save compact plot table.
    preferred_cols = [
        "target_emotion",
        "pair_name",
        "source_layer",
        "restore_layer",
        "n_source_features",
        "n_restore_features",
        "mean_m_mean_drop_after_x_ablation",
        "rate_x_ablation_decreases_m_activation",
        "mean_x_to_y_logit_loss",
        "mean_m_to_y_logit_loss",
        "mean_mediation_restored_delta",
        "mean_mediation_recovery_ratio",
        "rate_supports_route_instance",
        "supports_route_pair",
        "flip_rate_clean_to_x_ablate",
        "flip_rate_clean_to_m_ablate",
        "flip_rate_x_ablate_to_restore",
    ]
    plot_table = df[[c for c in preferred_cols if c in df.columns]].copy()
    out_csv = os.path.join(output_dir, f"{prefix}_plot_table.csv")
    plot_table.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return plot_table
