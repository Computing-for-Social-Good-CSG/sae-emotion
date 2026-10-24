"""
Utilities for SAE-based activation ablation and activation patching/swap experiments.

This file combines two workflows:
1. Per-layer, per-emotion feature ablation.
2. Cross-emotion, layer-local feature activation patching/swap.

Expected project dependencies:
- logit_ablation_utils.py with load_llm, build_emotion_token_map, SAECache,
  zero_ablate_features, cleanup_memory, safe_save_csv, zero_shot_template.
- nnsight-compatible model object returned by load_llm.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sae_emotion.logit_ablation_utils import (
    load_llm,
    build_emotion_token_map,
    SAECache,
    zero_ablate_features,
    cleanup_memory,
    safe_save_csv,
    zero_shot_template,
)


ACTIVATION_PATCHING_UTILS_VERSION = "ablation-plus-swap-sae-size-2026-05-17-v2"

DEFAULT_EMOTIONS = ["joy", "surprise", "disgust", "anger", "sadness", "fear"]



def build_sae_cache(
    model_name: str,
    device: str = "cuda",
    sae_size: Optional[str] = None,
    sae_cache_kwargs: Optional[Dict] = None,
):
    """
    Build an SAECache while forwarding SAE-size information when supported.

    This is needed for Gemma-3 SAE ids such as:
        resid_post_all/layer_21_width_16k_l0_big
        resid_post_all/layer_21_width_16k_l0_small

    Notes:
        - If your local logit_ablation_utils.SAECache already accepts `sae_size`,
          this function passes it directly.
        - If it does not, this function falls back to constructing SAECache and
          attaching `sae_size` as an attribute. In that case, SAECache.load must
          read `self.sae_size` internally. If your SAECache.load is hardcoded to
          `l0_big`, you still need to update logit_ablation_utils.py accordingly.
    """
    if sae_cache_kwargs is None:
        sae_cache_kwargs = {}

    if sae_size is not None:
        try:
            return SAECache(
                model_name=model_name,
                device=device,
                sae_size=sae_size,
                **sae_cache_kwargs,
            )
        except TypeError:
            pass

    cache = SAECache(
        model_name=model_name,
        device=device,
        **sae_cache_kwargs,
    )

    if sae_size is not None:
        # Useful if SAECache.load uses self.sae_size but __init__ has not been updated.
        cache.sae_size = sae_size
        cache.gemma3_sae_size = sae_size
        cache.l0_size = sae_size

    return cache


# ============================================================
# Shared helpers
# ============================================================


def get_emotion_logits(
    logits: torch.Tensor,
    emotion_to_token: Dict[str, int],
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> torch.Tensor:
    """Extract the logits corresponding to emotion label tokens."""
    return torch.stack(
        [logits[:, emotion_to_token[e]] for e in emotions],
        dim=1,
    )


def get_emotion_preds(
    emotion_logits: torch.Tensor,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> List[str]:
    """Convert emotion logits into string predictions."""
    pred_idx = torch.argmax(emotion_logits, dim=1)
    return [emotions[i] for i in pred_idx.detach().cpu().tolist()]


def feature_affects_emotion(affected_emotions: object, emotion: str) -> bool:
    """Return True if an affected_emotions string contains the target emotion."""
    parts = [x.strip() for x in str(affected_emotions).split(",")]
    return emotion in parts


def build_correct_prediction_df(
    pred_path: str,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    """Load prediction CSV and keep only correctly predicted examples."""
    df = pd.read_csv(pred_path)

    required_cols = ["input_text", "emotion", "pred_emotion"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    emotions = list(emotions)
    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
    df["pred_emotion"] = df["pred_emotion"].astype(str).str.lower().str.strip()

    df = df[df["emotion"].isin(emotions)].copy()
    df = df[df["pred_emotion"].isin(emotions)].copy()

    df_correct = df[df["emotion"] == df["pred_emotion"]].copy()
    df_correct = df_correct.reset_index(drop=False).rename(
        columns={"index": "original_instance_idx"}
    )

    df_correct["prompt"] = df_correct["input_text"].apply(zero_shot_template)

    return df_correct


def load_selected_features(feature_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load feature CSV and deduplicate by layer_idx, feature_idx."""
    df = pd.read_csv(feature_path)

    required_cols = ["layer_idx", "feature_idx", "affected_emotions"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature file missing columns: {missing}")

    df["layer_idx"] = df["layer_idx"].astype(int)
    df["feature_idx"] = df["feature_idx"].astype(int)
    df["affected_emotions"] = df["affected_emotions"].fillna("").astype(str)

    df = df.drop_duplicates(subset=["layer_idx", "feature_idx"]).copy()

    if verbose:
        print("Loaded selected features:", df.shape)
        print("Features per layer:")
        print(df.groupby("layer_idx")["feature_idx"].nunique())

    return df


def get_emotion_features_at_layer(
    feature_df: pd.DataFrame,
    emotion: str,
    layer_idx: int,
) -> List[int]:
    """Return selected feature indices for one emotion at one layer."""
    sub = feature_df[
        (feature_df["layer_idx"] == int(layer_idx))
        & (
            feature_df["affected_emotions"].apply(
                lambda x: feature_affects_emotion(x, emotion)
            )
        )
    ].copy()

    return sorted(sub["feature_idx"].astype(int).tolist())


def _safe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# Part 1: Per-layer, per-emotion ablation
# ============================================================


def run_one_layer_one_emotion_ablation(
    llm,
    sae,
    df_correct_emotion: pd.DataFrame,
    target_emotion: str,
    layer_idx: int,
    feature_indices: Sequence[int],
    emotion_to_token: Dict[str, int],
    batch_size: int = 8,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    """Zero-ablate selected SAE features for one target emotion and one layer."""
    rows = []
    emotions = list(emotions)
    feature_indices = sorted([int(x) for x in feature_indices])

    prompts = df_correct_emotion["prompt"].tolist()
    raw_texts = df_correct_emotion["input_text"].tolist()
    original_indices = df_correct_emotion["original_instance_idx"].tolist()

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"ablate {target_emotion} L{layer_idx} n={len(feature_indices)}",
        leave=False,
    ):
        batch_prompts = prompts[start:start + batch_size]
        batch_raw_texts = raw_texts[start:start + batch_size]
        batch_original_indices = original_indices[start:start + batch_size]

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

        with torch.no_grad():
            with llm.trace(batch_prompts):
                llm.model.layers[int(layer_idx)].output[:, -1, :] = zero_ablate_features(
                    sae=sae,
                    source_sae_features=source_sae_features,
                    feature_indices=feature_indices,
                    residual=residual,
                )
                ablated_logits = llm.output.logits[:, -1, :].save()

        clean_emotion_logits = get_emotion_logits(clean_logits, emotion_to_token, emotions)
        ablated_emotion_logits = get_emotion_logits(ablated_logits, emotion_to_token, emotions)

        clean_preds = get_emotion_preds(clean_emotion_logits, emotions)
        ablated_preds = get_emotion_preds(ablated_emotion_logits, emotions)

        clean_np = clean_emotion_logits.detach().float().cpu().numpy()
        ablated_np = ablated_emotion_logits.detach().float().cpu().numpy()

        for i in range(len(batch_prompts)):
            row = {
                "original_instance_idx": int(batch_original_indices[i]),
                "text": batch_raw_texts[i],
                "prompt": batch_prompts[i],
                "target_emotion": target_emotion,
                "true_emotion": target_emotion,
                "layer_idx": int(layer_idx),
                "n_ablated_features": int(len(feature_indices)),
                "ablated_features": ",".join(map(str, feature_indices)),
                "clean_pred": clean_preds[i],
                "ablated_pred": ablated_preds[i],
                "prediction_flipped": clean_preds[i] != ablated_preds[i],
                "flipped_away_from_true": (
                    clean_preds[i] == target_emotion
                    and ablated_preds[i] != target_emotion
                ),
            }

            for j, e in enumerate(emotions):
                clean_logit = clean_np[i, j]
                ablated_logit = ablated_np[i, j]

                row[f"clean_logit_{e}"] = float(clean_logit)
                row[f"ablated_logit_{e}"] = float(ablated_logit)
                row[f"logit_drop_{e}"] = float(clean_logit - ablated_logit)

            row["target_logit_drop"] = row[f"logit_drop_{target_emotion}"]
            rows.append(row)

        del source_hidden, clean_logits, source_sae_features
        del source_hidden_decoded, residual, ablated_logits
        cleanup_memory()

    return pd.DataFrame(rows)


def summarize_ablation_results(
    instance_df: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    """Summarize per-instance ablation results."""
    rows = []
    emotions = list(emotions)

    if len(instance_df) == 0:
        return pd.DataFrame()

    for (target_emotion, layer_idx), sub in instance_df.groupby(
        ["target_emotion", "layer_idx"]
    ):
        row = {
            "target_emotion": target_emotion,
            "layer_idx": int(layer_idx),
            "n_instances": int(len(sub)),
            "n_ablated_features": int(sub["n_ablated_features"].iloc[0]),
            "prediction_flip_rate": float(sub["prediction_flipped"].mean()),
            "flip_away_from_true_rate": float(sub["flipped_away_from_true"].mean()),
            "avg_target_logit_drop": float(sub["target_logit_drop"].mean()),
            "std_target_logit_drop": float(sub["target_logit_drop"].std()),
        }

        for e in emotions:
            row[f"avg_logit_drop_{e}"] = float(sub[f"logit_drop_{e}"].mean())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["target_emotion", "layer_idx"]
    ).reset_index(drop=True)


def run_per_layer_per_emotion_ablation_pipeline(
    model_name: str,
    pred_path: str,
    feature_path: str,
    output_dir: str,
    output_instance_level: Optional[str] = None,
    output_summary: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 32,
    resume: bool = True,
    sae_size: Optional[str] = None,
    sae_cache_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full per-layer per-emotion ablation workflow."""
    os.makedirs(output_dir, exist_ok=True)

    if output_instance_level is None:
        output_instance_level = os.path.join(
            output_dir,
            "per_layer_per_emotion_ablation_instance_level.csv",
        )
    if output_summary is None:
        output_summary = os.path.join(
            output_dir,
            "per_layer_per_emotion_ablation_summary.csv",
        )

    emotions = list(emotions)

    print("Loading selected features...")
    feature_df = load_selected_features(feature_path)

    print("\nLoading correctly predicted instances...")
    df_correct = build_correct_prediction_df(pred_path, emotions)
    print("Correct instances:")
    print(df_correct["emotion"].value_counts())

    print("\nLoading model...")
    llm = load_llm(model_name=model_name, device=device, dtype=dtype)

    print("\nBuilding emotion token map...")
    emotion_to_token = build_emotion_token_map(llm=llm, emotions=emotions)

    sae_cache = build_sae_cache(
        model_name=model_name,
        device=device,
        sae_size=sae_size,
        sae_cache_kwargs=sae_cache_kwargs,
    )

    all_results: List[pd.DataFrame] = []
    done_keys = set()

    if resume and os.path.exists(output_instance_level):
        print("\nResuming from existing ablation instance-level file...")
        existing_instance_df = pd.read_csv(output_instance_level)
        all_results.append(existing_instance_df)
        done_keys = set(
            zip(
                existing_instance_df["target_emotion"].astype(str),
                existing_instance_df["layer_idx"].astype(int),
            )
        )
        print("Completed combinations:", len(done_keys))

    for target_emotion in emotions:
        print("\n===================================================")
        print(f"Target emotion: {target_emotion}")
        print("===================================================")

        emo_feature_df = feature_df[
            feature_df["affected_emotions"].apply(
                lambda x: feature_affects_emotion(x, target_emotion)
            )
        ].copy()

        if len(emo_feature_df) == 0:
            print(f"No features found for {target_emotion}. Skipping.")
            continue

        df_correct_emotion = df_correct[df_correct["emotion"] == target_emotion].copy()
        if len(df_correct_emotion) == 0:
            print(f"No correct examples for {target_emotion}. Skipping.")
            continue

        print(f"Correct examples for {target_emotion}: {len(df_correct_emotion)}")
        print("Features by layer:")
        print(emo_feature_df.groupby("layer_idx")["feature_idx"].nunique())

        for layer_idx, layer_feature_df in emo_feature_df.groupby("layer_idx"):
            key = (target_emotion, int(layer_idx))
            if key in done_keys:
                print(f"Skipping completed: {key}")
                continue

            feature_indices = sorted(layer_feature_df["feature_idx"].astype(int).tolist())
            sae = sae_cache.load(int(layer_idx))

            layer_result_df = run_one_layer_one_emotion_ablation(
                llm=llm,
                sae=sae,
                df_correct_emotion=df_correct_emotion,
                target_emotion=target_emotion,
                layer_idx=int(layer_idx),
                feature_indices=feature_indices,
                emotion_to_token=emotion_to_token,
                batch_size=batch_size,
                emotions=emotions,
            )

            all_results.append(layer_result_df)
            done_keys.add(key)

            partial_instance_df = _safe_concat(all_results)
            partial_summary_df = summarize_ablation_results(partial_instance_df, emotions)

            safe_save_csv(partial_instance_df, output_instance_level)
            safe_save_csv(partial_summary_df, output_summary)

            print(
                f"Saved {target_emotion} L{layer_idx}: "
                f"{len(feature_indices)} features"
            )

    instance_df = _safe_concat(all_results)
    summary_df = summarize_ablation_results(instance_df, emotions)

    safe_save_csv(instance_df, output_instance_level)
    safe_save_csv(summary_df, output_summary)

    print("\nDone ablation pipeline.")
    print("Saved instance-level file:", output_instance_level)
    print("Saved summary file:", output_summary)

    return instance_df, summary_df


# ============================================================
# Part 2: Cross-emotion layer-local activation patching/swap
# ============================================================


def patch_feature_values(
    sae,
    source_hidden: torch.Tensor,
    feature_to_value: Dict[int, float],
) -> Tuple[torch.Tensor, Dict[int, np.ndarray]]:
    """Patch selected SAE feature activations to fixed scalar values."""
    source_sae_features = sae.encode(source_hidden)
    source_hidden_decoded = sae.decode(source_sae_features)
    residual = source_hidden - source_hidden_decoded

    patched_sae_features = source_sae_features.clone()
    clean_values: Dict[int, np.ndarray] = {}

    for feature_idx, patch_value in feature_to_value.items():
        feature_idx = int(feature_idx)
        clean_values[feature_idx] = (
            source_sae_features[:, feature_idx]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        patched_sae_features[:, feature_idx] = float(patch_value)

    patched_hidden = sae.decode(patched_sae_features) + residual
    return patched_hidden, clean_values


def compute_feature_means_by_group_emotion(
    llm,
    sae_cache: SAECache,
    df_correct: pd.DataFrame,
    feature_df: pd.DataFrame,
    batch_size: int = 8,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    """Compute mean activation for selected features in each correct emotion group."""
    rows = []
    emotions = list(emotions)

    for layer_idx, layer_feature_df in feature_df.groupby("layer_idx"):
        layer_idx = int(layer_idx)
        feature_indices = sorted(layer_feature_df["feature_idx"].astype(int).tolist())
        sae = sae_cache.load(layer_idx)

        for group_emotion in emotions:
            df_group = df_correct[df_correct["emotion"] == group_emotion].copy()
            prompts = df_group["prompt"].tolist()

            if len(prompts) == 0:
                continue

            all_acts = []

            for start in tqdm(
                range(0, len(prompts), batch_size),
                desc=f"Mean acts: {group_emotion} L{layer_idx}",
                leave=False,
            ):
                batch_prompts = prompts[start:start + batch_size]

                with torch.no_grad():
                    with llm.trace(batch_prompts):
                        source_hidden = (
                            llm.model.layers[layer_idx]
                            .output[:, -1, :]
                            .save()
                        )

                source_sae_features = sae.encode(source_hidden)

                batch_acts = (
                    source_sae_features[:, feature_indices]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                all_acts.append(batch_acts)

                del source_hidden, source_sae_features
                cleanup_memory()

            all_acts = np.concatenate(all_acts, axis=0)
            mean_acts = all_acts.mean(axis=0)
            std_acts = all_acts.std(axis=0)

            for feature_idx, mean_act, std_act in zip(feature_indices, mean_acts, std_acts):
                rows.append(
                    {
                        "group_emotion": group_emotion,
                        "layer_idx": layer_idx,
                        "feature_idx": int(feature_idx),
                        "mean_activation": float(mean_act),
                        "std_activation": float(std_act),
                        "n_group_examples": int(all_acts.shape[0]),
                    }
                )

    return pd.DataFrame(rows)


def build_feature_to_target_profile(
    feature_mean_df: pd.DataFrame,
    layer_idx: int,
    feature_indices: Sequence[int],
    target_emotion: str,
) -> Dict[int, float]:
    """Map feature_idx -> target-emotion mean activation for a layer."""
    feature_to_value: Dict[int, float] = {}

    for feature_idx in feature_indices:
        matched = feature_mean_df[
            (feature_mean_df["group_emotion"] == target_emotion)
            & (feature_mean_df["layer_idx"] == int(layer_idx))
            & (feature_mean_df["feature_idx"] == int(feature_idx))
        ]

        if len(matched) == 0:
            continue

        feature_to_value[int(feature_idx)] = float(matched["mean_activation"].iloc[0])

    return feature_to_value


def run_one_layer_feature_swap(
    llm,
    sae,
    df_source: pd.DataFrame,
    source_emotion: str,
    target_emotion: str,
    layer_idx: int,
    source_feature_indices: Sequence[int],
    target_feature_indices: Sequence[int],
    feature_to_value: Dict[int, float],
    emotion_to_token: Dict[str, int],
    batch_size: int = 8,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    """Patch source examples toward target-emotion feature activation profile."""
    rows = []
    emotions = list(emotions)
    source_feature_indices = sorted([int(x) for x in source_feature_indices])
    target_feature_indices = sorted([int(x) for x in target_feature_indices])

    prompts = df_source["prompt"].tolist()
    raw_texts = df_source["input_text"].tolist()
    original_indices = df_source["original_instance_idx"].tolist()

    patched_feature_indices = sorted(feature_to_value.keys())

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"swap {source_emotion}->{target_emotion} L{layer_idx}",
        leave=False,
    ):
        batch_prompts = prompts[start:start + batch_size]
        batch_raw_texts = raw_texts[start:start + batch_size]
        batch_original_indices = original_indices[start:start + batch_size]

        with torch.no_grad():
            with llm.trace(batch_prompts):
                source_hidden = (
                    llm.model.layers[int(layer_idx)]
                    .output[:, -1, :]
                    .save()
                )
                clean_logits = llm.output.logits[:, -1, :].save()

        patched_hidden, clean_values = patch_feature_values(
            sae=sae,
            source_hidden=source_hidden,
            feature_to_value=feature_to_value,
        )

        with torch.no_grad():
            with llm.trace(batch_prompts):
                llm.model.layers[int(layer_idx)].output[:, -1, :] = patched_hidden
                patched_logits = llm.output.logits[:, -1, :].save()

        clean_emotion_logits = get_emotion_logits(clean_logits, emotion_to_token, emotions)
        patched_emotion_logits = get_emotion_logits(patched_logits, emotion_to_token, emotions)

        clean_preds = get_emotion_preds(clean_emotion_logits, emotions)
        patched_preds = get_emotion_preds(patched_emotion_logits, emotions)

        clean_np = clean_emotion_logits.detach().float().cpu().numpy()
        patched_np = patched_emotion_logits.detach().float().cpu().numpy()

        source_clean_means = []
        source_patch_means = []
        target_clean_means = []
        target_patch_means = []

        for feature_idx in source_feature_indices:
            if feature_idx in clean_values:
                source_clean_means.append(clean_values[feature_idx])
                source_patch_means.append(
                    np.full_like(clean_values[feature_idx], feature_to_value[feature_idx])
                )

        for feature_idx in target_feature_indices:
            if feature_idx in clean_values:
                target_clean_means.append(clean_values[feature_idx])
                target_patch_means.append(
                    np.full_like(clean_values[feature_idx], feature_to_value[feature_idx])
                )

        if len(source_clean_means) > 0:
            source_clean_mean = np.stack(source_clean_means, axis=1).mean(axis=1)
            source_patch_mean = np.stack(source_patch_means, axis=1).mean(axis=1)
        else:
            source_clean_mean = np.full(len(batch_prompts), np.nan)
            source_patch_mean = np.full(len(batch_prompts), np.nan)

        if len(target_clean_means) > 0:
            target_clean_mean = np.stack(target_clean_means, axis=1).mean(axis=1)
            target_patch_mean = np.stack(target_patch_means, axis=1).mean(axis=1)
        else:
            target_clean_mean = np.full(len(batch_prompts), np.nan)
            target_patch_mean = np.full(len(batch_prompts), np.nan)

        for i in range(len(batch_prompts)):
            row = {
                "original_instance_idx": int(batch_original_indices[i]),
                "text": batch_raw_texts[i],
                "prompt": batch_prompts[i],
                "source_emotion": source_emotion,
                "target_emotion": target_emotion,
                "layer_idx": int(layer_idx),
                "n_source_features": int(len(source_feature_indices)),
                "n_target_features": int(len(target_feature_indices)),
                "n_patched_features": int(len(patched_feature_indices)),
                "source_features": ",".join(map(str, source_feature_indices)),
                "target_features": ",".join(map(str, target_feature_indices)),
                "patched_features": ",".join(map(str, patched_feature_indices)),
                "clean_source_feature_mean_activation": float(source_clean_mean[i]),
                "patched_source_feature_mean_activation": float(source_patch_mean[i]),
                "source_feature_mean_delta": float(source_patch_mean[i] - source_clean_mean[i]),
                "clean_target_feature_mean_activation": float(target_clean_mean[i]),
                "patched_target_feature_mean_activation": float(target_patch_mean[i]),
                "target_feature_mean_delta": float(target_patch_mean[i] - target_clean_mean[i]),
                "clean_pred": clean_preds[i],
                "patched_pred": patched_preds[i],
                "clean_matches_source": clean_preds[i] == source_emotion,
                "prediction_flipped": clean_preds[i] != patched_preds[i],
                "flipped_away_from_source": (
                    clean_preds[i] == source_emotion
                    and patched_preds[i] != source_emotion
                ),
                "flipped_to_target": patched_preds[i] == target_emotion,
            }

            for j, e in enumerate(emotions):
                clean_logit = clean_np[i, j]
                patched_logit = patched_np[i, j]

                row[f"clean_logit_{e}"] = float(clean_logit)
                row[f"patched_logit_{e}"] = float(patched_logit)
                row[f"logit_increase_{e}"] = float(patched_logit - clean_logit)

            row["source_logit_change"] = row[f"logit_increase_{source_emotion}"]
            row["target_logit_change"] = row[f"logit_increase_{target_emotion}"]
            rows.append(row)

        del source_hidden, clean_logits, patched_hidden, patched_logits
        cleanup_memory()

    return pd.DataFrame(rows)


def summarize_swap_results(
    instance_df: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> pd.DataFrame:
    """Summarize cross-emotion activation patching/swap results."""
    rows = []
    emotions = list(emotions)

    if len(instance_df) == 0:
        return pd.DataFrame()

    grouped = instance_df.groupby(["source_emotion", "target_emotion", "layer_idx"])

    for (source_emotion, target_emotion, layer_idx), sub in grouped:
        clean_source_sub = sub[sub["clean_matches_source"] == True].copy()

        row = {
            "source_emotion": source_emotion,
            "target_emotion": target_emotion,
            "layer_idx": int(layer_idx),
            "n_instances": int(len(sub)),
            "n_clean_source_instances": int(len(clean_source_sub)),
            "n_source_features": int(sub["n_source_features"].iloc[0]),
            "n_target_features": int(sub["n_target_features"].iloc[0]),
            "n_patched_features": int(sub["n_patched_features"].iloc[0]),
            "prediction_flip_rate_all": float(sub["prediction_flipped"].mean()),
            "flip_away_from_source_rate_all": float(sub["flipped_away_from_source"].mean()),
            "flip_to_target_rate_all": float(sub["flipped_to_target"].mean()),
            "avg_source_logit_change_all": float(sub["source_logit_change"].mean()),
            "avg_target_logit_change_all": float(sub["target_logit_change"].mean()),
            "avg_source_feature_delta_all": float(sub["source_feature_mean_delta"].mean()),
            "avg_target_feature_delta_all": float(sub["target_feature_mean_delta"].mean()),
        }

        if len(clean_source_sub) > 0:
            row["prediction_flip_rate_clean_source_only"] = float(
                clean_source_sub["prediction_flipped"].mean()
            )
            row["flip_away_from_source_rate_clean_source_only"] = float(
                clean_source_sub["flipped_away_from_source"].mean()
            )
            row["flip_to_target_rate_clean_source_only"] = float(
                clean_source_sub["flipped_to_target"].mean()
            )
            row["avg_source_logit_change_clean_source_only"] = float(
                clean_source_sub["source_logit_change"].mean()
            )
            row["avg_target_logit_change_clean_source_only"] = float(
                clean_source_sub["target_logit_change"].mean()
            )
            row["avg_source_feature_delta_clean_source_only"] = float(
                clean_source_sub["source_feature_mean_delta"].mean()
            )
            row["avg_target_feature_delta_clean_source_only"] = float(
                clean_source_sub["target_feature_mean_delta"].mean()
            )
        else:
            row["prediction_flip_rate_clean_source_only"] = np.nan
            row["flip_away_from_source_rate_clean_source_only"] = np.nan
            row["flip_to_target_rate_clean_source_only"] = np.nan
            row["avg_source_logit_change_clean_source_only"] = np.nan
            row["avg_target_logit_change_clean_source_only"] = np.nan
            row["avg_source_feature_delta_clean_source_only"] = np.nan
            row["avg_target_feature_delta_clean_source_only"] = np.nan

        for e in emotions:
            row[f"avg_logit_increase_{e}"] = float(sub[f"logit_increase_{e}"].mean())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["source_emotion", "target_emotion", "layer_idx"]
    ).reset_index(drop=True)


def run_cross_emotion_layer_feature_swap_pipeline(
    model_name: str,
    pred_path: str,
    feature_path: str,
    output_dir: str,
    output_feature_means: Optional[str] = None,
    output_instance_level: Optional[str] = None,
    output_summary: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 64,
    resume: bool = True,
    sae_size: Optional[str] = None,
    sae_cache_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full cross-emotion layer-local feature swap workflow."""
    os.makedirs(output_dir, exist_ok=True)

    if output_feature_means is None:
        output_feature_means = os.path.join(
            output_dir,
            "feature_mean_activations_by_group_emotion_layer.csv",
        )
    if output_instance_level is None:
        output_instance_level = os.path.join(
            output_dir,
            "cross_emotion_layer_feature_swap_instance_level.csv",
        )
    if output_summary is None:
        output_summary = os.path.join(
            output_dir,
            "cross_emotion_layer_feature_swap_summary.csv",
        )

    emotions = list(emotions)

    print("Loading selected features...")
    feature_df = load_selected_features(feature_path)

    print("\nLoading correctly predicted instances...")
    df_correct = build_correct_prediction_df(pred_path, emotions)
    print("Correct instances:")
    print(df_correct["emotion"].value_counts())

    print("\nLoading model...")
    llm = load_llm(model_name=model_name, device=device, dtype=dtype)

    print("\nBuilding emotion token map...")
    emotion_to_token = build_emotion_token_map(llm=llm, emotions=emotions)

    sae_cache = build_sae_cache(
        model_name=model_name,
        device=device,
        sae_size=sae_size,
        sae_cache_kwargs=sae_cache_kwargs,
    )

    if resume and os.path.exists(output_feature_means):
        print("\nLoading existing feature means...")
        feature_mean_df = pd.read_csv(output_feature_means)
    else:
        print("\nComputing feature means by group emotion...")
        feature_mean_df = compute_feature_means_by_group_emotion(
            llm=llm,
            sae_cache=sae_cache,
            df_correct=df_correct,
            feature_df=feature_df,
            batch_size=batch_size,
            emotions=emotions,
        )
        safe_save_csv(feature_mean_df, output_feature_means)

    print("Feature mean table:", feature_mean_df.shape)

    all_results: List[pd.DataFrame] = []
    done_keys = set()

    if resume and os.path.exists(output_instance_level):
        print("\nResuming from existing swap instance-level file...")
        existing_instance_df = pd.read_csv(output_instance_level)
        all_results.append(existing_instance_df)
        done_keys = set(
            zip(
                existing_instance_df["source_emotion"].astype(str),
                existing_instance_df["target_emotion"].astype(str),
                existing_instance_df["layer_idx"].astype(int),
            )
        )
        print("Completed combinations:", len(done_keys))

    all_layers = sorted(feature_df["layer_idx"].unique().tolist())

    for source_emotion in emotions:
        df_source = df_correct[df_correct["emotion"] == source_emotion].copy()

        if len(df_source) == 0:
            continue

        for target_emotion in emotions:
            if target_emotion == source_emotion:
                continue

            for layer_idx in all_layers:
                layer_idx = int(layer_idx)
                key = (source_emotion, target_emotion, layer_idx)

                if key in done_keys:
                    print(f"Skipping completed: {key}")
                    continue

                source_feature_indices = get_emotion_features_at_layer(
                    feature_df=feature_df,
                    emotion=source_emotion,
                    layer_idx=layer_idx,
                )
                target_feature_indices = get_emotion_features_at_layer(
                    feature_df=feature_df,
                    emotion=target_emotion,
                    layer_idx=layer_idx,
                )

                if len(source_feature_indices) == 0 and len(target_feature_indices) == 0:
                    continue

                patched_feature_indices = sorted(
                    set(source_feature_indices) | set(target_feature_indices)
                )

                feature_to_value = build_feature_to_target_profile(
                    feature_mean_df=feature_mean_df,
                    layer_idx=layer_idx,
                    feature_indices=patched_feature_indices,
                    target_emotion=target_emotion,
                )

                if len(feature_to_value) == 0:
                    continue

                print("\n===================================================")
                print(
                    f"Swap local state: {source_emotion} -> {target_emotion} | "
                    f"L{layer_idx} | "
                    f"source feats={len(source_feature_indices)} | "
                    f"target feats={len(target_feature_indices)} | "
                    f"patched={len(feature_to_value)}"
                )
                print("===================================================")

                sae = sae_cache.load(layer_idx)

                result_df = run_one_layer_feature_swap(
                    llm=llm,
                    sae=sae,
                    df_source=df_source,
                    source_emotion=source_emotion,
                    target_emotion=target_emotion,
                    layer_idx=layer_idx,
                    source_feature_indices=source_feature_indices,
                    target_feature_indices=target_feature_indices,
                    feature_to_value=feature_to_value,
                    emotion_to_token=emotion_to_token,
                    batch_size=batch_size,
                    emotions=emotions,
                )

                all_results.append(result_df)
                done_keys.add(key)

                partial_instance_df = _safe_concat(all_results)
                partial_summary_df = summarize_swap_results(partial_instance_df, emotions)

                safe_save_csv(partial_instance_df, output_instance_level)
                safe_save_csv(partial_summary_df, output_summary)

                print("Saved partial results.")

    instance_df = _safe_concat(all_results)
    summary_df = summarize_swap_results(instance_df, emotions)

    safe_save_csv(instance_df, output_instance_level)
    safe_save_csv(summary_df, output_summary)

    print("\nDone swap pipeline.")
    print("Saved instance-level file:", output_instance_level)
    print("Saved summary file:", output_summary)
    print("Saved feature mean file:", output_feature_means)

    return feature_mean_df, instance_df, summary_df
