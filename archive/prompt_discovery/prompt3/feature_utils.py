import os
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


_DESC_CACHE = {}


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def get_sae_config(model_name: str, sae_size: str = "big"):

    if sae_size not in {"big", "medium", "small"}:
        raise ValueError(f"Unsupported sae_size: {sae_size}")

    if "gemma-2-2b" in model_name:
        release = "gemma-scope-2b-pt-res-canonical"
        sae_id_fn = lambda i: f"layer_{i}/width_16k/canonical"

    elif "gemma-2-9b" in model_name:
        release = "gemma-scope-9b-pt-res-canonical"
        sae_id_fn = lambda i: f"layer_{i}/width_16k/canonical"

    elif "gemma-3-1b-pt" in model_name:
        release = "gemma-scope-2-1b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-1b-it" in model_name:
        release = "gemma-scope-2-1b-it-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-4b-pt" in model_name:
        release = "gemma-scope-2-4b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-4b-it" in model_name:
        release = "gemma-scope-2-4b-it-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-12b-pt" in model_name:
        release = "gemma-scope-2-12b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-12b-it" in model_name:
        release = "gemma-scope-2-12b-it-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "Llama-3.1-8B" in model_name:
        release = "llama_scope_lxr_8x"
        sae_id_fn = lambda i: f"l{i}r_8x"

    elif "DeepSeek" in model_name:
        return None, None

    else:
        raise ValueError(f"Unsupported model for SAE loading: {model_name}")

    return release, sae_id_fn


def fetch_description(
    sae_release: str,
    sae_id: str,
    feature_idx: int,
    sleep_seconds: float = 0.01,
) -> Tuple[Optional[str], bool]:

    cache_key = (sae_release, sae_id, int(feature_idx))

    if cache_key in _DESC_CACHE:
        return _DESC_CACHE[cache_key], True

    try:
        # ============================================================
        # Convert SAE config -> Neuronpedia format
        # ============================================================

        if sae_release == "gemma-scope-2b-pt-res-canonical":
            model_slug = "gemma-2-2b"
            layer_idx = int(sae_id.split("/")[0].replace("layer_", ""))
            neuronpedia_sae_id = f"{layer_idx}-gemmascope-res-16k"

        elif sae_release == "gemma-scope-9b-pt-res-canonical":
            model_slug = "gemma-2-9b"
            layer_idx = int(sae_id.split("/")[0].replace("layer_", ""))
            neuronpedia_sae_id = f"{layer_idx}-gemmascope-res-16k"

        elif sae_release in {
            "gemma-scope-2-1b-pt-res-all",
            "gemma-scope-2-1b-it-res-all",
            "gemma-scope-2-4b-pt-res-all",
            "gemma-scope-2-4b-it-res-all",
        }:
            if sae_release == "gemma-scope-2-1b-pt-res-all":
                model_slug = "gemma-3-1b"
            elif sae_release == "gemma-scope-2-1b-it-res-all":
                model_slug = "gemma-3-1b-it"
            elif sae_release == "gemma-scope-2-4b-pt-res-all":
                model_slug = "gemma-3-4b"
            else:
                model_slug = "gemma-3-4b-it"
        
            parts = sae_id.split("_")
            layer_idx = int(parts[1])
        
            neuronpedia_sae_id = f"{layer_idx}-gemmascope-2-res-16k"

        elif sae_release == "llama_scope_lxr_8x":
            model_slug = "llama3.1-8b"

            # SAE Lens:
            #   l16r_8x
            #
            # Neuronpedia:
            #   16-llamascope-res-32k
            layer_idx = int(sae_id.replace("l", "").split("r")[0])
            neuronpedia_sae_id = f"{layer_idx}-llamascope-res-32k"

        else:
            print(
                f"Unsupported SAE release for descriptions: "
                f"{sae_release}"
            )
            _DESC_CACHE[cache_key] = None
            return None, False

        # ============================================================
        # Build URL
        # ============================================================

        url = (
            f"https://www.neuronpedia.org/api/feature/"
            f"{model_slug}/{neuronpedia_sae_id}/{int(feature_idx)}"
        )

        response = requests.get(url, timeout=10)

        if response.ok:
            data = response.json()
            desc = None

            explanations = data.get("explanations")

            if isinstance(explanations, list) and len(explanations) > 0:
                desc = explanations[0].get("description")

            _DESC_CACHE[cache_key] = desc

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            return desc, False

        print(f"Request failed: {response.status_code} | {url}")

    except Exception as e:
        print(f"Fetch failed for feature {feature_idx}: {e}")

    _DESC_CACHE[cache_key] = None

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return None, False


def extract_nonzero_activated_features(
    sae_activations: np.ndarray,
    df_correct: pd.DataFrame,
    model_name: str,
    emotions: List[str],
    emotion_col: str = "pred_emotion",
    top_percent: Optional[float] = None,
    ranking_metric: str = "avg_activation",
    fetch_descriptions: bool = False,
    sleep_seconds: float = 0.01,
    sae_size: str = "big",
) -> pd.DataFrame:

    if ranking_metric not in {
        "avg_activation",
        "avg_act_when_active",
        "activation_rate",
    }:
        raise ValueError(f"Unsupported ranking_metric: {ranking_metric}")

    sae_release, sae_id_fn = get_sae_config(
        model_name=model_name,
        sae_size=sae_size,
    )

    n_layers, n_prompts, _ = sae_activations.shape

    correct_indices = df_correct.index.to_numpy()
    correct_indices = correct_indices[
        (correct_indices >= 0) & (correct_indices < n_prompts)
    ]

    if len(correct_indices) == 0:
        raise ValueError("No valid correct indices found.")

    df_analysis = df_correct.loc[correct_indices].copy()
    analysis_emotions = df_analysis[emotion_col].to_numpy()
    analysis_acts = sae_activations[:, correct_indices, :]

    rows = []

    for emotion in tqdm(emotions, desc="Extracting nonzero features"):
        emotion_mask = analysis_emotions == emotion

        if emotion_mask.sum() == 0:
            continue

        emotion_acts = analysis_acts[:, emotion_mask, :]

        for layer_idx in range(n_layers):
            layer_acts = emotion_acts[layer_idx]

            avg_acts = layer_acts.mean(axis=0)

            active_mask = layer_acts > 0
            activation_count = active_mask.sum(axis=0)
            activation_rate = activation_count / layer_acts.shape[0]

            act_sum = layer_acts.sum(axis=0)

            avg_act_when_active = np.zeros_like(avg_acts)
            nonzero_active_mask = activation_count > 0
            avg_act_when_active[nonzero_active_mask] = (
                act_sum[nonzero_active_mask]
                / activation_count[nonzero_active_mask]
            )

            nonzero_feature_idx = np.where(avg_acts != 0)[0]

            if len(nonzero_feature_idx) == 0:
                continue

            metric_values = {
                "avg_activation": avg_acts,
                "avg_act_when_active": avg_act_when_active,
                "activation_rate": activation_rate,
            }[ranking_metric]

            if top_percent is not None:
                if top_percent <= 0:
                    raise ValueError("top_percent must be positive.")

                if top_percent > 1:
                    top_percent = top_percent / 100.0

                top_k = max(1, math.ceil(len(nonzero_feature_idx) * top_percent))
                nonzero_metric_values = metric_values[nonzero_feature_idx]
                top_relative_idx = np.argsort(nonzero_metric_values)[-top_k:][::-1]
                selected_feature_idx = nonzero_feature_idx[top_relative_idx]
            else:
                selected_feature_idx = nonzero_feature_idx

            sae_id = sae_id_fn(layer_idx)

            for feature_idx in selected_feature_idx:
                feature_idx = int(feature_idx)

                desc = None
                from_cache = False

                if fetch_descriptions:
                    desc, from_cache = fetch_description(
                        sae_release=sae_release,
                        sae_id=sae_id,
                        feature_idx=feature_idx,
                        sleep_seconds=sleep_seconds,
                    )

                rows.append({
                    "emotion": emotion,
                    "layer_idx": int(layer_idx),
                    "feature_idx": feature_idx,
                    "avg_activation": float(avg_acts[feature_idx]),
                    "avg_act_when_active": float(avg_act_when_active[feature_idx]),
                    "activation_count": int(activation_count[feature_idx]),
                    "activation_rate": float(activation_rate[feature_idx]),
                    "ranking_metric": ranking_metric,
                    "description": desc,
                    "description_from_cache": from_cache,
                    "sae_release": sae_release,
                    "sae_id": sae_id,
                })

    return pd.DataFrame(rows)


def filter_effective_and_core_features(
    df_features: pd.DataFrame,
    activation_rate_min: float = 0.01,
    top_percent: float = 0.1,
    ranking_metric: str = "avg_activation",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if top_percent > 1:
        top_percent = top_percent / 100.0

    df_eff = df_features[df_features["activation_rate"] > activation_rate_min].copy()

    rank_col = f"{ranking_metric}_rank_pct"

    df_core = df_eff.copy()
    df_core[rank_col] = (
        df_core
        .groupby("emotion")[ranking_metric]
        .rank(method="first", ascending=False, pct=True)
    )

    df_core = df_core[df_core[rank_col] <= top_percent].copy()

    return df_eff, df_core


def enrich_with_descriptions(
    df_input: pd.DataFrame,
    save_path: str,
    sleep_seconds: float = 0.01,
    save_every: int = 100,
) -> pd.DataFrame:

    df_out = df_input.copy().reset_index(drop=True)

    if "description" not in df_out.columns:
        df_out["description"] = None

    if "description_from_cache" not in df_out.columns:
        df_out["description_from_cache"] = False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for idx in tqdm(range(len(df_out)), desc="Fetching descriptions"):
        if pd.notna(df_out.loc[idx, "description"]):
            continue

        row = df_out.loc[idx]

        desc, from_cache = fetch_description(
            sae_release=row["sae_release"],
            sae_id=row["sae_id"],
            feature_idx=row["feature_idx"],
            sleep_seconds=sleep_seconds,
        )

        df_out.loc[idx, "description"] = desc
        df_out.loc[idx, "description_from_cache"] = from_cache

        if save_every is not None and (idx + 1) % save_every == 0:
            df_out.to_csv(save_path, index=False)

    df_out.to_csv(save_path, index=False)
    return df_out


def save_feature_tables(
    df_features: pd.DataFrame,
    output_dir: str,
    model_name: str,
    activation_rate_min: float = 0.01,
    top_percent: float = 0.1,
    ranking_metric: str = "avg_activation",
):

    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_safe_name(model_name)

    nonzero_path = f"{output_dir}/nonzero_activated_features_{safe_name}.csv"
    df_features.to_csv(nonzero_path, index=False)

    df_eff, df_core = filter_effective_and_core_features(
        df_features=df_features,
        activation_rate_min=activation_rate_min,
        top_percent=top_percent,
        ranking_metric=ranking_metric,
    )

    effective_path = (
        f"{output_dir}/effective_features_activation_rate_gt_{activation_rate_min}.csv"
    )

    core_path = (
        f"{output_dir}/core_emotion_features_top{top_percent}_by_{ranking_metric}.csv"
    )

    df_eff.to_csv(effective_path, index=False)
    df_core.to_csv(core_path, index=False)

    return {
        "nonzero": df_features,
        "effective": df_eff,
        "core": df_core,
        "paths": {
            "nonzero": nonzero_path,
            "effective": effective_path,
            "core": core_path,
        },
    }