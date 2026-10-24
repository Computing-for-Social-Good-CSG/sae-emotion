import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from sae_emotion.logit_ablation_utils import (
    cleanup_memory,
    safe_save_csv,
    build_examples_by_emotion,
    load_llm,
    SAECache,
    zero_ablate_feature,
)


# ============================================================
# Emotion feature graph / edge patching pipeline
# ============================================================


def clear_cuda():
    """
    Backward-compatible alias used by older notebooks.
    """
    cleanup_memory()



def zero_ablate(
    sae,
    source_sae_features: torch.Tensor,
    feature_idx: int,
    residual: torch.Tensor,
) -> torch.Tensor:
    """
    Backward-compatible single-feature zero ablation helper.
    Reconstructs the hidden state with one SAE feature zeroed out.
    """
    return zero_ablate_feature(
        sae=sae,
        source_sae_features=source_sae_features,
        feature_idx=feature_idx,
        residual=residual,
    )



def parse_affected_emotions(
    x,
    emotions: List[str],
) -> List[str]:
    """
    Parse comma-separated affected_emotions field.
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "":
        return []

    emotion_set = set(emotions)

    return [
        e.strip()
        for e in x.split(",")
        if e.strip() in emotion_set
    ]



def edge_patching(
    llm,
    source_sae,
    target_sae,
    examples: List[str],
    source_layer: int,
    source_features,
    target_layer: int,
    target_features,
    use_abs_effect: bool = False,
    condition_on_target_active: bool = False,
    target_active_threshold: float = 0.0,
    min_target_active_count: int = 5,
) -> List[dict]:
    """
    Estimate directed feature edges from source_features to target_features.

    Edge weight:
        clean target activation - target activation after source feature ablation

    If use_abs_effect=True:
        abs(clean target activation - ablated target activation)

    If condition_on_target_active=True:
        average only over examples where the clean target feature is active.
    """
    source_features = list(map(int, source_features))
    target_features = list(map(int, target_features))

    if len(examples) == 0 or len(source_features) == 0 or len(target_features) == 0:
        return []

    edges = []

    # ------------------------------------------------------------
    # 1. Clean activations
    # ------------------------------------------------------------
    with torch.no_grad():
        with llm.trace(examples):
            source_hidden = llm.model.layers[int(source_layer)].output[:, -1, :].save()
            target_hidden = llm.model.layers[int(target_layer)].output[:, -1, :].save()

    source_sae_features = source_sae.encode(source_hidden)
    source_hidden_decoded = source_sae.decode(source_sae_features)
    source_sae_residual = source_hidden - source_hidden_decoded

    target_sae_features_clean = target_sae.encode(target_hidden)
    clean_targets = target_sae_features_clean[:, target_features]

    # ------------------------------------------------------------
    # 2. Ablate each source feature and collect target hidden states
    # ------------------------------------------------------------
    target_hiddens = []

    for source_feat in source_features:
        with torch.no_grad():
            with llm.trace(examples):
                llm.model.layers[int(source_layer)].output[:, -1, :] = zero_ablate(
                    sae=source_sae,
                    source_sae_features=source_sae_features,
                    feature_idx=int(source_feat),
                    residual=source_sae_residual,
                )

                target_hidden_ablated = llm.model.layers[int(target_layer)].output[:, -1, :].save()
                target_hiddens.append(target_hidden_ablated)

    target_hiddens = torch.stack(target_hiddens)

    # ------------------------------------------------------------
    # 3. Encode ablated target hidden states
    # ------------------------------------------------------------
    num_source_feats = len(source_features)
    batch_size = target_hiddens.shape[1]

    target_hiddens_flat = target_hiddens.reshape(num_source_feats * batch_size, -1)

    with torch.no_grad():
        target_sae_features_flat = target_sae.encode(target_hiddens_flat)

    target_sae_features = target_sae_features_flat.reshape(
        num_source_feats,
        batch_size,
        -1,
    )

    ablated_targets = target_sae_features[:, :, target_features]
    clean_targets_expanded = clean_targets.unsqueeze(0)

    diffs = clean_targets_expanded - ablated_targets

    if use_abs_effect:
        diffs = diffs.abs()

    # ------------------------------------------------------------
    # 4. Aggregate into edges
    # ------------------------------------------------------------
    for s_i, source_feat in enumerate(source_features):
        for t_i, target_feat in enumerate(target_features):
            target_clean = clean_targets[:, t_i]
            edge_diffs = diffs[s_i, :, t_i]

            if condition_on_target_active:
                mask = target_clean > target_active_threshold
                n_used = int(mask.sum().item())

                if n_used < min_target_active_count:
                    mean_effect = float("nan")
                    mean_abs_effect = float("nan")
                else:
                    selected_diffs = edge_diffs[mask]
                    mean_effect = float(selected_diffs.mean().item())
                    mean_abs_effect = float(selected_diffs.abs().mean().item())
            else:
                n_used = int(batch_size)
                mean_effect = float(edge_diffs.mean().item())
                mean_abs_effect = float(edge_diffs.abs().mean().item())

            clean_mean = float(target_clean.mean().item())
            clean_active_count = int((target_clean > target_active_threshold).sum().item())

            if abs(clean_mean) > 1e-8:
                normalized_effect = mean_effect / (abs(clean_mean) + 1e-8)
            else:
                normalized_effect = float("nan")

            edges.append(
                {
                    "source_layer": int(source_layer),
                    "source_feature": int(source_feat),
                    "target_layer": int(target_layer),
                    "target_feature": int(target_feat),
                    "n_examples_used": int(n_used),
                    "clean_target_mean": clean_mean,
                    "clean_target_active_count": clean_active_count,
                    "mean_effect": mean_effect,
                    "mean_abs_effect": mean_abs_effect,
                    "normalized_effect": normalized_effect,
                }
            )

    del source_hidden
    del target_hidden
    del source_sae_features
    del source_hidden_decoded
    del source_sae_residual
    del target_sae_features_clean
    del clean_targets
    del target_hiddens
    del target_hiddens_flat
    del target_sae_features_flat
    del target_sae_features
    del ablated_targets
    del clean_targets_expanded
    del diffs

    cleanup_memory()

    return edges



def build_emotion_feature_graphs(
    llm,
    emotion_spec: pd.DataFrame,
    examples_by_emotion: Dict[str, List[str]],
    model_name: str,
    output_dir: str,
    emotions: List[str],
    sae_size: str = "big",
    device: str = "cuda",
    min_source_target_layer_gap: int = 1,
    use_abs_effect: bool = False,
    condition_on_target_active: bool = False,
    target_active_threshold: float = 0.0,
    min_target_active_count: int = 10,
) -> pd.DataFrame:
    """
    Build directed feature-edge graphs among selected emotion features.

    This expects one row per selected feature with:
        layer_idx, feature_idx, affected_emotions

    It saves one CSV per conditioning emotion and one combined CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    required_cols = ["layer_idx", "feature_idx", "affected_emotions"]
    missing = [c for c in required_cols if c not in emotion_spec.columns]
    if missing:
        raise ValueError(f"emotion_spec is missing columns: {missing}")

    emotion_spec = emotion_spec.copy()
    emotion_spec["layer_idx"] = emotion_spec["layer_idx"].astype(int)
    emotion_spec["feature_idx"] = emotion_spec["feature_idx"].astype(int)

    emotion_spec["affected_emotion_list"] = emotion_spec["affected_emotions"].apply(
        lambda x: parse_affected_emotions(x, emotions=emotions)
    )

    emotion_spec["num_affected_emotions"] = emotion_spec["affected_emotion_list"].apply(len)

    sae_cache = SAECache(
        model_name=model_name,
        device=device,
        sae_size=sae_size,
    )

    all_edges = []

    for emotion in emotions:
        print(f"\nBuilding graph for emotion: {emotion}")

        if emotion not in examples_by_emotion:
            print(f"Skipping {emotion}: no examples found.")
            continue

        examples = examples_by_emotion[emotion]

        if len(examples) == 0:
            print(f"Skipping {emotion}: empty example list.")
            continue

        emotion_features = emotion_spec[
            emotion_spec["affected_emotion_list"].apply(lambda xs: emotion in xs)
        ].copy()

        if len(emotion_features) == 0:
            print(f"Skipping {emotion}: no features found.")
            continue

        layer_to_features = (
            emotion_features
            .groupby("layer_idx")["feature_idx"]
            .apply(lambda x: sorted(list(map(int, x))))
            .to_dict()
        )

        layers = sorted(layer_to_features.keys())
        emotion_edges = []

        for source_layer in tqdm(layers, desc=f"{emotion} source layers"):
            source_sae = sae_cache.load(int(source_layer))
            source_features = layer_to_features[int(source_layer)]

            for target_layer in layers:
                if int(target_layer) - int(source_layer) < int(min_source_target_layer_gap):
                    continue

                target_sae = sae_cache.load(int(target_layer))
                target_features = layer_to_features[int(target_layer)]

                edges = edge_patching(
                    llm=llm,
                    source_sae=source_sae,
                    target_sae=target_sae,
                    examples=examples,
                    source_layer=int(source_layer),
                    source_features=source_features,
                    target_layer=int(target_layer),
                    target_features=target_features,
                    use_abs_effect=use_abs_effect,
                    condition_on_target_active=condition_on_target_active,
                    target_active_threshold=target_active_threshold,
                    min_target_active_count=min_target_active_count,
                )

                for edge in edges:
                    edge["conditioning_emotion"] = emotion

                    src_row = emotion_features[
                        (emotion_features["layer_idx"] == edge["source_layer"])
                        & (emotion_features["feature_idx"] == edge["source_feature"])
                    ].iloc[0]

                    tgt_row = emotion_features[
                        (emotion_features["layer_idx"] == edge["target_layer"])
                        & (emotion_features["feature_idx"] == edge["target_feature"])
                    ].iloc[0]

                    edge["source_affected_emotions"] = src_row["affected_emotions"]
                    edge["source_num_affected_emotions"] = int(src_row["num_affected_emotions"])
                    edge["target_affected_emotions"] = tgt_row["affected_emotions"]
                    edge["target_num_affected_emotions"] = int(tgt_row["num_affected_emotions"])

                emotion_edges.extend(edges)
                cleanup_memory()

        emotion_edges_df = pd.DataFrame(emotion_edges)

        emotion_output_path = os.path.join(
            output_dir,
            f"{emotion}_affected_feature_edges.csv",
        )

        safe_save_csv(emotion_edges_df, emotion_output_path)
        print(f"Saved {emotion} edges to: {emotion_output_path}")

        all_edges.append(emotion_edges_df)

    all_edges_df = (
        pd.concat(all_edges, ignore_index=True)
        if all_edges
        else pd.DataFrame()
    )

    all_output_path = os.path.join(
        output_dir,
        "all_affected_feature_edges.csv",
    )

    safe_save_csv(all_edges_df, all_output_path)
    print(f"\nSaved all edges to: {all_output_path}")

    return all_edges_df



def run_emotion_feature_graph_pipeline(
    model_name: str,
    emotion_spec_csv_path: str,
    dataset_csv_path: str,
    output_dir: str,
    emotions: List[str],
    sae_size: str = "big",
    text_col: str = "input_text",
    label_col: str = "emotion",
    sample_size_per_emotion: Optional[int] = 300,
    random_state: int = 42,
    apply_prompt_template: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
    min_source_target_layer_gap: int = 1,
    use_abs_effect: bool = False,
    condition_on_target_active: bool = False,
    target_active_threshold: float = 0.0,
    min_target_active_count: int = 10,
) -> pd.DataFrame:
    """
    Full wrapper for the emotion feature graph code.

    It performs:
    1. Load selected unique emotion features.
    2. Build balanced examples_by_emotion from dataset.
    3. Load LLM.
    4. Build source-feature -> target-feature edge graphs.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because device='cuda', but torch.cuda.is_available() is False.")

    emotion_spec = pd.read_csv(emotion_spec_csv_path)

    examples_by_emotion = build_examples_by_emotion(
        dataset_csv_path=dataset_csv_path,
        emotions=emotions,
        text_col=text_col,
        label_col=label_col,
        sample_size_per_emotion=sample_size_per_emotion,
        random_state=random_state,
        apply_prompt_template=apply_prompt_template,
    )

    llm = load_llm(
        model_name=model_name,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
    )

    all_edges_df = build_emotion_feature_graphs(
        llm=llm,
        emotion_spec=emotion_spec,
        examples_by_emotion=examples_by_emotion,
        model_name=model_name,
        output_dir=output_dir,
        emotions=emotions,
        sae_size=sae_size,
        device=device,
        min_source_target_layer_gap=min_source_target_layer_gap,
        use_abs_effect=use_abs_effect,
        condition_on_target_active=condition_on_target_active,
        target_active_threshold=target_active_threshold,
        min_target_active_count=min_target_active_count,
    )

    return all_edges_df
