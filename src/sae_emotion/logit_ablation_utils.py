import os
import gc
import shutil
import random
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from nnsight import LanguageModel
from sae_lens import SAE


# ============================================================
# Basic helperszero_ablate_feature
# ============================================================

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def zero_shot_template(text: str) -> str:
    return f"In this text: {text}, the emotion implied is:"


# ============================================================
# Model / SAE config
# ============================================================

def get_sae_config(
    model_name: str,
    sae_size: str = "big",
):

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

    elif "gemma-3-4b-pt" in model_name:
        release = "gemma-scope-2-4b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-1b-it" in model_name:
        release = "gemma-scope-2-1b-it-res-all"
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


def load_llm(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
):
    kwargs = {
        "device_map": device,
        "dtype": dtype,
    }

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    llm = LanguageModel(
        model_name,
        **kwargs,
    )

    try:
        _ = llm.model.layers
    except Exception:
        llm.model.layers = llm.model.language_model.layers

    return llm


def build_emotion_token_map(
    llm,
    emotions: List[str],
) -> Dict[str, int]:
    emotion_to_token = {}

    for emotion in emotions:
        token_ids = llm.tokenizer.encode(" " + emotion)
        emotion_to_token[emotion] = token_ids[-1]
        print(f"{emotion}: {token_ids} -> using {token_ids[-1]}")

    return emotion_to_token


# ============================================================
# Dataset helpers
# ============================================================

def build_examples_by_emotion(
    dataset_csv_path: str,
    emotions: List[str],
    text_col: str = "input_text",
    label_col: str = "emotion",
    sample_size_per_emotion: Optional[int] = 300,
    random_state: int = 42,
    apply_prompt_template: bool = True,
    prompt_fn: Callable[[str], str] = zero_shot_template,
) -> Dict[str, List[str]]:

    dataset = pd.read_csv(dataset_csv_path)

    if text_col not in dataset.columns:
        raise ValueError(f"text_col={text_col} not found in dataset.")

    if label_col not in dataset.columns:
        raise ValueError(f"label_col={label_col} not found in dataset.")

    dataset = dataset[[text_col, label_col]].dropna().copy()
    dataset[label_col] = dataset[label_col].astype(str).str.lower().str.strip()

    examples_by_emotion = {
        emotion: dataset.loc[dataset[label_col] == emotion, text_col].tolist()
        for emotion in emotions
    }

    print("Original counts:")
    for emotion, examples in examples_by_emotion.items():
        print(emotion, len(examples))

    if sample_size_per_emotion is not None:
        random.seed(random_state)

        sampled_examples_by_emotion = {}

        for emotion, examples in examples_by_emotion.items():
            if len(examples) >= sample_size_per_emotion:
                sampled = random.sample(examples, sample_size_per_emotion)
            else:
                sampled = examples

            sampled_examples_by_emotion[emotion] = sampled

        examples_by_emotion = sampled_examples_by_emotion

    if apply_prompt_template:
        examples_by_emotion = {
            emotion: [prompt_fn(x) for x in examples]
            for emotion, examples in examples_by_emotion.items()
        }

    print("\nFinal counts:")
    for emotion, examples in examples_by_emotion.items():
        print(emotion, len(examples))

    return examples_by_emotion


# ============================================================
# SAE ablation
# ============================================================

class SAECache:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        sae_size: str = "big",
    ):
        self.model_name = model_name
        self.device = device
        self.sae_size = sae_size

        self.release, self.sae_id_fn = get_sae_config(
            model_name=model_name,
            sae_size=sae_size,
        )

        self.cache = {}

    def load(self, layer_idx: int):
        layer_idx = int(layer_idx)

        if layer_idx in self.cache:
            return self.cache[layer_idx]

        sae_id = self.sae_id_fn(layer_idx)

        sae = SAE.from_pretrained(
            release=self.release,
            sae_id=sae_id,
            device=self.device,
        )

        self.cache[layer_idx] = sae
        print(f"Loaded SAE: {sae_id}")

        return sae


def zero_ablate_feature(
    sae,
    source_sae_features: torch.Tensor,
    feature_idx: int,
    residual: torch.Tensor,
) -> torch.Tensor:

    modified_sae_features = source_sae_features.clone()
    modified_sae_features[:, int(feature_idx)] = 0.0

    modified_hidden = sae.decode(modified_sae_features)

    return modified_hidden + residual


def compute_feature_logit_effect(
    llm,
    sae,
    examples: List[str],
    layer_idx: int,
    feature_idx: int,
    target_token: int,
    batch_size: int = 16,
    use_abs: bool = True,
) -> float:

    all_scores = []

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]

        with torch.no_grad():
            with llm.trace(batch_examples):
                source_hidden = (
                    llm.model.layers[int(layer_idx)]
                    .output[:, -1, :]
                    .save()
                )

                clean_logits = (
                    llm.output.logits[:, -1, target_token]
                    .save()
                )

        source_sae_features = sae.encode(source_hidden)
        source_hidden_decoded = sae.decode(source_sae_features)
        residual = source_hidden - source_hidden_decoded

        with torch.no_grad():
            with llm.trace(batch_examples):
                llm.model.layers[int(layer_idx)].output[:, -1, :] = zero_ablate_feature(
                    sae=sae,
                    source_sae_features=source_sae_features,
                    feature_idx=int(feature_idx),
                    residual=residual,
                )

                ablated_logits = (
                    llm.output.logits[:, -1, target_token]
                    .save()
                )

        if use_abs:
            batch_scores = (clean_logits - ablated_logits).abs()
        else:
            batch_scores = clean_logits - ablated_logits

        all_scores.append(batch_scores.detach().float().cpu())

        del source_hidden
        del clean_logits
        del source_sae_features
        del source_hidden_decoded
        del residual
        del ablated_logits
        del batch_scores

        cleanup_memory()

    all_scores = torch.cat(all_scores)

    return all_scores.mean().item()


# ============================================================
# Main pipeline
# ============================================================

def prepare_feature_df_for_logit_ablation(
    input_feature_csv_path: str,
    output_csv_path: str,
    emotions: List[str],
    resume_from_output: bool = True,
    preserve_existing_input_logits: bool = True,
    backup_existing_output: bool = True,
) -> pd.DataFrame:

    logit_cols = [f"logit_{emotion}" for emotion in emotions]

    if resume_from_output and os.path.exists(output_csv_path):
        print("Resuming from existing output file:", output_csv_path)

        if backup_existing_output:
            backup_path = output_csv_path.replace(".csv", "_backup_before_resume.csv")
            shutil.copy2(output_csv_path, backup_path)
            print("Backup saved to:", backup_path)

        df_features = pd.read_csv(output_csv_path)

    else:
        print("Starting from input feature file:", input_feature_csv_path)
        df_features = pd.read_csv(input_feature_csv_path)

        if not preserve_existing_input_logits:
            for col in logit_cols:
                if col in df_features.columns:
                    df_features = df_features.drop(columns=[col])

    required_cols = ["layer_idx", "feature_idx"]

    for col in required_cols:
        if col not in df_features.columns:
            raise ValueError(f"Missing required column: {col}")

    df_features["layer_idx"] = df_features["layer_idx"].astype(int)
    df_features["feature_idx"] = df_features["feature_idx"].astype(int)

    for col in logit_cols:
        if col not in df_features.columns:
            df_features[col] = pd.NA

        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    return df_features


def run_feature_logit_ablation_pipeline(
    model_name: str,
    input_feature_csv_path: str,
    output_csv_path: str,
    dataset_csv_path: str,
    emotions: List[str],
    sae_size: str = "big",
    text_col: str = "input_text",
    label_col: str = "emotion",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
    sample_size_per_emotion: Optional[int] = 300,
    random_state: int = 42,
    apply_prompt_template: bool = True,
    ablation_batch_size: int = 16,
    use_abs: bool = True,
    save_every: int = 50,
    resume_from_output: bool = True,
    preserve_existing_input_logits: bool = True,
    backup_existing_output: bool = True,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df_features = prepare_feature_df_for_logit_ablation(
        input_feature_csv_path=input_feature_csv_path,
        output_csv_path=output_csv_path,
        emotions=emotions,
        resume_from_output=resume_from_output,
        preserve_existing_input_logits=preserve_existing_input_logits,
        backup_existing_output=backup_existing_output,
    )

    logit_cols = [f"logit_{emotion}" for emotion in emotions]

    total_cells = len(df_features) * len(logit_cols)
    filled_cells = int(df_features[logit_cols].notna().sum().sum())
    missing_cells = total_cells - filled_cells

    print("Total logit cells:", total_cells)
    print("Already filled:", filled_cells)
    print("Missing:", missing_cells)

    if missing_cells == 0:
        print("Everything is already complete. Saving final file.")
        safe_save_csv(df_features, output_csv_path)
        return df_features

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

    emotion_to_token = build_emotion_token_map(
        llm=llm,
        emotions=emotions,
    )

    sae_cache = SAECache(
        model_name=model_name,
        device=device,
        sae_size=sae_size,
    )

    newly_filled = 0

    for layer_idx, df_layer in tqdm(
        df_features.groupby("layer_idx"),
        desc="Layers",
    ):
        layer_row_indices = df_layer.index.tolist()

        if df_features.loc[layer_row_indices, logit_cols].notna().all(axis=None):
            print(f"Skipping layer {layer_idx}, already complete.")
            continue

        sae = sae_cache.load(layer_idx)

        for row_idx in tqdm(
            layer_row_indices,
            desc=f"Layer {layer_idx}",
            leave=False,
        ):
            if df_features.loc[row_idx, logit_cols].notna().all():
                continue

            feature_idx = int(df_features.loc[row_idx, "feature_idx"])

            for emotion in emotions:
                col = f"logit_{emotion}"

                if pd.notna(df_features.loc[row_idx, col]):
                    continue

                examples = examples_by_emotion[emotion]

                if len(examples) == 0:
                    print(f"Skipping {emotion}: no examples.")
                    df_features.loc[row_idx, col] = pd.NA
                    continue

                score = compute_feature_logit_effect(
                    llm=llm,
                    sae=sae,
                    examples=examples,
                    layer_idx=int(layer_idx),
                    feature_idx=feature_idx,
                    target_token=emotion_to_token[emotion],
                    batch_size=ablation_batch_size,
                    use_abs=use_abs,
                )

                df_features.loc[row_idx, col] = score
                newly_filled += 1

                if save_every is not None and newly_filled % save_every == 0:
                    safe_save_csv(df_features, output_csv_path)
                    print(f"Saved progress after {newly_filled} newly filled cells.")

        safe_save_csv(df_features, output_csv_path)
        print(f"Saved after layer {layer_idx}.")

    safe_save_csv(df_features, output_csv_path)

    final_filled = int(df_features[logit_cols].notna().sum().sum())

    print("Done. Saved to:", output_csv_path)
    print("Final filled cells:", final_filled)
    print("Newly filled this run:", newly_filled)

    return df_features

# ============================================================
# Joint top-k SAE ablation
# ============================================================

def zero_ablate_features(
    sae,
    source_sae_features: torch.Tensor,
    feature_indices,
    residual: torch.Tensor,
) -> torch.Tensor:
    """
    Jointly zero-ablate multiple SAE features at one layer.
    """
    modified_sae_features = source_sae_features.clone()
    feature_indices = [int(x) for x in feature_indices]

    if len(feature_indices) > 0:
        modified_sae_features[:, feature_indices] = 0.0

    modified_hidden = sae.decode(modified_sae_features)
    return modified_hidden + residual


def compute_topk_joint_logit_drop(
    llm,
    sae,
    examples: List[str],
    layer_idx: int,
    feature_indices,
    target_token: int,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    Ablate a set of SAE features together and measure target-emotion logit drop.

    signed_drop = clean_logit - ablated_logit

    Positive signed_drop means removing these features lowers the target emotion logit.
    """
    all_signed_drops = []
    all_abs_drops = []
    all_clean_logits = []
    all_ablated_logits = []

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]

        with torch.no_grad():
            with llm.trace(batch_examples):
                source_hidden = (
                    llm.model.layers[int(layer_idx)]
                    .output[:, -1, :]
                    .save()
                )

                clean_logits = (
                    llm.output.logits[:, -1, int(target_token)]
                    .save()
                )

        source_sae_features = sae.encode(source_hidden)
        source_hidden_decoded = sae.decode(source_sae_features)
        residual = source_hidden - source_hidden_decoded

        with torch.no_grad():
            with llm.trace(batch_examples):
                llm.model.layers[int(layer_idx)].output[:, -1, :] = zero_ablate_features(
                    sae=sae,
                    source_sae_features=source_sae_features,
                    feature_indices=feature_indices,
                    residual=residual,
                )

                ablated_logits = (
                    llm.output.logits[:, -1, int(target_token)]
                    .save()
                )

        signed_drop = clean_logits - ablated_logits
        abs_drop = signed_drop.abs()

        all_signed_drops.append(signed_drop.detach().float().cpu())
        all_abs_drops.append(abs_drop.detach().float().cpu())
        all_clean_logits.append(clean_logits.detach().float().cpu())
        all_ablated_logits.append(ablated_logits.detach().float().cpu())

        del source_hidden
        del clean_logits
        del source_sae_features
        del source_hidden_decoded
        del residual
        del ablated_logits
        del signed_drop
        del abs_drop

        cleanup_memory()

    all_signed_drops = torch.cat(all_signed_drops)
    all_abs_drops = torch.cat(all_abs_drops)
    all_clean_logits = torch.cat(all_clean_logits)
    all_ablated_logits = torch.cat(all_ablated_logits)

    return {
        "mean_signed_drop": all_signed_drops.mean().item(),
        "std_signed_drop": all_signed_drops.std(unbiased=False).item(),
        "mean_abs_drop": all_abs_drops.mean().item(),
        "std_abs_drop": all_abs_drops.std(unbiased=False).item(),
        "mean_clean_logit": all_clean_logits.mean().item(),
        "mean_ablated_logit": all_ablated_logits.mean().item(),
        "n_examples": int(len(all_signed_drops)),
    }


def get_topk_features_for_layer_emotion(
    df: pd.DataFrame,
    layer_idx: int,
    emotion: str,
    k: int,
    use_abs_ranking: bool = True,
):
    """
    Pick top-k features within one layer for one target emotion.
    Ranking is based on existing single-feature ablation effect.
    """
    logit_col = f"logit_{emotion}"

    df_layer = df[df["layer_idx"] == int(layer_idx)].copy()
    df_layer[logit_col] = pd.to_numeric(df_layer[logit_col], errors="coerce")
    df_layer = df_layer.dropna(subset=[logit_col])

    if len(df_layer) == 0:
        return [], df_layer

    if use_abs_ranking:
        df_layer["_rank_score"] = df_layer[logit_col].abs()
    else:
        df_layer["_rank_score"] = df_layer[logit_col]

    df_layer = df_layer.sort_values("_rank_score", ascending=False)
    selected = df_layer.head(int(k)).copy()

    return selected["feature_idx"].astype(int).tolist(), selected


def load_existing_joint_ablation_results(out_path: str):
    """
    Resume joint top-k ablation from an existing output file.
    """
    if not os.path.exists(out_path):
        return [], set()

    existing = pd.read_csv(out_path)

    if len(existing) == 0:
        return [], set()

    required_cols = ["layer_idx", "target_emotion", "top_k"]
    missing = [c for c in required_cols if c not in existing.columns]
    if missing:
        raise ValueError(
            f"Existing output is missing columns {missing}. "
            f"Move or delete this file before resuming: {out_path}"
        )

    done_keys = set(
        zip(
            existing["layer_idx"].astype(int),
            existing["target_emotion"].astype(str),
            existing["top_k"].astype(int),
        )
    )

    results = existing.to_dict("records")

    print(f"Resuming from: {out_path}")
    print(f"Already completed rows: {len(done_keys)}")

    return results, done_keys


def prepare_topk_feature_df(
    feature_csv_path: str,
    emotions: List[str],
    entropy_group: Optional[str] = "emotion_specific",
) -> pd.DataFrame:
    """
    Load feature file for top-k joint ablation.
    If entropy_group is not None, keep only rows with that entropy group.
    """
    df = pd.read_csv(feature_csv_path)

    if entropy_group is not None:
        if "entropy_group" not in df.columns:
            raise ValueError("Missing column in feature file: entropy_group")
        df = df[df["entropy_group"] == entropy_group].copy()

    required_cols = ["layer_idx", "feature_idx"] + [f"logit_{e}" for e in emotions]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature file: {missing}")

    df["layer_idx"] = df["layer_idx"].astype(int)
    df["feature_idx"] = df["feature_idx"].astype(int)

    for e in emotions:
        df[f"logit_{e}"] = pd.to_numeric(df[f"logit_{e}"], errors="coerce")

    return df


def run_topk_joint_ablation_pipeline(
    model_name: str,
    feature_csv_path: str,
    dataset_csv_path: str,
    output_csv_path: str,
    emotions: List[str],
    top_k_list: List[int],
    sae_size: str = "big",
    text_col: str = "input_text",
    label_col: str = "emotion",
    entropy_group: Optional[str] = "emotion_specific",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
    sample_size_per_emotion: Optional[int] = 300,
    batch_size: int = 128,
    random_state: int = 42,
    apply_prompt_template: bool = False,
    use_abs_ranking: bool = False,
    save_every_row: bool = True,
) -> pd.DataFrame:
    """
    Run top-k joint ablation per layer and emotion.

    Output rows are keyed by:
    layer_idx, target_emotion, top_k.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because device='cuda'.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df = prepare_topk_feature_df(
        feature_csv_path=feature_csv_path,
        emotions=emotions,
        entropy_group=entropy_group,
    )

    layers = sorted(df["layer_idx"].unique().tolist())

    print("Feature df shape:", df.shape)
    print("Layers:", layers)
    print("Top-k list:", top_k_list)
    print("Output path:", output_csv_path)

    results, done_keys = load_existing_joint_ablation_results(output_csv_path)

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

    emotion_to_token = build_emotion_token_map(
        llm=llm,
        emotions=emotions,
    )

    sae_cache = SAECache(
        model_name=model_name,
        device=device,
        sae_size=sae_size,
    )

    for layer_idx in tqdm(layers, desc="Layers"):
        sae = sae_cache.load(layer_idx)

        for emotion in tqdm(emotions, desc=f"Layer {layer_idx} emotions", leave=False):
            examples = examples_by_emotion[emotion]
            target_token = emotion_to_token[emotion]

            if len(examples) == 0:
                print(f"Skipping {emotion}: no examples.")
                continue

            for k in top_k_list:
                key = (int(layer_idx), str(emotion), int(k))

                if key in done_keys:
                    continue

                feature_indices, selected_df = get_topk_features_for_layer_emotion(
                    df=df,
                    layer_idx=layer_idx,
                    emotion=emotion,
                    k=k,
                    use_abs_ranking=use_abs_ranking,
                )

                if len(feature_indices) < k:
                    print(
                        f"Stopping L{layer_idx} {emotion}: only {len(feature_indices)} "
                        f"features available, fewer than top{k}."
                    )
                    break

                stats = compute_topk_joint_logit_drop(
                    llm=llm,
                    sae=sae,
                    examples=examples,
                    layer_idx=layer_idx,
                    feature_indices=feature_indices,
                    target_token=target_token,
                    batch_size=batch_size,
                )

                logit_col = f"logit_{emotion}"

                row = {
                    "layer_idx": int(layer_idx),
                    "target_emotion": emotion,
                    "top_k": int(k),
                    "n_selected_features": int(len(feature_indices)),
                    "features": ",".join(map(str, feature_indices)),
                    "single_feature_abs_effect_sum": float(selected_df[logit_col].abs().sum()),
                    "single_feature_abs_effect_mean": float(selected_df[logit_col].abs().mean()),
                    "single_feature_abs_effect_min": float(selected_df[logit_col].abs().min()),
                    "single_feature_abs_effect_max": float(selected_df[logit_col].abs().max()),
                    **stats,
                }

                results.append(row)
                done_keys.add(key)

                result_df_partial = pd.DataFrame(results)

                if save_every_row:
                    safe_save_csv(result_df_partial, output_csv_path)

                print(
                    f"L{layer_idx:02d} {emotion:8s} top{k:<3d} "
                    f"signed_drop={stats['mean_signed_drop']:.6f} "
                    f"abs_drop={stats['mean_abs_drop']:.6f} "
                    f"features={feature_indices}"
                )

        result_df_partial = pd.DataFrame(results)
        safe_save_csv(result_df_partial, output_csv_path)
        print(f"Saved after layer {layer_idx}: {output_csv_path}")

    result_df = pd.DataFrame(results)
    safe_save_csv(result_df, output_csv_path)

    print("\nFinal result shape:", result_df.shape)
    print("Saved to:", output_csv_path)

    if len(result_df) > 0:
        print("\nMean signed drop by emotion and k:")
        print(
            result_df
            .groupby(["target_emotion", "top_k"])["mean_signed_drop"]
            .mean()
            .unstack()
        )

        print("\nMean signed drop by layer and k:")
        print(
            result_df
            .groupby(["layer_idx", "top_k"])["mean_signed_drop"]
            .mean()
            .unstack()
        )

    return result_df


# ============================================================
# Auto-select core features from top-k joint ablation
# ============================================================

def parse_feature_list(x) -> List[int]:
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "":
        return []

    return [int(v.strip()) for v in x.split(",") if v.strip() != ""]


def choose_min_k_for_layer_emotion(
    df_group: pd.DataFrame,
    min_signed_drop: float = 0.1,
    coverage_ratio: float = 0.80,
):
    """
    Choose the smallest k whose mean signed drop reaches coverage_ratio
    of the max valid signed drop for the same layer and emotion.
    """
    df_valid = df_group.dropna(subset=["mean_signed_drop"]).copy()
    df_valid = df_valid[df_valid["mean_signed_drop"] >= min_signed_drop].copy()

    if len(df_valid) == 0:
        return None

    max_drop = df_valid["mean_signed_drop"].max()
    threshold = coverage_ratio * max_drop

    df_valid = df_valid.sort_values("top_k", ascending=True)
    df_hit = df_valid[df_valid["mean_signed_drop"] >= threshold].copy()

    if len(df_hit) == 0:
        return None

    selected = df_hit.iloc[0].copy()
    selected["max_signed_drop"] = max_drop
    selected["coverage_threshold"] = threshold
    selected["coverage_ratio_actual"] = selected["mean_signed_drop"] / max_drop

    return selected


def select_core_features_from_joint_ablation(
    feature_csv_path: str,
    joint_ablation_csv_path: str,
    output_csv_path: str,
    emotions: List[str],
    min_signed_drop: float = 0.1,
    coverage_ratio: float = 0.80,
    entropy_group: Optional[str] = "emotion_specific",
) -> pd.DataFrame:
    """
    Use the top-k joint ablation table to auto-select core emotion features.
    """
    feature_df = pd.read_csv(feature_csv_path)
    joint_df = pd.read_csv(joint_ablation_csv_path)

    if entropy_group is not None:
        if "entropy_group" not in feature_df.columns:
            raise ValueError("Missing column in feature file: entropy_group")
        feature_df = feature_df[feature_df["entropy_group"] == entropy_group].copy()

    has_description = "description" in feature_df.columns

    required_joint_cols = [
        "layer_idx",
        "target_emotion",
        "top_k",
        "mean_signed_drop",
        "features",
    ]

    missing = [c for c in required_joint_cols if c not in joint_df.columns]
    if missing:
        raise ValueError(f"Missing columns in joint ablation file: {missing}")

    required_feature_cols = [
        "layer_idx",
        "feature_idx",
    ] + [f"logit_{e}" for e in emotions]

    if entropy_group is not None:
        required_feature_cols.append("entropy_group")

    missing = [c for c in required_feature_cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature file: {missing}")

    feature_df["layer_idx"] = feature_df["layer_idx"].astype(int)
    feature_df["feature_idx"] = feature_df["feature_idx"].astype(int)

    joint_df["layer_idx"] = joint_df["layer_idx"].astype(int)
    joint_df["top_k"] = joint_df["top_k"].astype(int)
    joint_df["target_emotion"] = joint_df["target_emotion"].astype(str)
    joint_df["mean_signed_drop"] = pd.to_numeric(
        joint_df["mean_signed_drop"],
        errors="coerce",
    )

    for e in emotions:
        feature_df[f"logit_{e}"] = pd.to_numeric(
            feature_df[f"logit_{e}"],
            errors="coerce",
        )

    rows = []

    for (layer_idx, emotion), df_group in joint_df.groupby(
        ["layer_idx", "target_emotion"]
    ):
        if emotion not in emotions:
            continue

        selected = choose_min_k_for_layer_emotion(
            df_group,
            min_signed_drop=min_signed_drop,
            coverage_ratio=coverage_ratio,
        )

        if selected is None:
            rows.append(
                {
                    "layer_idx": int(layer_idx),
                    "target_emotion": emotion,
                    "selected_top_k": 0,
                    "selected_mean_signed_drop": np.nan,
                    "max_signed_drop": np.nan,
                    "coverage_threshold": np.nan,
                    "coverage_ratio_actual": np.nan,
                    "feature_idx": np.nan,
                    "selection_status": "no_k_passed_threshold",
                }
            )
            continue

        selected_features = parse_feature_list(selected["features"])

        for feature_idx in selected_features:
            matched = feature_df[
                (feature_df["layer_idx"] == int(layer_idx))
                & (feature_df["feature_idx"] == int(feature_idx))
            ]

            if len(matched) == 0:
                continue

            feature_row = matched.iloc[0]

            out = {
                "layer_idx": int(layer_idx),
                "target_emotion": emotion,
                "selected_top_k": int(selected["top_k"]),
                "selected_mean_signed_drop": float(selected["mean_signed_drop"]),
                "max_signed_drop": float(selected["max_signed_drop"]),
                "coverage_threshold": float(selected["coverage_threshold"]),
                "coverage_ratio_actual": float(selected["coverage_ratio_actual"]),
                "feature_idx": int(feature_idx),
                "selection_status": "selected",
            }

            if "entropy_group" in feature_row.index:
                out["entropy_group"] = feature_row["entropy_group"]

            for e in emotions:
                out[f"logit_{e}"] = feature_row[f"logit_{e}"]

            if has_description:
                out["description"] = feature_row["description"]

            rows.append(out)

    result_df = pd.DataFrame(rows)

    if len(result_df) > 0:
        selected_only = result_df[result_df["selection_status"] == "selected"].copy()

        if len(selected_only) > 0:
            shared_map = (
                selected_only
                .groupby(["layer_idx", "feature_idx"])["target_emotion"]
                .apply(lambda x: ",".join(sorted(set(x), key=emotions.index)))
                .reset_index()
                .rename(columns={"target_emotion": "selected_emotions_for_feature"})
            )

            count_map = (
                selected_only
                .groupby(["layer_idx", "feature_idx"])["target_emotion"]
                .nunique()
                .reset_index()
                .rename(columns={"target_emotion": "num_selected_emotions_for_feature"})
            )

            result_df = result_df.merge(
                shared_map,
                on=["layer_idx", "feature_idx"],
                how="left",
            )

            result_df = result_df.merge(
                count_map,
                on=["layer_idx", "feature_idx"],
                how="left",
            )

        if "selected_emotions_for_feature" not in result_df.columns:
            result_df["selected_emotions_for_feature"] = ""

        if "num_selected_emotions_for_feature" not in result_df.columns:
            result_df["num_selected_emotions_for_feature"] = 0

        result_df["selected_emotions_for_feature"] = result_df[
            "selected_emotions_for_feature"
        ].fillna("")

        result_df["num_selected_emotions_for_feature"] = result_df[
            "num_selected_emotions_for_feature"
        ].fillna(0).astype(int)

        result_df = result_df.sort_values(
            by=[
                "layer_idx",
                "target_emotion",
                "selected_top_k",
                "num_selected_emotions_for_feature",
                "feature_idx",
            ],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    safe_save_csv(result_df, output_csv_path)

    print("Final selected result shape:", result_df.shape)
    print("Saved to:", output_csv_path)

    return result_df


def build_unique_selected_feature_table(
    selected_feature_df: pd.DataFrame,
    output_csv_path: Optional[str],
    emotions: List[str],
) -> pd.DataFrame:
    """
    Convert the long selected-feature table into one row per unique feature.
    Adds affected_emotions based on target_emotion appearances.
    """
    if selected_feature_df is None or len(selected_feature_df) == 0:
        feature_rows = pd.DataFrame()
        if output_csv_path is not None:
            safe_save_csv(feature_rows, output_csv_path)
        return feature_rows

    required_cols = ["layer_idx", "feature_idx", "target_emotion", "selection_status"]
    missing = [c for c in required_cols if c not in selected_feature_df.columns]
    if missing:
        raise ValueError(f"Missing columns in selected_feature_df: {missing}")

    selected_only = selected_feature_df[
        selected_feature_df["selection_status"] == "selected"
    ].copy()

    if len(selected_only) == 0:
        feature_rows = pd.DataFrame()
        if output_csv_path is not None:
            safe_save_csv(feature_rows, output_csv_path)
        return feature_rows

    affected_map = (
        selected_only
        .groupby(["layer_idx", "feature_idx"])["target_emotion"]
        .apply(lambda x: ",".join(sorted(set(x), key=emotions.index)))
        .reset_index()
        .rename(columns={"target_emotion": "affected_emotions"})
    )

    feature_rows = (
        selected_only
        .drop_duplicates(subset=["layer_idx", "feature_idx"])
        .copy()
    )

    keep_cols = [
        "layer_idx",
        "feature_idx",
    ] + [f"logit_{e}" for e in emotions]

    if "description" in feature_rows.columns:
        keep_cols.append("description")

    existing_keep_cols = [c for c in keep_cols if c in feature_rows.columns]
    feature_rows = feature_rows[existing_keep_cols]

    feature_rows = feature_rows.merge(
        affected_map,
        on=["layer_idx", "feature_idx"],
        how="left",
    )

    feature_rows = feature_rows.sort_values(
        by=["layer_idx", "feature_idx"]
    ).reset_index(drop=True)

    if output_csv_path is not None:
        safe_save_csv(feature_rows, output_csv_path)
        print("Saved unique selected feature table to:", output_csv_path)
        print("Unique selected feature table shape:", feature_rows.shape)

    return feature_rows


def run_topk_selection_pipeline(
    feature_csv_path: str,
    joint_ablation_csv_path: str,
    selected_output_csv_path: str,
    unique_output_csv_path: Optional[str],
    emotions: List[str],
    min_signed_drop: float = 0.1,
    coverage_ratio: float = 0.80,
    entropy_group: Optional[str] = "emotion_specific",
):
    """
    Convenience wrapper:
    1. Select core features from top-k joint ablation.
    2. Save long selected table.
    3. Save one-row-per-feature table with affected_emotions.
    """
    selected_df = select_core_features_from_joint_ablation(
        feature_csv_path=feature_csv_path,
        joint_ablation_csv_path=joint_ablation_csv_path,
        output_csv_path=selected_output_csv_path,
        emotions=emotions,
        min_signed_drop=min_signed_drop,
        coverage_ratio=coverage_ratio,
        entropy_group=entropy_group,
    )

    unique_df = build_unique_selected_feature_table(
        selected_feature_df=selected_df,
        output_csv_path=unique_output_csv_path,
        emotions=emotions,
    )

    return selected_df, unique_df
