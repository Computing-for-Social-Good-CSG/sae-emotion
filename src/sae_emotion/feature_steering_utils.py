"""
feature_steering_utils.py

Utilities for global sparse SAE feature steering.

This module packages the notebook workflow into reusable components:
1. Load selected SAE features from a CSV.
2. Create one trainable multiplier per unique (layer_idx, feature_idx).
3. Apply feature multipliers through SAE encode/decode hooks.
4. Train multipliers on an emotion detection dataset.
5. Evaluate validation/test accuracy, per-label F1, and general-language perplexity.

Expected feature CSV columns:
    layer_idx, feature_idx

Expected emotion dataset columns by default:
    input_text, emotion

If your dataset has raw text in another column, use prepare_emotion_dataframe(...)
or pass train_text_col / train_label_col to run_feature_steering_experiment().
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sae_lens import SAE
except Exception as exc:  # pragma: no cover
    SAE = None
    _SAE_IMPORT_ERROR = exc
else:
    _SAE_IMPORT_ERROR = None


# ============================================================
# Defaults
# ============================================================

DEFAULT_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]


# ============================================================
# Basic helpers
# ============================================================

def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_makedirs_for_file(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: str) -> None:
    safe_makedirs_for_file(path)
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def safe_save_json(obj, path: str) -> None:
    safe_makedirs_for_file(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def zero_shot_template(text: str) -> str:
    return f"In this text: {text}, the emotion implied is:"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SAE config and cache
# ============================================================

def normalize_sae_size(model_name: str, sae_size: Optional[str]) -> str:
    """
    Gemma-2 canonical SAEs and Llama SAEs do not use big/small IDs.
    We still normalize None -> big so get_sae_config() has a valid value.
    """
    if sae_size is None:
        return "big"
    return sae_size


def get_sae_config(model_name: str, sae_size: Optional[str] = "big"):
    sae_size = normalize_sae_size(model_name, sae_size)

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

    elif "Llama-3.1-8B" in model_name or "Llama-3.1-8b" in model_name:
        release = "llama_scope_lxr_8x"
        sae_id_fn = lambda i: f"l{i}r_8x"

    elif "DeepSeek" in model_name:
        raise ValueError("DeepSeek SAE loading is not configured in this utility.")

    else:
        raise ValueError(f"Unsupported model for SAE loading: {model_name}")

    return release, sae_id_fn


class SAECache:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        sae_size: Optional[str] = "big",
    ):
        if SAE is None:
            raise ImportError(
                "Could not import sae_lens.SAE. Install sae_lens first."
            ) from _SAE_IMPORT_ERROR

        self.model_name = model_name
        self.device = device
        self.sae_size = normalize_sae_size(model_name, sae_size)
        self.release, self.sae_id_fn = get_sae_config(model_name, self.sae_size)
        self.cache: Dict[int, SAE] = {}

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
        print(f"Loaded SAE: release={self.release}, sae_id={sae_id}")
        return sae


# ============================================================
# Dataset helpers
# ============================================================

def prepare_emotion_dataframe(
    csv_path: str,
    text_col: str = "input_text",
    label_col: str = "emotion",
    output_text_col: str = "input_text",
    output_label_col: str = "emotion",
    apply_prompt_template: bool = False,
    prompt_fn=zero_shot_template,
    sep: str = ",",
    keep_emotions: Optional[List[str]] = None,
    drop_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load an emotion dataset and return a dataframe with columns:
        input_text, emotion
    """
    df = pd.read_csv(csv_path, sep=sep)

    if text_col not in df.columns:
        raise ValueError(f"text_col={text_col} not found in {csv_path}. Columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"label_col={label_col} not found in {csv_path}. Columns: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna().copy()
    df = df.rename(columns={text_col: output_text_col, label_col: output_label_col})
    df[output_label_col] = df[output_label_col].astype(str).str.lower().str.strip()
    df[output_text_col] = df[output_text_col].astype(str)

    if drop_labels is not None:
        drop_set = {str(x).lower().strip() for x in drop_labels}
        df = df[~df[output_label_col].isin(drop_set)].copy()

    if keep_emotions is not None:
        keep_set = {str(x).lower().strip() for x in keep_emotions}
        df = df[df[output_label_col].isin(keep_set)].copy()

    if apply_prompt_template:
        df[output_text_col] = df[output_text_col].apply(prompt_fn)

    return df.reset_index(drop=True)


def split_and_balance_train_val(
    df: pd.DataFrame,
    label_col: str = "emotion",
    val_size: float = 0.2,
    max_per_class: Optional[int] = 450,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df[label_col],
    )

    if max_per_class is not None:
        train_df = pd.concat(
            [
                group.sample(n=min(len(group), max_per_class), random_state=seed)
                for _, group in train_df.groupby(label_col)
            ],
            ignore_index=True,
        )
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_general_texts_wikitext(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    cache_dir: Optional[str] = None,
    min_chars: int = 100,
    max_texts: int = 500,
) -> List[str]:
    from datasets import load_dataset

    wiki = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=cache_dir,
    )
    texts = [t for t in wiki["text"] if len(str(t).strip()) > min_chars]
    return texts[:max_texts]


def load_general_texts_from_csv(
    csv_path: str,
    text_col: str,
    sep: str = ",",
    min_chars: int = 100,
    max_texts: Optional[int] = 500,
) -> List[str]:
    df = pd.read_csv(csv_path, sep=sep)
    if text_col not in df.columns:
        raise ValueError(f"text_col={text_col} not found in {csv_path}.")
    texts = [str(t) for t in df[text_col].dropna().tolist() if len(str(t).strip()) > min_chars]
    if max_texts is not None:
        texts = texts[:max_texts]
    return texts


# ============================================================
# Feature optimizer
# ============================================================

class FeatureOptimizer(nn.Module):
    def __init__(
        self,
        feature_circuits: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        feature_csv_path: Optional[str] = None,
        layer_filter: Optional[List[int]] = None,
        init_value: float = 1.0,
        min_multiplier: Optional[float] = None,
        max_multiplier: Optional[float] = None,
    ):
        """
        Create one trainable multiplier per unique SAE feature.

        You can constrain multipliers by passing min_multiplier/max_multiplier.
        Constraints are applied after each optimizer step by clamp_parameters().
        """
        super().__init__()

        if feature_circuits is None and feature_csv_path is None:
            raise ValueError("Provide either feature_circuits or feature_csv_path.")

        all_features: List[Tuple[int, int]] = []

        if feature_circuits is not None:
            for _, features in feature_circuits.items():
                for layer_idx, feature_idx in features:
                    all_features.append((int(layer_idx), int(feature_idx)))

        if feature_csv_path is not None:
            df = pd.read_csv(feature_csv_path)
            required_cols = {"layer_idx", "feature_idx"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns in feature CSV: {missing}")
            df["layer_idx"] = df["layer_idx"].astype(int)
            df["feature_idx"] = df["feature_idx"].astype(int)
            all_features.extend(list(zip(df["layer_idx"].tolist(), df["feature_idx"].tolist())))

        if layer_filter is not None:
            layer_set = {int(x) for x in layer_filter}
            all_features = [(l, f) for l, f in all_features if l in layer_set]

        all_features = sorted(set(all_features))
        if len(all_features) == 0:
            raise ValueError("No features left after deduplication/layer_filter.")

        self.all_features = all_features
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

        layer_features = defaultdict(list)
        for layer_idx, feature_idx in all_features:
            layer_features[int(layer_idx)].append(int(feature_idx))

        self.layer_features = {
            int(layer_idx): sorted(set(feature_indices))
            for layer_idx, feature_indices in layer_features.items()
        }

        self.multipliers = nn.ParameterDict()
        self.num_train_parameters = 0

        for layer_idx in sorted(self.layer_features.keys()):
            feature_indices = self.layer_features[layer_idx]
            n_features = len(feature_indices)
            self.multipliers[str(layer_idx)] = nn.Parameter(
                torch.full((n_features,), float(init_value), dtype=torch.float32)
            )
            self.num_train_parameters += n_features

    def get_layer_params(self, layer_idx: int):
        layer_idx = int(layer_idx)
        layer_key = str(layer_idx)
        if layer_key in self.multipliers:
            return self.multipliers[layer_key], self.layer_features[layer_idx]
        return None, []

    def get_num_trainable_parameters(self) -> int:
        return int(self.num_train_parameters)

    def get_all_features(self) -> List[Tuple[int, int]]:
        return list(self.all_features)

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        if self.min_multiplier is None and self.max_multiplier is None:
            return
        for param in self.multipliers.values():
            param.clamp_(
                min=self.min_multiplier if self.min_multiplier is not None else -float("inf"),
                max=self.max_multiplier if self.max_multiplier is not None else float("inf"),
            )

    def print_summary(self) -> None:
        print(f"Total unique trainable parameters: {self.num_train_parameters}")
        print("Features by layer:")
        for layer_idx in sorted(self.layer_features.keys()):
            print(f"  Layer {layer_idx}: {len(self.layer_features[layer_idx])}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for layer_key, param in self.multipliers.items():
            layer_idx = int(layer_key)
            feature_indices = self.layer_features[layer_idx]
            values = param.detach().float().cpu().numpy().tolist()
            for feature_idx, multiplier in zip(feature_indices, values):
                rows.append(
                    {
                        "layer_idx": layer_idx,
                        "feature_idx": int(feature_idx),
                        "multiplier": float(multiplier),
                        "delta_from_1": float(multiplier - 1.0),
                    }
                )
        return pd.DataFrame(rows).sort_values(["layer_idx", "feature_idx"]).reset_index(drop=True)


# ============================================================
# SAE Feature Finetuner
# ============================================================

class SAEFeatureFinetuner:
    def __init__(
        self,
        model_name: str,
        feature_csv_path: str,
        emotions: Optional[List[str]] = None,
        target_logits: Optional[List[int]] = None,
        layer_filter: Optional[List[int]] = None,
        sae_size: Optional[str] = "big",
        feature_circuits: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        lambda_kl: float = 0.01,
        lambda_l1: float = 0.005,
        temperature: float = 2.0,
        lr: float = 1e-2,
        use_masked_kl: bool = False,
        label_smoothing: float = 0.1,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        device_map: Union[str, Dict] = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        init_multiplier: float = 1.0,
        min_multiplier: Optional[float] = None,
        max_multiplier: Optional[float] = None,
        max_length: int = 512,
    ):
        if emotions is None:
            emotions = DEFAULT_EMOTIONS
        self.emotions = [str(e).lower().strip() for e in emotions]

        self.model_name = model_name
        self.feature_csv_path = feature_csv_path
        self.sae_size = normalize_sae_size(model_name, sae_size)
        self.lambda_kl = float(lambda_kl)
        self.lambda_l1 = float(lambda_l1)
        self.temperature = float(temperature)
        self.use_masked_kl = bool(use_masked_kl)
        self.label_smoothing = float(label_smoothing)
        self.cache_dir = cache_dir
        self.device = device
        self.max_length = int(max_length)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Important for decoder-only models such as Llama/Gemma.
        # Most code below classifies from the final non-padding token.
        # Left padding makes logits[:, -1, :] safe, and the explicit
        # final-token indexing helpers below keep this robust even if a
        # tokenizer/model later uses right padding.
        self.tokenizer.padding_side = "left"

        # Set during forward_with_interventions() so hooks know which
        # sequence position is the final real token for each example.
        self._current_final_token_indices = None

        if target_logits is None:
            target_logits = self.build_emotion_token_ids(self.emotions)
        self.target_logits = [int(x) for x in target_logits]

        self.emotion_to_index = {emotion: i for i, emotion in enumerate(self.emotions)}
        self.index_to_emotion = {i: emotion for emotion, i in self.emotion_to_index.items()}

        if layer_filter is None:
            layer_filter = self.infer_layer_filter_from_feature_csv(feature_csv_path)
        self.layer_filter = [int(x) for x in layer_filter]

        self.feature_optimizer = FeatureOptimizer(
            feature_circuits=feature_circuits,
            feature_csv_path=feature_csv_path,
            layer_filter=self.layer_filter,
            init_value=init_multiplier,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        ).to(self.device)
        self.feature_optimizer.print_summary()

        self.sae_cache = SAECache(
            model_name=model_name,
            device=device,
            sae_size=self.sae_size,
        )
        self.saes = {}
        for layer_idx in sorted(self.feature_optimizer.layer_features.keys()):
            self.saes[int(layer_idx)] = self.sae_cache.load(int(layer_idx))

        self.layers = self.get_transformer_layers(self.model)

        self.optimizer = torch.optim.AdamW(
            self.feature_optimizer.parameters(),
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
        )

    def build_emotion_token_ids(self, emotions: List[str]) -> List[int]:
        """Build emotion target token IDs in the same way as logit_ablation_utils.

        The existing Gemma/Llama ablation code uses:
            tokenizer.encode(" " + emotion)[-1]

        We intentionally keep that convention here so feature steering scores
        the same target token as the working ablation and prediction code.
        """
        token_ids = []
        print("Emotion target tokens:")
        for emotion in emotions:
            # Match the working NNsight utility: encode with tokenizer defaults
            # and use the last token in the encoded label string.
            ids = self.tokenizer.encode(" " + emotion)
            if len(ids) == 0:
                raise ValueError(f"Could not tokenize emotion label: {emotion}")

            token_id = int(ids[-1])
            token_ids.append(token_id)
            pieces = [self.tokenizer.decode([int(x)]) for x in ids]
            print(
                f"  {emotion}: {ids} -> pieces={pieces} -> "
                f"using last token {token_id} ({self.tokenizer.decode([token_id])!r})"
            )

        if len(set(token_ids)) != len(token_ids):
            pairs = list(zip(emotions, token_ids))
            raise ValueError(
                "Some emotion labels map to the same target token ID: "
                f"{pairs}. Pass explicit target_logits if needed."
            )

        return token_ids

    @staticmethod
    def infer_layer_filter_from_feature_csv(feature_csv_path: str) -> List[int]:
        df = pd.read_csv(feature_csv_path)
        if "layer_idx" not in df.columns:
            raise ValueError("feature_csv_path must contain layer_idx.")
        return sorted(df["layer_idx"].astype(int).unique().tolist())

    @staticmethod
    def get_transformer_layers(model):
        candidates = [
            ("model.layers", lambda m: m.model.layers),
            ("model.language_model.layers", lambda m: m.model.language_model.layers),
            ("language_model.layers", lambda m: m.language_model.layers),
            ("transformer.h", lambda m: m.transformer.h),
            ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ]
        for name, getter in candidates:
            try:
                layers = getter(model)
                _ = layers[0]
                print(f"Using transformer layers from {name}")
                return layers
            except Exception:
                continue
        raise ValueError("Could not locate transformer layers for this model.")

    def tokenize(self, input_texts: List[str]):
        return self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    @staticmethod
    def _final_token_indices_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the last non-padding token index for each row.

        This is robust to both left padding and right padding.
        Do not use attention_mask.sum(dim=1) - 1 here: that only works for
        right-padded batches and is wrong for left-padded Llama/Gemma inputs.
        """
        mask = attention_mask.long()
        seq_positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
        final_indices = (mask * seq_positions).max(dim=1).values
        return final_indices.long()

    @staticmethod
    def _gather_final_token_logits(logits: torch.Tensor, final_token_indices: torch.Tensor) -> torch.Tensor:
        """Gather logits at each example's final real token."""
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        return logits[batch_idx, final_token_indices.to(logits.device), :]

    def _get_current_final_token_indices(self, hidden: torch.Tensor) -> torch.Tensor:
        """Get final-token indices for hooks, defaulting to -1 if unavailable."""
        if self._current_final_token_indices is None:
            return torch.full(
                (hidden.size(0),),
                hidden.size(1) - 1,
                dtype=torch.long,
                device=hidden.device,
            )
        return self._current_final_token_indices.to(hidden.device)

    def _modify_hidden_with_sae(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        final_token_only: bool,
    ) -> torch.Tensor:
        sae = self.saes[int(layer_idx)]
        mult, feature_indices = self.feature_optimizer.get_layer_params(int(layer_idx))
        if mult is None or len(feature_indices) == 0:
            return hidden

        mult = mult.to(hidden.device).to(torch.float32)
        feature_idx_tensor = torch.tensor(feature_indices, dtype=torch.long, device=hidden.device)

        if final_token_only:
            final_token_indices = self._get_current_final_token_indices(hidden)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)

            target_hidden = hidden[batch_idx, final_token_indices, :]
            sae_acts = sae.encode(target_hidden)
            sae_clean_decode = sae.decode(sae_acts)
            sae_error = target_hidden - sae_clean_decode

            modified_acts = sae_acts.clone()
            modified_acts[:, feature_idx_tensor] = (
                sae_acts[:, feature_idx_tensor] * mult.unsqueeze(0)
            )

            reconstructed = sae.decode(modified_acts) + sae_error
            reconstructed = reconstructed.to(hidden.dtype)
            new_hidden = hidden.clone()
            new_hidden[batch_idx, final_token_indices, :] = reconstructed
            return new_hidden

        sae_acts = sae.encode(hidden)
        sae_clean_decode = sae.decode(sae_acts)
        sae_error = hidden - sae_clean_decode

        modified_acts = sae_acts.clone()
        modified_acts[..., feature_idx_tensor] = (
            sae_acts[..., feature_idx_tensor] * mult.view(*([1] * (sae_acts.ndim - 1)), -1)
        )
        new_hidden = sae.decode(modified_acts) + sae_error
        return new_hidden.to(hidden.dtype)

    def apply_optimized_intervention(
        self,
        layer_idx: int,
        output,
        final_token_only: bool = True,
    ):
        """
        HF decoder layers sometimes return a tensor and sometimes a tuple.
        This function preserves the original output structure.
        """
        if isinstance(output, tuple):
            hidden = output[0]
            modified_hidden = self._modify_hidden_with_sae(layer_idx, hidden, final_token_only)
            return (modified_hidden,) + output[1:]

        hidden = output
        return self._modify_hidden_with_sae(layer_idx, hidden, final_token_only)

    def _register_hooks(self, final_token_only: bool = True):
        hooks = []
        for layer_idx in sorted(self.feature_optimizer.layer_features.keys()):
            def make_hook(current_layer_idx):
                def hook(module, inputs, output):
                    return self.apply_optimized_intervention(
                        current_layer_idx,
                        output,
                        final_token_only=final_token_only,
                    )
                return hook

            hooks.append(self.layers[int(layer_idx)].register_forward_hook(make_hook(int(layer_idx))))
        return hooks

    def forward_with_interventions(
        self,
        input_texts: List[str],
        return_full_logits: bool = False,
        final_token_only: bool = True,
    ):
        inputs = self.tokenize(input_texts)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        final_token_indices = self._final_token_indices_from_attention_mask(attention_mask)

        hooks = self._register_hooks(final_token_only=final_token_only)
        self._current_final_token_indices = final_token_indices
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            full_logits = self._gather_final_token_logits(outputs.logits, final_token_indices)
            target_logits = full_logits[:, self.target_logits]
        finally:
            self._current_final_token_indices = None
            for hook in hooks:
                hook.remove()

        if return_full_logits:
            return target_logits, full_logits, input_ids, attention_mask
        return target_logits

    @torch.no_grad()
    def get_clean_logits(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if attention_mask is None:
            return outputs.logits[:, -1, :]
        final_token_indices = self._final_token_indices_from_attention_mask(attention_mask)
        return self._gather_final_token_logits(outputs.logits, final_token_indices)

    @torch.no_grad()
    def forward_clean(
        self,
        input_texts: List[str],
        return_full_logits: bool = False,
    ):
        """Clean forward pass with no SAE hooks. Useful for debugging baseline accuracy."""
        inputs = self.tokenize(input_texts)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        final_token_indices = self._final_token_indices_from_attention_mask(attention_mask)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        full_logits = self._gather_final_token_logits(outputs.logits, final_token_indices)
        target_logits = full_logits[:, self.target_logits]
        if return_full_logits:
            return target_logits, full_logits, input_ids, attention_mask
        return target_logits

    @torch.no_grad()
    def predict_clean(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[List[str], List[str], np.ndarray]:
        """Predict without SAE hooks. This is the true clean model baseline."""
        self.model.eval()
        all_preds: List[str] = []
        all_labels: List[str] = []
        all_logits = []

        for i in tqdm(range(0, len(df), batch_size), desc="predict clean", leave=False):
            batch_df = df.iloc[i:i + batch_size]
            input_texts = batch_df["input_text"].tolist()
            labels = [str(x).lower().strip() for x in batch_df["emotion"].tolist()]

            logits = self.forward_clean(input_texts)
            pred_indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            preds = [self.index_to_emotion[int(idx)] for idx in pred_indices]

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_logits.append(logits.detach().float().cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, len(self.emotions)))
        return all_labels, all_preds, logits_np

    @torch.no_grad()
    def evaluate_clean(self, df: pd.DataFrame, batch_size: int = 32) -> Dict[str, float]:
        """True clean model accuracy/F1 with no SAE hooks."""
        y_true, y_pred, _ = self.predict_clean(df, batch_size=batch_size)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=self.emotions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=self.emotions, average="weighted", zero_division=0)),
        }

    def compute_kl_loss(self, clean_logits, intervened_logits):
        clean_probs = F.softmax(clean_logits / self.temperature, dim=-1).detach()
        intervened_log_probs = F.log_softmax(intervened_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(intervened_log_probs, clean_probs, reduction="batchmean")
        return kl_loss * (self.temperature ** 2)

    def compute_kl_loss_masked(self, clean_logits, intervened_logits):
        mask = torch.ones(clean_logits.size(-1), dtype=torch.bool, device=clean_logits.device)
        mask[self.target_logits] = False
        clean_masked = clean_logits[:, mask]
        intervened_masked = intervened_logits[:, mask]
        clean_probs = F.softmax(clean_masked / self.temperature, dim=-1).detach()
        intervened_log_probs = F.log_softmax(intervened_masked / self.temperature, dim=-1)
        kl_loss = F.kl_div(intervened_log_probs, clean_probs, reduction="batchmean")
        return kl_loss * (self.temperature ** 2)

    def train_step(self, batch: Dict[str, List[str]]) -> Dict[str, float]:
        self.feature_optimizer.train()
        input_texts = batch["input_texts"]
        label_texts = [str(x).lower().strip() for x in batch["label_texts"]]

        labels = torch.tensor(
            [self.emotion_to_index[label] for label in label_texts],
            device=self.device,
            dtype=torch.long,
        )

        target_logits, full_logits, input_ids, attention_mask = self.forward_with_interventions(
            input_texts,
            return_full_logits=True,
            final_token_only=True,
        )

        with torch.no_grad():
            clean_logits = self.get_clean_logits(input_ids=input_ids, attention_mask=attention_mask)

        cls_loss = F.cross_entropy(
            target_logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        if self.use_masked_kl:
            kl_loss = self.compute_kl_loss_masked(clean_logits, full_logits)
        else:
            kl_loss = self.compute_kl_loss(clean_logits, full_logits)

        l1_loss = torch.zeros((), device=self.device)
        for param in self.feature_optimizer.parameters():
            l1_loss = l1_loss + torch.sum(torch.abs(param - 1.0))

        total_loss = cls_loss + self.lambda_kl * kl_loss + self.lambda_l1 * l1_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.feature_optimizer.parameters(), 1.0)
        self.optimizer.step()
        self.feature_optimizer.clamp_parameters()

        result = {
            "cls_loss": float(cls_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "l1_loss": float(l1_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }

        del target_logits, full_logits, clean_logits, input_ids, attention_mask
        del labels, cls_loss, kl_loss, l1_loss, total_loss
        cleanup_memory()
        return result

    def train_epoch(self, df: pd.DataFrame, batch_size: int = 32) -> Dict[str, float]:
        self.feature_optimizer.train()
        metrics = defaultdict(list)
        shuffled = df.sample(frac=1.0).reset_index(drop=True)

        for i in tqdm(range(0, len(shuffled), batch_size), desc="train", leave=False):
            batch_df = shuffled.iloc[i:i + batch_size]
            batch = {
                "input_texts": batch_df["input_text"].tolist(),
                "label_texts": batch_df["emotion"].tolist(),
            }
            step_metrics = self.train_step(batch)
            for k, v in step_metrics.items():
                metrics[k].append(v)

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[List[str], List[str], np.ndarray]:
        self.feature_optimizer.eval()
        all_preds: List[str] = []
        all_labels: List[str] = []
        all_logits = []

        for i in tqdm(range(0, len(df), batch_size), desc="predict", leave=False):
            batch_df = df.iloc[i:i + batch_size]
            input_texts = batch_df["input_text"].tolist()
            labels = [str(x).lower().strip() for x in batch_df["emotion"].tolist()]

            logits = self.forward_with_interventions(input_texts, final_token_only=True)
            pred_indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            preds = [self.index_to_emotion[int(idx)] for idx in pred_indices]

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_logits.append(logits.detach().float().cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, len(self.emotions)))
        return all_labels, all_preds, logits_np

    @torch.no_grad()
    def evaluate(self, df: pd.DataFrame, batch_size: int = 32) -> Dict[str, float]:
        y_true, y_pred, _ = self.predict(df, batch_size=batch_size)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=self.emotions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=self.emotions, average="weighted", zero_division=0)),
        }

    @torch.no_grad()
    def evaluate_per_label(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[Dict, str]:
        y_true, y_pred, _ = self.predict(df, batch_size=batch_size)
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=self.emotions,
            output_dict=True,
            zero_division=0,
        )
        report_str = classification_report(
            y_true,
            y_pred,
            labels=self.emotions,
            zero_division=0,
        )
        print(report_str)
        return report_dict, report_str

    @torch.no_grad()
    def evaluate_perplexity_clean(self, texts: List[str], batch_size: int = 8) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="ppl clean", leave=False):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenize(batch_texts)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            num_tokens = int((labels != -100).sum().item())
            total_loss += float(outputs.loss.item()) * num_tokens
            total_tokens += num_tokens

            del input_ids, attention_mask, labels, outputs
            cleanup_memory()

        avg_loss = total_loss / max(total_tokens, 1)
        return {"perplexity": float(math.exp(avg_loss)), "avg_loss": float(avg_loss), "total_tokens": int(total_tokens)}

    @torch.no_grad()
    def evaluate_perplexity(self, texts: List[str], batch_size: int = 8) -> Dict[str, float]:
        self.feature_optimizer.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="ppl steered", leave=False):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenize(batch_texts)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            hooks = self._register_hooks(final_token_only=False)
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            finally:
                for hook in hooks:
                    hook.remove()

            num_tokens = int((labels != -100).sum().item())
            total_loss += float(outputs.loss.item()) * num_tokens
            total_tokens += num_tokens

            del input_ids, attention_mask, labels, outputs
            cleanup_memory()

        avg_loss = total_loss / max(total_tokens, 1)
        return {"perplexity": float(math.exp(avg_loss)), "avg_loss": float(avg_loss), "total_tokens": int(total_tokens)}

    def save_state(self, path: str) -> None:
        safe_makedirs_for_file(path)
        torch.save(
            {
                "feature_optimizer_state_dict": self.feature_optimizer.state_dict(),
                "model_name": self.model_name,
                "feature_csv_path": self.feature_csv_path,
                "sae_size": self.sae_size,
                "emotions": self.emotions,
                "target_logits": self.target_logits,
                "layer_filter": self.layer_filter,
                "lambda_kl": self.lambda_kl,
                "lambda_l1": self.lambda_l1,
                "temperature": self.temperature,
                "use_masked_kl": self.use_masked_kl,
                "label_smoothing": self.label_smoothing,
            },
            path,
        )

    def load_state(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.feature_optimizer.load_state_dict(state["feature_optimizer_state_dict"])

    def export_multipliers(
        self,
        output_csv_path: str,
        description_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.feature_optimizer.to_dataframe()

        if description_csv_path is not None and os.path.exists(description_csv_path):
            desc_df = pd.read_csv(description_csv_path)
            keep_cols = [c for c in ["layer_idx", "feature_idx", "description", "affected_emotions"] if c in desc_df.columns]
            if {"layer_idx", "feature_idx"}.issubset(keep_cols):
                desc_df = desc_df[keep_cols].drop_duplicates(["layer_idx", "feature_idx"])
                desc_df["layer_idx"] = desc_df["layer_idx"].astype(int)
                desc_df["feature_idx"] = desc_df["feature_idx"].astype(int)
                df = df.merge(desc_df, on=["layer_idx", "feature_idx"], how="left")

        safe_save_csv(df, output_csv_path)
        return df


# ============================================================
# Training loop and experiment runner
# ============================================================

@dataclass
class TestDatasetSpec:
    name: str
    path: str
    text_col: str = "input_text"
    label_col: str = "emotion"
    sep: str = ","
    apply_prompt_template: bool = False
    drop_labels: Optional[List[str]] = None


@dataclass
class FeatureSteeringConfig:
    model_name: str
    feature_csv_path: str
    train_csv_path: str
    output_dir: str

    sae_size: Optional[str] = "big"
    emotions: List[str] = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    target_logits: Optional[List[int]] = None
    layer_filter: Optional[List[int]] = None

    train_text_col: str = "input_text"
    train_label_col: str = "emotion"
    train_sep: str = ","
    train_apply_prompt_template: bool = True
    val_size: float = 0.2
    max_per_class: Optional[int] = 450
    seed: int = 42

    num_epochs: int = 5
    num_steps: Optional[int] = None
    eval_every_steps: Optional[int] = None
    batch_size: int = 16
    eval_batch_size: int = 32
    ppl_batch_size: int = 8
    lr: float = 1e-2
    lambda_kl: float = 0.01
    lambda_l1: float = 0.005
    temperature: float = 2.0
    use_masked_kl: bool = False
    label_smoothing: float = 0.1
    init_multiplier: float = 1.0
    min_multiplier: Optional[float] = None
    max_multiplier: Optional[float] = None

    cache_dir: Optional[str] = None
    device: str = "cuda"
    device_map: Union[str, Dict] = "cuda"
    torch_dtype: str = "bfloat16"
    max_length: int = 512

    general_texts: Optional[List[str]] = None
    general_text_csv_path: Optional[str] = None
    general_text_col: Optional[str] = None
    general_text_sep: str = ","
    use_wikitext: bool = True
    wikitext_cache_dir: Optional[str] = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
    max_general_texts: int = 500

    # Epoch-level reporting controls. Evaluation snapshots are saved at
    # epoch 0 before training and after every epoch. Dataset perplexity can
    # be expensive, so max_eval_ppl_texts caps the number of texts per dataset.
    eval_main_full: bool = True
    eval_main_val: bool = True
    eval_general_ppl: bool = True
    eval_dataset_ppl: bool = True
    max_eval_ppl_texts: Optional[int] = 500
    save_epoch_predictions: bool = False

    test_datasets: List[TestDatasetSpec] = field(default_factory=list)

    # When supplied, use the saved train/held-out rows without splitting or
    # balancing again. main_csv_path retains the full-corpus diagnostic.
    val_csv_path: Optional[str] = None
    main_csv_path: Optional[str] = None
    test_csv_path: Optional[str] = None


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[key]


def _save_epoch_multipliers(
    finetuner: SAEFeatureFinetuner,
    output_dir: str,
    epoch: int,
    step: int,
) -> pd.DataFrame:
    """Save feature multipliers for a snapshot.

    For ordinary epoch-level evaluation, the legacy filename
    epoch_XXX_multipliers.csv is still written. For step-level evaluation,
    an additional step-specific file is also written so multiple evaluations
    inside the same epoch do not overwrite one another.
    """
    multipliers_dir = os.path.join(output_dir, "epoch_multipliers")
    os.makedirs(multipliers_dir, exist_ok=True)

    df = finetuner.feature_optimizer.to_dataframe()
    df.insert(0, "step", int(step))
    df.insert(0, "epoch", int(epoch))

    legacy_path = os.path.join(
        multipliers_dir,
        f"epoch_{int(epoch):03d}_multipliers.csv",
    )
    step_path = os.path.join(
        multipliers_dir,
        f"epoch_{int(epoch):03d}_step_{int(step):06d}_multipliers.csv",
    )

    safe_save_csv(df, legacy_path)
    safe_save_csv(df, step_path)
    return df


def _evaluate_dataset_snapshot(
    finetuner: SAEFeatureFinetuner,
    df: pd.DataFrame,
    dataset_name: str,
    epoch: int,
    step: int,
    output_dir: str,
    batch_size: int = 32,
    ppl_batch_size: int = 8,
    eval_dataset_ppl: bool = True,
    max_eval_ppl_texts: Optional[int] = 500,
    save_predictions: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate one dataset and save compact epoch-level reports."""
    epoch_dir = os.path.join(output_dir, "epoch_eval_reports", f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    y_true, y_pred, logits = finetuner.predict(df, batch_size=batch_size)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=finetuner.emotions,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=finetuner.emotions,
        zero_division=0,
    )

    if eval_dataset_ppl:
        ppl_texts = df["input_text"].astype(str).tolist()
        if max_eval_ppl_texts is not None:
            ppl_texts = ppl_texts[: int(max_eval_ppl_texts)]
        ppl_metrics = finetuner.evaluate_perplexity(
            ppl_texts,
            batch_size=ppl_batch_size,
        ) if len(ppl_texts) > 0 else {"perplexity": np.nan, "avg_loss": np.nan, "total_tokens": 0}
    else:
        ppl_metrics = {"perplexity": np.nan, "avg_loss": np.nan, "total_tokens": 0}

    metrics = {
        "epoch": int(epoch),
        "step": int(step),
        "dataset": dataset_name,
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=finetuner.emotions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=finetuner.emotions, average="weighted", zero_division=0)),
        "perplexity": float(ppl_metrics["perplexity"]),
        "ppl_avg_loss": float(ppl_metrics["avg_loss"]),
        "ppl_total_tokens": int(ppl_metrics["total_tokens"]),
    }

    per_emotion_rows = []
    for emotion in finetuner.emotions:
        emotion_report = report_dict.get(emotion, {})
        per_emotion_rows.append(
            {
                "epoch": int(epoch),
                "step": int(step),
                "dataset": dataset_name,
                "emotion": emotion,
                "precision": float(emotion_report.get("precision", 0.0)),
                "recall": float(emotion_report.get("recall", 0.0)),
                "f1": float(emotion_report.get("f1-score", 0.0)),
                "support": int(emotion_report.get("support", 0)),
            }
        )

    safe_save_json(metrics, os.path.join(epoch_dir, f"{dataset_name}_metrics.json"))
    safe_save_json(report_dict, os.path.join(epoch_dir, f"{dataset_name}_classification_report.json"))
    with open(os.path.join(epoch_dir, f"{dataset_name}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)

    if save_predictions:
        pred_df = df.copy().reset_index(drop=True)
        pred_df["pred_emotion"] = y_pred
        for j, emotion in enumerate(finetuner.emotions):
            pred_df[f"logit_{emotion}"] = logits[:, j]
        safe_save_csv(pred_df, os.path.join(epoch_dir, f"{dataset_name}_predictions.csv"))

    print(
        f"[epoch={epoch:03d} step={step}] {dataset_name}: "
        f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}, "
        f"ppl={metrics['perplexity']:.4f}"
    )
    return metrics, per_emotion_rows


def _append_and_save_epoch_reports(
    metric_rows: List[Dict[str, float]],
    per_emotion_rows: List[Dict[str, float]],
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.DataFrame(metric_rows)
    per_emotion_df = pd.DataFrame(per_emotion_rows)
    safe_save_csv(metrics_df, os.path.join(output_dir, "epoch_metrics_long.csv"))
    safe_save_csv(per_emotion_df, os.path.join(output_dir, "epoch_per_emotion_f1.csv"))
    return metrics_df, per_emotion_df


def train_feature_steering(
    finetuner: SAEFeatureFinetuner,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    general_texts: Optional[List[str]] = None,
    eval_datasets: Optional[Dict[str, pd.DataFrame]] = None,
    num_epochs: int = 5,
    num_steps: Optional[int] = None,
    eval_every_steps: Optional[int] = None,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    ppl_batch_size: int = 8,
    output_dir: str = "feature_steering_outputs",
    eval_general_ppl: bool = True,
    eval_dataset_ppl: bool = True,
    max_eval_ppl_texts: Optional[int] = 500,
    save_epoch_predictions: bool = False,
) -> Tuple[SAEFeatureFinetuner, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train feature multipliers and save evaluation reports.

    Supports both:
      1. epoch-level evaluation, when eval_every_steps is None;
      2. step-level evaluation, when eval_every_steps is an integer.

    Saved files include:
      - training_history.csv
      - epoch_metrics_long.csv
      - epoch_per_emotion_f1.csv
      - epoch_multipliers/epoch_XXX_multipliers.csv
      - epoch_multipliers/epoch_XXX_step_YYYYYY_multipliers.csv
      - epoch_eval_reports/epoch_XXX/*
    """
    os.makedirs(output_dir, exist_ok=True)

    if num_steps is not None and int(num_steps) <= 0:
        raise ValueError("num_steps must be positive when provided.")
    if eval_every_steps is not None and int(eval_every_steps) <= 0:
        raise ValueError("eval_every_steps must be positive when provided.")
    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    eval_datasets = eval_datasets or {"main_val": val_df}
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    steps_per_epoch = int(math.ceil(len(train_df) / batch_size))
    max_steps = int(num_steps) if num_steps is not None else max(num_epochs, 0) * steps_per_epoch

    best_val_macro_f1 = -1.0
    best_path = os.path.join(output_dir, "best_feature_steering.pt")
    history: List[Dict[str, float]] = []
    epoch_metric_rows: List[Dict[str, float]] = []
    epoch_per_emotion_rows: List[Dict[str, float]] = []
    last_eval_step = -1

    def run_epoch_snapshot(epoch: int, step: int, train_metrics: Optional[Dict[str, float]] = None) -> None:
        nonlocal best_val_macro_f1, epoch_metric_rows, epoch_per_emotion_rows, last_eval_step

        _save_epoch_multipliers(finetuner, output_dir=output_dir, epoch=epoch, step=step)

        if eval_general_ppl and general_texts is not None and len(general_texts) > 0:
            general_ppl = finetuner.evaluate_perplexity(general_texts, batch_size=ppl_batch_size)
        else:
            general_ppl = {"perplexity": np.nan, "avg_loss": np.nan, "total_tokens": 0}

        snapshot_metrics = []
        for dataset_name, dataset_df in eval_datasets.items():
            metrics, per_emotion = _evaluate_dataset_snapshot(
                finetuner=finetuner,
                df=dataset_df,
                dataset_name=dataset_name,
                epoch=epoch,
                step=step,
                output_dir=output_dir,
                batch_size=eval_batch_size,
                ppl_batch_size=ppl_batch_size,
                eval_dataset_ppl=eval_dataset_ppl,
                max_eval_ppl_texts=max_eval_ppl_texts,
                save_predictions=save_epoch_predictions,
            )
            metrics["general_perplexity"] = float(general_ppl["perplexity"])
            metrics["general_ppl_avg_loss"] = float(general_ppl["avg_loss"])
            metrics["general_ppl_total_tokens"] = int(general_ppl["total_tokens"])
            snapshot_metrics.append(metrics)
            epoch_metric_rows.append(metrics)
            epoch_per_emotion_rows.extend(per_emotion)

        _append_and_save_epoch_reports(epoch_metric_rows, epoch_per_emotion_rows, output_dir)

        main_val_row = next((r for r in snapshot_metrics if r["dataset"] == "main_val"), None)
        if main_val_row is None:
            main_val_row = snapshot_metrics[0] if snapshot_metrics else None

        current_lr = finetuner.optimizer.param_groups[0]["lr"]
        if main_val_row is not None:
            finetuner.scheduler.step(main_val_row["macro_f1"])
            current_lr = finetuner.optimizer.param_groups[0]["lr"]

        row = {
            "epoch": int(epoch),
            "step": int(step),
            "lr": float(current_lr),
            "general_perplexity": float(general_ppl["perplexity"]),
            "general_ppl_avg_loss": float(general_ppl["avg_loss"]),
            "general_ppl_total_tokens": int(general_ppl["total_tokens"]),
        }
        if train_metrics is not None:
            row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        if main_val_row is not None:
            row.update({
                "main_val_accuracy": main_val_row["accuracy"],
                "main_val_macro_f1": main_val_row["macro_f1"],
                "main_val_weighted_f1": main_val_row["weighted_f1"],
                "main_val_perplexity": main_val_row["perplexity"],
            })
        history.append(row)
        safe_save_csv(pd.DataFrame(history), os.path.join(output_dir, "training_history.csv"))

        if main_val_row is not None and main_val_row["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(main_val_row["macro_f1"])
            finetuner.save_state(best_path)
            print(f"Saved new best model to {best_path} with main_val_macro_f1={best_val_macro_f1:.4f}")

        last_eval_step = int(step)

    # Step 0, before any training.
    print("\nStep 0 pre-training evaluation")
    print("-" * 60)
    run_epoch_snapshot(epoch=0, step=0, train_metrics=None)

    global_step = 0
    pbar = tqdm(total=max_steps, desc="train steps")
    rolling_train_metrics = defaultdict(list)

    for epoch_idx in range(1, num_epochs + 1):
        if global_step >= max_steps:
            break

        shuffled = train_df.sample(frac=1.0, random_state=None).reset_index(drop=True)
        print(f"\nEpoch {epoch_idx}/{num_epochs}")
        print("-" * 60)
        epoch_train_metrics = defaultdict(list)

        for start_idx in range(0, len(shuffled), batch_size):
            if global_step >= max_steps:
                break

            batch_df = shuffled.iloc[start_idx:start_idx + batch_size]
            batch = {
                "input_texts": batch_df["input_text"].tolist(),
                "label_texts": batch_df["emotion"].tolist(),
            }
            step_metrics = finetuner.train_step(batch)
            global_step += 1
            pbar.update(1)

            for k, v in step_metrics.items():
                epoch_train_metrics[k].append(v)
                rolling_train_metrics[k].append(v)

            if eval_every_steps is not None and global_step % int(eval_every_steps) == 0:
                train_metrics = {
                    k: float(np.mean(v))
                    for k, v in rolling_train_metrics.items()
                    if len(v) > 0
                }
                print(f"\nStep-level evaluation at step {global_step}")
                print("-" * 60)
                run_epoch_snapshot(epoch=epoch_idx, step=global_step, train_metrics=train_metrics)
                rolling_train_metrics = defaultdict(list)

        if eval_every_steps is None:
            train_metrics = {k: float(np.mean(v)) for k, v in epoch_train_metrics.items()}
            run_epoch_snapshot(epoch=epoch_idx, step=global_step, train_metrics=train_metrics)

    if eval_every_steps is not None and global_step != last_eval_step:
        train_metrics = {
            k: float(np.mean(v))
            for k, v in rolling_train_metrics.items()
            if len(v) > 0
        }
        print(f"\nFinal step-level evaluation at step {global_step}")
        print("-" * 60)
        run_epoch_snapshot(epoch=epoch_idx if 'epoch_idx' in locals() else 0, step=global_step, train_metrics=train_metrics)

    pbar.close()

    if os.path.exists(best_path):
        finetuner.load_state(best_path)
        print(f"Loaded best model from {best_path}")

    history_df = pd.DataFrame(history)
    metrics_df = pd.DataFrame(epoch_metric_rows)
    per_emotion_df = pd.DataFrame(epoch_per_emotion_rows)
    return finetuner, history_df, metrics_df, per_emotion_df

def _load_general_texts_from_config(config: FeatureSteeringConfig) -> Optional[List[str]]:
    if config.general_texts is not None:
        return config.general_texts

    if config.general_text_csv_path is not None:
        if config.general_text_col is None:
            raise ValueError("general_text_col must be provided with general_text_csv_path.")
        return load_general_texts_from_csv(
            csv_path=config.general_text_csv_path,
            text_col=config.general_text_col,
            sep=config.general_text_sep,
            max_texts=config.max_general_texts,
        )

    if config.use_wikitext:
        return load_general_texts_wikitext(
            cache_dir=config.wikitext_cache_dir,
            max_texts=config.max_general_texts,
        )

    return None


def evaluate_and_save_report(
    finetuner: SAEFeatureFinetuner,
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    dataset_dir = os.path.join(output_dir, "eval_reports")
    os.makedirs(dataset_dir, exist_ok=True)

    y_true, y_pred, logits = finetuner.predict(df, batch_size=batch_size)
    metrics = {
        "dataset": dataset_name,
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=finetuner.emotions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=finetuner.emotions, average="weighted", zero_division=0)),
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=finetuner.emotions,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=finetuner.emotions,
        zero_division=0,
    )

    pred_df = df.copy().reset_index(drop=True)
    pred_df["pred_emotion"] = y_pred
    for j, emotion in enumerate(finetuner.emotions):
        pred_df[f"logit_{emotion}"] = logits[:, j]

    safe_save_csv(pred_df, os.path.join(dataset_dir, f"{dataset_name}_predictions.csv"))
    safe_save_json(metrics, os.path.join(dataset_dir, f"{dataset_name}_metrics.json"))
    safe_save_json(report_dict, os.path.join(dataset_dir, f"{dataset_name}_classification_report.json"))
    with open(os.path.join(dataset_dir, f"{dataset_name}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"\n[{dataset_name}]")
    print(report_str)
    print(metrics)
    return metrics


def prepare_training_data(config: FeatureSteeringConfig):
    """Return full, training, and held-out data; preserve explicit split rows.

    A config without val_csv_path retains the original seeded split behavior.
    Explicit splits reject invalid rows instead of silently dropping them.
    """
    if config.val_csv_path is not None and config.main_csv_path is None:
        raise ValueError("Set main_csv_path when supplying val_csv_path.")

    main_path = config.main_csv_path if config.val_csv_path is not None else config.train_csv_path
    main_df = prepare_emotion_dataframe(
        csv_path=main_path,
        text_col=config.train_text_col,
        label_col=config.train_label_col,
        sep=config.train_sep,
        apply_prompt_template=config.train_apply_prompt_template,
        keep_emotions=config.emotions,
    )
    if config.val_csv_path is None:
        train_df, val_df = split_and_balance_train_val(
            main_df, label_col="emotion", val_size=config.val_size,
            max_per_class=config.max_per_class, seed=config.seed,
        )
        return main_df, train_df, val_df

    def load_saved_split(path):
        df = pd.read_csv(path, sep=config.train_sep)
        columns = [config.train_text_col, config.train_label_col]
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing split columns {sorted(missing)} in {path}")
        df = df[columns].copy()
        if df.empty or df.isna().any().any():
            raise ValueError(f"Empty split or missing values in {path}")
        df.columns = ["input_text", "emotion"]
        df["input_text"] = df["input_text"].astype(str)
        df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
        if df["input_text"].str.strip().eq("").any():
            raise ValueError(f"Blank input text in {path}")
        if not df["emotion"].isin(config.emotions).all():
            raise ValueError(f"Unexpected emotion labels in {path}")
        if config.train_apply_prompt_template:
            df["input_text"] = df["input_text"].apply(zero_shot_template)
        return df.reset_index(drop=True)

    return main_df, load_saved_split(config.train_csv_path), load_saved_split(config.val_csv_path)


def run_feature_steering_experiment(config: FeatureSteeringConfig):
    """
    Full runner with epoch-level evaluation.

    This version evaluates at epoch 0 before training and after every epoch.
    It saves per-dataset metrics, per-emotion F1, perplexity, and feature
    multiplier snapshots for every epoch.
    """
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    safe_save_json(asdict(config), os.path.join(config.output_dir, "config.json"))

    train_all_df, train_df, val_df = prepare_training_data(config)

    safe_save_csv(train_all_df, os.path.join(config.output_dir, "main_full_df.csv"))
    safe_save_csv(train_df, os.path.join(config.output_dir, "train_df.csv"))
    safe_save_csv(val_df, os.path.join(config.output_dir, "val_df.csv"))

    print(f"Main full size: {len(train_all_df)}")
    print(f"Train size:     {len(train_df)}")
    print(f"Val size:       {len(val_df)}")
    print("Train label counts:")
    print(train_df["emotion"].value_counts())

    general_texts = _load_general_texts_from_config(config)
    if general_texts is not None:
        print(f"General texts for perplexity: {len(general_texts)}")

    finetuner = SAEFeatureFinetuner(
        model_name=config.model_name,
        feature_csv_path=config.feature_csv_path,
        emotions=config.emotions,
        target_logits=config.target_logits,
        layer_filter=config.layer_filter,
        sae_size=config.sae_size,
        lambda_kl=config.lambda_kl,
        lambda_l1=config.lambda_l1,
        temperature=config.temperature,
        lr=config.lr,
        use_masked_kl=config.use_masked_kl,
        label_smoothing=config.label_smoothing,
        cache_dir=config.cache_dir,
        device=config.device,
        device_map=config.device_map,
        torch_dtype=parse_torch_dtype(config.torch_dtype),
        init_multiplier=config.init_multiplier,
        min_multiplier=config.min_multiplier,
        max_multiplier=config.max_multiplier,
        max_length=config.max_length,
    )

    test_datasets_loaded: List[Tuple[TestDatasetSpec, pd.DataFrame]] = []
    for spec in config.test_datasets:
        test_df = prepare_emotion_dataframe(
            csv_path=spec.path,
            text_col=spec.text_col,
            label_col=spec.label_col,
            sep=spec.sep,
            apply_prompt_template=spec.apply_prompt_template,
            keep_emotions=config.emotions,
            drop_labels=spec.drop_labels,
        )
        safe_save_csv(test_df, os.path.join(config.output_dir, f"{spec.name}_df.csv"))
        test_datasets_loaded.append((spec, test_df))
        print(f"{spec.name} size: {len(test_df)}")

    # Only validation metrics may drive scheduling and checkpoint selection.
    eval_datasets = {"main_val": val_df}
    final_datasets = {spec.name: df for spec, df in test_datasets_loaded}
    if config.test_csv_path is not None:
        test_df = pd.read_csv(config.test_csv_path)
        for selection_df in (train_df, val_df):
            if set(test_df["input_text"]) & set(selection_df["input_text"]):
                raise ValueError("Test sentences overlap training or validation.")
        final_datasets["main_test"] = test_df
        safe_save_csv(test_df, os.path.join(config.output_dir, "test_df.csv"))
    if set(train_df["input_text"]) & set(val_df["input_text"]):
        raise ValueError("Training and validation sentences overlap.")

    finetuner, history_df, epoch_metrics_df, epoch_per_emotion_df = train_feature_steering(
        finetuner=finetuner,
        train_df=train_df,
        val_df=val_df,
        general_texts=general_texts,
        eval_datasets=eval_datasets,
        num_epochs=config.num_epochs,
        num_steps=config.num_steps,
        eval_every_steps=config.eval_every_steps,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        ppl_batch_size=config.ppl_batch_size,
        output_dir=config.output_dir,
        eval_general_ppl=config.eval_general_ppl,
        eval_dataset_ppl=config.eval_dataset_ppl,
        max_eval_ppl_texts=config.max_eval_ppl_texts,
        save_epoch_predictions=config.save_epoch_predictions,
    )

    # train_feature_steering restores the checkpoint chosen on validation.
    selected_checkpoint_metrics = [
        evaluate_and_save_report(finetuner, df, name, config.output_dir,
                                 batch_size=config.eval_batch_size)
        for name, df in final_datasets.items()
    ]
    safe_save_csv(pd.DataFrame(selected_checkpoint_metrics),
                  os.path.join(config.output_dir, "selected_checkpoint_metrics.csv"))

    multipliers_df = finetuner.export_multipliers(
        output_csv_path=os.path.join(config.output_dir, "learned_feature_multipliers.csv"),
        description_csv_path=config.feature_csv_path,
    )

    final_epoch = int(epoch_metrics_df["epoch"].max()) if len(epoch_metrics_df) else int(config.num_epochs)
    final_metrics = epoch_metrics_df[epoch_metrics_df["epoch"] == final_epoch].to_dict(orient="records") if len(epoch_metrics_df) else []
    baseline_metrics = epoch_metrics_df[epoch_metrics_df["epoch"] == 0].to_dict(orient="records") if len(epoch_metrics_df) else []

    summary = {
        "baseline_epoch_metrics": baseline_metrics,
        "last_validation_epoch_metrics": final_metrics,
        "selected_checkpoint_test_metrics": selected_checkpoint_metrics,
        "num_trainable_feature_multipliers": finetuner.feature_optimizer.get_num_trainable_parameters(),
        "output_dir": config.output_dir,
        "saved_files": {
            "training_history": os.path.join(config.output_dir, "training_history.csv"),
            "epoch_metrics_long": os.path.join(config.output_dir, "epoch_metrics_long.csv"),
            "epoch_per_emotion_f1": os.path.join(config.output_dir, "epoch_per_emotion_f1.csv"),
            "epoch_multipliers_dir": os.path.join(config.output_dir, "epoch_multipliers"),
            "final_multipliers": os.path.join(config.output_dir, "learned_feature_multipliers.csv"),
        },
    }
    safe_save_json(summary, os.path.join(config.output_dir, "summary.json"))

    outputs = {
        "train_all_df": train_all_df,
        "train_df": train_df,
        "val_df": val_df,
        "history_df": history_df,
        "epoch_metrics_df": epoch_metrics_df,
        "epoch_per_emotion_df": epoch_per_emotion_df,
        "multipliers_df": multipliers_df,
        "summary": summary,
    }
    return finetuner, outputs


# ============================================================
# Convenience example builder
# ============================================================

def build_default_gemma2_2b_config(
    feature_csv_path: str = "data/google-gemma-2-2b-feature-act/auto_selected_unique_features_with_affected_emotions.csv",
    train_csv_path: str = "data/train_df.csv",
    output_dir: str = "data/google-gemma-2-2b-feature-act/global_feature_steering",
    cache_dir: Optional[str] = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
    wikitext_cache_dir: Optional[str] = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")),
    val_csv_path: Optional[str] = "data/val_df.csv",
    main_csv_path: Optional[str] = "data/combined_emotions.csv",
    test_csv_path: Optional[str] = "data/test_df.csv",
) -> FeatureSteeringConfig:
    return FeatureSteeringConfig(
        model_name="google/gemma-2-2b",
        sae_size=None,
        feature_csv_path=feature_csv_path,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        main_csv_path=main_csv_path,
        test_csv_path=test_csv_path,
        output_dir=output_dir,
        train_text_col="input_text",
        train_label_col="emotion",
        train_apply_prompt_template=False,
        max_per_class=450,
        num_epochs=5,
        batch_size=16,
        eval_batch_size=32,
        ppl_batch_size=8,
        lambda_kl=0.01,
        lambda_l1=0.005,
        lr=1e-2,
        cache_dir=cache_dir,
        wikitext_cache_dir=wikitext_cache_dir,
        test_datasets=[
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
        ],
    )
