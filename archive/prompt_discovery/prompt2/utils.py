import os
import gc
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel
from tqdm import tqdm


# ============================================================
# Basic utilities
# ============================================================

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "-")


# ============================================================
# Model utilities
# ============================================================

def load_llm(
    model_name: str,
    device_map: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    llm = LanguageModel(
        model_name,
        device_map=device_map,
        dtype=dtype,
    )
    return llm


def get_model_layers_and_dims(llm) -> Tuple[int, int]:
    try:
        n_layers = llm.model.config.num_hidden_layers
        hidden_dim = llm.model.config.hidden_size
    except Exception:
        n_layers = llm.model.language_model.config.num_hidden_layers
        hidden_dim = llm.model.language_model.config.hidden_size
        llm.model.layers = llm.model.language_model.layers

    return n_layers, hidden_dim


# ============================================================
# Prompt / emotion utilities
# ============================================================

def zero_shot_template(text: str) -> str:
    return f"In this text: {text}, the emotion implied is:"

    
def build_emotion_token_maps(
    llm,
    emotions: List[str],
    verbose: bool = True,
) -> Tuple[Dict[int, str], Dict[str, int]]:

    token_to_emotion = {}
    emotion_to_token = {}

    for emotion in emotions:
        token_ids = llm.tokenizer.encode(" " + emotion)

        if verbose:
            print(f"Emotion: {emotion}, Token IDs: {token_ids}")

        token_id = token_ids[-1]
        token_to_emotion[token_id] = emotion
        emotion_to_token[emotion] = token_id

    if verbose:
        print("token_to_emotion:", token_to_emotion)
        print("emotion_to_token:", emotion_to_token)

    return token_to_emotion, emotion_to_token


# ============================================================
# Hidden-state extraction
# ============================================================

def extract_hidden_states_and_predictions(
    llm,
    df: pd.DataFrame,
    text_col: str = "text",
    emotions: Optional[List[str]] = None,
    batch_size: int = 4,
    save_hidden_path: Optional[str] = None,
    save_preds_path: Optional[str] = None,
    save_correct_path: Optional[str] = None,
    label_col: Optional[str] = None,
    pred_col: str = "pred_emotion",
    prompt_fn: Callable[[str], str] = zero_shot_template,
) -> Tuple[pd.DataFrame, np.ndarray]:

    if emotions is None:
        emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    df_out = df.copy().reset_index(drop=True)

    n_layers, hidden_dim = get_model_layers_and_dims(llm)
    n_prompts = len(df_out)

    token_to_emotion, emotion_to_token = build_emotion_token_maps(
        llm=llm,
        emotions=emotions,
        verbose=True,
    )

    emotion_logit_indices = [emotion_to_token[e] for e in emotions]

    activations = np.zeros(
        (n_layers, n_prompts, hidden_dim),
        dtype=np.float32,
    )

    for start in tqdm(range(0, n_prompts, batch_size), desc="Extracting hidden states"):
        end = min(start + batch_size, n_prompts)

        batch_texts = df_out[text_col].iloc[start:end].tolist()
        batch_prompts = [prompt_fn(text) for text in batch_texts]

        saved_acts = []

        with llm.trace(batch_prompts):
            for layer_idx in range(n_layers):
                hidden_states = llm.model.layers[layer_idx].output
                final_token_acts = hidden_states[:, -1, :].save()
                saved_acts.append(final_token_acts)

            final_token_logits = llm.output.logits[:, -1, :].save()
            saved_acts.append(final_token_logits)

        for layer_idx in range(n_layers):
            activations[layer_idx, start:end, :] = (
                saved_acts[layer_idx]
                .float()
                .cpu()
                .detach()
                .numpy()
            )

        final_token_logits = saved_acts[-1]

        emotion_logits = (
            final_token_logits[:, emotion_logit_indices]
            .float()
            .cpu()
            .detach()
            .numpy()
        )

        pred_indices = np.argmax(emotion_logits, axis=1)
        pred_token_ids = [emotion_logit_indices[idx] for idx in pred_indices]
        pred_emotions = [token_to_emotion[token_id] for token_id in pred_token_ids]

        df_out.loc[start:end - 1, pred_col] = pred_emotions

        del saved_acts
        cleanup_memory()

    if save_preds_path is not None:
        os.makedirs(os.path.dirname(save_preds_path), exist_ok=True)
        df_out.to_csv(save_preds_path)

    if save_hidden_path is not None:
        os.makedirs(os.path.dirname(save_hidden_path), exist_ok=True)
        np.save(save_hidden_path, activations)

    if save_correct_path is not None:
        if label_col is None:
            raise ValueError("label_col must be provided when save_correct_path is used.")

        correct_df = df_out[df_out[label_col] == df_out[pred_col]].copy()
        os.makedirs(os.path.dirname(save_correct_path), exist_ok=True)
        correct_df.to_csv(save_correct_path)

    return df_out, activations


def run_hidden_state_pipeline(
    model_name: str,
    input_csv: str,
    output_dir: str = "data",
    text_col: str = "text",
    label_col: Optional[str] = "emotion",
    emotions: Optional[List[str]] = None,
    batch_size: int = 4,
    device_map: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):

    llm = load_llm(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
    )

    df = pd.read_csv(input_csv)
    safe_name = model_safe_name(model_name)

    pred_path = f"{output_dir}/combined_zeroshot_preds_{safe_name}.csv"
    hidden_path = f"{output_dir}/combined_zeroshot_hidden_states_{safe_name}.npy"

    correct_path = None
    if label_col is not None:
        correct_path = f"{output_dir}/combined_zeroshot_filtered_{safe_name}.csv"

    df_out, activations = extract_hidden_states_and_predictions(
        llm=llm,
        df=df,
        text_col=text_col,
        emotions=emotions,
        batch_size=batch_size,
        save_hidden_path=hidden_path,
        save_preds_path=pred_path,
        save_correct_path=correct_path,
        label_col=label_col,
    )

    return df_out, activations


# ============================================================
# SAE utilities
# ============================================================

from sae_lens import SAE

def get_sae_config(model_name: str, sae_size: str = "big"):
    """
    sae_size: "big", "medium", or "small"
    """

    if sae_size not in {"big", "medium", "small"}:
        raise ValueError(f"Unsupported sae_size: {sae_size}")

    # Gemma 2 canonical
    if "gemma-2-2b" in model_name:
        release = "gemma-scope-2b-pt-res-canonical"
        sae_id_fn = lambda i: f"layer_{i}/width_16k/canonical"

    elif "gemma-2-9b" in model_name:
        release = "gemma-scope-9b-pt-res-canonical"
        sae_id_fn = lambda i: f"layer_{i}/width_16k/canonical"

    # Gemma 3 1B
    elif "gemma-3-1b-pt" in model_name:
        release = "gemma-scope-2-1b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-1b-it" in model_name:
        release = "gemma-scope-2-1b-it-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    # Gemma 3 4B
    elif "gemma-3-4b-pt" in model_name:
        release = "gemma-scope-2-4b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-4b-it" in model_name:
        release = "gemma-scope-2-4b-it-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    # Gemma 3 12B
    elif "gemma-3-12b-pt" in model_name:
        release = "gemma-scope-2-12b-pt-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    elif "gemma-3-12b-it" in model_name:
        release = "gemma-scope-2-12b-it-res-all"
        sae_id_fn = lambda i: f"layer_{i}_width_16k_l0_{sae_size}"

    # Llama
    elif "Llama-3.1-8B" in model_name:
        release = "llama_scope_lxr_8x"
        sae_id_fn = lambda i: f"l{i}r_8x"

    elif "DeepSeek" in model_name:
        return None, None

    else:
        raise ValueError(f"Unsupported model for SAE loading: {model_name}")

    return release, sae_id_fn


def load_saes_for_model(
    model_name: str,
    n_layers: int,
    device: str = "cuda",
    sae_size: str = "big",
):
    release, sae_id_fn = get_sae_config(
        model_name=model_name,
        sae_size=sae_size,
    )

    if release is None:
        print(f"Skipping SAE loading for {model_name}")
        return None, None, None

    saes = {}

    for layer_idx in tqdm(range(n_layers), desc="Loading SAEs"):
        sae_id = sae_id_fn(layer_idx)

        print(f"Loading SAE: {release} | {sae_id}")

        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )

        saes[layer_idx] = sae

    return saes, release, sae_id_fn


def encode_hidden_states_with_saes(
    activations: np.ndarray,
    saes: Dict[int, object],
    batch_size: int = 4096,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> np.ndarray:

    if saes is None:
        raise ValueError("saes is None. This model does not have configured SAEs.")

    n_layers, n_instances, _ = activations.shape
    first_sae = saes[0]
    sae_width = first_sae.cfg.d_sae

    sae_activations = np.zeros(
        (n_layers, n_instances, sae_width),
        dtype=np.float32,
    )

    for layer_idx in tqdm(range(n_layers), desc="Encoding activations with SAEs"):
        sae = saes[layer_idx]
        layer_acts_np = activations[layer_idx]

        for start in tqdm(
            range(0, n_instances, batch_size),
            desc=f"Layer {layer_idx}",
            leave=False,
        ):
            end = min(start + batch_size, n_instances)

            layer_acts = torch.from_numpy(layer_acts_np[start:end]).float().to(device)

            with torch.no_grad():
                sae_acts = sae.encode(layer_acts)

            sae_activations[layer_idx, start:end, :] = (
                sae_acts.float()
                .cpu()
                .numpy()
            )

            del layer_acts, sae_acts
            cleanup_memory()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, sae_activations)

    return sae_activations


def run_sae_activation_pipeline(
    model_name: str,
    hidden_state_path: Optional[str] = None,
    output_dir: str = "data",
    batch_size: int = 4096,
    device: str = "cuda",
    sae_size: str = "big",
):
    safe_name = model_safe_name(model_name)

    if hidden_state_path is None:
        hidden_state_path = f"{output_dir}/combined_zeroshot_hidden_states_{safe_name}.npy"

    activations = np.load(hidden_state_path)
    n_layers = activations.shape[0]

    saes, release, sae_id_fn = load_saes_for_model(
        model_name=model_name,
        n_layers=n_layers,
        device=device,
        sae_size=sae_size,
    )

    if saes is None:
        print(f"Skipping SAE activation extraction for {model_name}")
        return None

    save_path = f"{output_dir}/sae_acts_combined_zeroshot_{safe_name}_{sae_size}.npy"

    sae_activations = encode_hidden_states_with_saes(
        activations=activations,
        saes=saes,
        batch_size=batch_size,
        device=device,
        save_path=save_path,
    )

    return sae_activations



