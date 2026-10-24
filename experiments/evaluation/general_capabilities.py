"""
Evaluate whether best SAE feature steering harms capabilities unrelated to
emotion recognition.

Models:
    - google/gemma-2-2b
    - google/gemma-2-9b
    - google/gemma-3-4b-pt
    - google/gemma-3-4b-it
    - google/gemma-3-12b-pt
    - google/gemma-3-12b-it
    - meta-llama/Llama-3.1-8B

Benchmarks:
    - MMLU
    - ARC-Challenge
    - BoolQ
    - PIQA

For every model and benchmark, this script compares:
    1. the vanilla model with no SAE hooks;
    2. the model using the best saved feature-steering checkpoint.

Evaluation method:
    Each example is converted to a zero-shot multiple-choice prompt ending in
    "Answer:". The score for each answer label is the next-token logit at the
    final prompt token. Steering is applied at that same final prompt token,
    matching the original emotion-classification intervention setup.

This is intended as a paired side-effect evaluation. Because it is zero-shot
answer-label scoring, its absolute scores should not be presented as directly
comparable to few-shot or lm-evaluation-harness leaderboard scores.

Expected files for each model:
    <feature_dir>/auto_selected_unique_features_with_affected_emotions.csv
    <steering_run_dir>/best_feature_steering.pt

Run from the repository root after restoring the research environment:
    python scripts/run.py general-capabilities
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from tqdm import tqdm

from sae_emotion.feature_steering_utils import (
    DEFAULT_EMOTIONS,
    SAEFeatureFinetuner,
    safe_save_csv,
    safe_save_json,
)

try:
    from scipy.stats import binomtest
except Exception:  # scipy is optional
    binomtest = None


# ============================================================
# User settings
# ============================================================

BENCHMARKS_TO_RUN = [
    "mmlu",
    "arc_challenge",
    "boolq",
    "piqa",
]

# Use "quick" first to verify paths and memory, then switch to "full".
# In full mode, every labeled example in the selected split is evaluated.
RUN_MODE = "full"  # "quick" or "full"

QUICK_LIMITS: Dict[str, Optional[int]] = {
    "mmlu": 200,
    "arc_challenge": 200,
    "boolq": 200,
    "piqa": 200,
}

FULL_LIMITS: Dict[str, Optional[int]] = {
    "mmlu": None,
    "arc_challenge": None,
    "boolq": None,
    "piqa": None,
}

SAMPLE_SEED = 42

OUTPUT_DIR = "data/general_capability_side_effects"
FEATURE_CSV_NAME = "auto_selected_unique_features_with_affected_emotions.csv"
STEERING_RUN_DIR_NAME = "lr_5e-3_l1_0p0"

CACHE_DIR: Optional[str] = None
DATASET_CACHE_DIR: Optional[str] = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))

DEVICE = "cuda"
DEVICE_MAP = "cuda"
TORCH_DTYPE = torch.bfloat16
MAX_LENGTH = 512

# Left truncation preserves the question, options, and trailing "Answer:"
# when a BoolQ passage exceeds MAX_LENGTH.
TRUNCATION_SIDE = "left"

SAVE_PROMPT_TEXT = False
SKIP_COMPLETED_CONDITIONS = True

# The evaluator uses final-token steering, which matches the emotion
# classification intervention and is much faster than scoring every option
# continuation separately.
STEERING_FINAL_TOKEN_ONLY = True


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    feature_dir: str
    sae_size: Optional[str]
    eval_batch_size: int

    @property
    def safe_name(self) -> str:
        return self.model_name.replace("/", "-")

    @property
    def feature_csv_path(self) -> str:
        return os.path.join(self.feature_dir, FEATURE_CSV_NAME)

    @property
    def steering_run_dir(self) -> str:
        return os.path.join(
            self.feature_dir,
            "global_feature_steering",
            STEERING_RUN_DIR_NAME,
        )

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.steering_run_dir, "best_feature_steering.pt")


MODEL_SPECS = [
    ModelSpec(
        model_name="google/gemma-2-2b",
        feature_dir="data/google-gemma-2-2b-feature-act",
        sae_size=None,
        eval_batch_size=16,
    ),
    ModelSpec(
        model_name="google/gemma-2-9b",
        feature_dir="data/google-gemma-2-9b-feature-act",
        sae_size=None,
        eval_batch_size=8,
    ),
    ModelSpec(
        model_name="google/gemma-3-4b-pt",
        feature_dir="data/google-gemma-3-4b-pt-feature-act-small",
        sae_size="small",
        eval_batch_size=12,
    ),
    ModelSpec(
        model_name="google/gemma-3-4b-it",
        feature_dir="data/google-gemma-3-4b-it-feature-act-small",
        sae_size="small",
        eval_batch_size=12,
    ),
    ModelSpec(
        model_name="google/gemma-3-12b-pt",
        feature_dir="data/google-gemma-3-12b-pt-feature-act-small",
        sae_size="small",
        eval_batch_size=4,
    ),
    ModelSpec(
        model_name="google/gemma-3-12b-it",
        feature_dir="data/google-gemma-3-12b-it-feature-act-small",
        sae_size="small",
        eval_batch_size=4,
    ),
    ModelSpec(
        model_name="meta-llama/Llama-3.1-8B",
        feature_dir="data/meta-llama-Llama-3.1-8B-feature-act",
        sae_size=None,
        eval_batch_size=8,
    ),
]


# ============================================================
# Canonical benchmark representation
# ============================================================

@dataclass
class BenchmarkExample:
    benchmark: str
    example_uid: str
    group: str
    prompt: str
    candidate_labels: List[str]
    gold_index: int
    question_preview: str

    @property
    def gold_label(self) -> str:
        return self.candidate_labels[self.gold_index]


@dataclass
class BenchmarkData:
    name: str
    split: str
    examples: List[BenchmarkExample]


# ============================================================
# Prompt builders
# ============================================================

def index_to_letter(index: int) -> str:
    if index < 0 or index >= 26:
        raise ValueError(f"Cannot convert option index {index} to a letter.")
    return chr(ord("A") + index)


def format_multiple_choice_prompt(
    question: str,
    choices: Sequence[str],
    instruction: str,
    context: Optional[str] = None,
) -> Tuple[str, List[str]]:
    labels = [index_to_letter(i) for i in range(len(choices))]
    lines = [instruction]
    if context:
        lines.extend([f"Context: {context}"])
    lines.append(f"Question: {question}")
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines), labels


def format_boolq_prompt(passage: str, question: str) -> Tuple[str, List[str]]:
    prompt = (
        "Read the passage and answer the question with Yes or No.\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt, ["Yes", "No"]


def format_piqa_prompt(goal: str, sol1: str, sol2: str) -> Tuple[str, List[str]]:
    return format_multiple_choice_prompt(
        question=goal,
        choices=[sol1, sol2],
        instruction="Choose the more plausible solution for the following goal.",
    )


# ============================================================
# Dataset loading
# ============================================================

def current_limits() -> Dict[str, Optional[int]]:
    if RUN_MODE == "quick":
        return QUICK_LIMITS
    if RUN_MODE == "full":
        return FULL_LIMITS
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


def apply_deterministic_limit(
    examples: List[BenchmarkExample],
    limit: Optional[int],
    seed: int,
) -> List[BenchmarkExample]:
    if limit is None or limit >= len(examples):
        return examples
    if limit <= 0:
        raise ValueError("Benchmark limits must be positive or None.")

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(examples)), k=int(limit)))
    return [examples[i] for i in selected_indices]


def _load_mmlu_all_split():
    """Load the MMLU test split, with a per-subject fallback."""
    try:
        return load_dataset(
            "cais/mmlu",
            "all",
            split="test",
            cache_dir=DATASET_CACHE_DIR,
        )
    except Exception as first_exc:
        print("Loading cais/mmlu config='all' failed. Falling back to subjects.")
        print(repr(first_exc))

        configs = get_dataset_config_names("cais/mmlu")
        excluded = {"all", "auxiliary_train"}
        subject_configs = [c for c in configs if c not in excluded]
        if not subject_configs:
            raise RuntimeError("No MMLU subject configurations were found.") from first_exc

        pieces = []
        for subject in tqdm(subject_configs, desc="load MMLU subjects"):
            ds = load_dataset(
                "cais/mmlu",
                subject,
                split="test",
                cache_dir=DATASET_CACHE_DIR,
            )
            if "subject" not in ds.column_names:
                ds = ds.add_column("subject", [subject] * len(ds))
            pieces.append(ds)
        return concatenate_datasets(pieces)


def _normalize_mmlu_answer(answer, num_choices: int) -> int:
    if isinstance(answer, (int, np.integer)):
        idx = int(answer)
    else:
        text = str(answer).strip()
        if text.isdigit():
            idx = int(text)
        else:
            idx = ord(text.upper()) - ord("A")
    if idx < 0 or idx >= num_choices:
        raise ValueError(f"Invalid MMLU answer {answer!r} for {num_choices} choices.")
    return idx


def load_mmlu(limit: Optional[int]) -> BenchmarkData:
    ds = _load_mmlu_all_split()
    examples: List[BenchmarkExample] = []

    for i, row in enumerate(ds):
        question = str(row["question"])
        choices = [str(x) for x in row["choices"]]
        subject = str(row.get("subject", "unknown_subject"))
        prompt, labels = format_multiple_choice_prompt(
            question=question,
            choices=choices,
            instruction=(
                "Choose the correct answer to the following multiple-choice "
                f"question about {subject.replace('_', ' ')}."
            ),
        )
        gold_index = _normalize_mmlu_answer(row["answer"], len(choices))
        examples.append(
            BenchmarkExample(
                benchmark="mmlu",
                example_uid=f"mmlu:{subject}:{i}",
                group=subject,
                prompt=prompt,
                candidate_labels=labels,
                gold_index=gold_index,
                question_preview=question[:300],
            )
        )

    examples = apply_deterministic_limit(examples, limit, SAMPLE_SEED)
    return BenchmarkData(name="mmlu", split="test", examples=examples)


def _find_arc_gold_index(choice_labels: Sequence[str], answer_key: str) -> int:
    normalized_labels = [str(x).strip().upper() for x in choice_labels]
    key = str(answer_key).strip().upper()

    if key in normalized_labels:
        return normalized_labels.index(key)

    if key.isdigit():
        numeric = int(key)
        if 1 <= numeric <= len(choice_labels):
            return numeric - 1
        if 0 <= numeric < len(choice_labels):
            return numeric

    if len(key) == 1 and "A" <= key <= "Z":
        idx = ord(key) - ord("A")
        if idx < len(choice_labels):
            return idx

    raise ValueError(
        f"Could not match ARC answerKey={answer_key!r} to labels={choice_labels!r}."
    )


def load_arc_challenge(limit: Optional[int]) -> BenchmarkData:
    # The test split distributed with AI2 ARC contains answerKey. If a future
    # dataset revision omits it, the code falls back to validation.
    split = "test"
    ds = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split=split,
        cache_dir=DATASET_CACHE_DIR,
    )
    if "answerKey" not in ds.column_names or any(
        row["answerKey"] in (None, "") for row in ds.select(range(min(20, len(ds))))
    ):
        split = "validation"
        ds = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            split=split,
            cache_dir=DATASET_CACHE_DIR,
        )

    examples: List[BenchmarkExample] = []
    for i, row in enumerate(ds):
        question = str(row["question"])
        choice_texts = [str(x) for x in row["choices"]["text"]]
        original_labels = [str(x) for x in row["choices"]["label"]]
        gold_index = _find_arc_gold_index(original_labels, row["answerKey"])
        prompt, labels = format_multiple_choice_prompt(
            question=question,
            choices=choice_texts,
            instruction="Choose the correct answer to the following science question.",
        )
        examples.append(
            BenchmarkExample(
                benchmark="arc_challenge",
                example_uid=f"arc_challenge:{split}:{row.get('id', i)}",
                group="all",
                prompt=prompt,
                candidate_labels=labels,
                gold_index=gold_index,
                question_preview=question[:300],
            )
        )

    examples = apply_deterministic_limit(examples, limit, SAMPLE_SEED)
    return BenchmarkData(name="arc_challenge", split=split, examples=examples)


def load_boolq(limit: Optional[int]) -> BenchmarkData:
    split = "validation"
    ds = load_dataset(
        "google/boolq",
        split=split,
        cache_dir=DATASET_CACHE_DIR,
    )
    examples: List[BenchmarkExample] = []

    for i, row in enumerate(ds):
        question = str(row["question"])
        passage = str(row["passage"])
        prompt, labels = format_boolq_prompt(passage, question)
        gold_index = 0 if bool(row["answer"]) else 1
        examples.append(
            BenchmarkExample(
                benchmark="boolq",
                example_uid=f"boolq:{split}:{i}",
                group="all",
                prompt=prompt,
                candidate_labels=labels,
                gold_index=gold_index,
                question_preview=question[:300],
            )
        )

    examples = apply_deterministic_limit(examples, limit, SAMPLE_SEED)
    return BenchmarkData(name="boolq", split=split, examples=examples)


def load_piqa(limit: Optional[int]) -> BenchmarkData:
    split = "validation"
    ds = load_dataset(
        "ybisk/piqa",
        split=split,
        cache_dir=DATASET_CACHE_DIR,
    )
    examples: List[BenchmarkExample] = []

    for i, row in enumerate(ds):
        label = int(row["label"])
        if label not in {0, 1}:
            continue
        goal = str(row["goal"])
        prompt, labels = format_piqa_prompt(
            goal=goal,
            sol1=str(row["sol1"]),
            sol2=str(row["sol2"]),
        )
        examples.append(
            BenchmarkExample(
                benchmark="piqa",
                example_uid=f"piqa:{split}:{i}",
                group="all",
                prompt=prompt,
                candidate_labels=labels,
                gold_index=label,
                question_preview=goal[:300],
            )
        )

    examples = apply_deterministic_limit(examples, limit, SAMPLE_SEED)
    return BenchmarkData(name="piqa", split=split, examples=examples)


def load_benchmarks() -> Dict[str, BenchmarkData]:
    limits = current_limits()
    loaders = {
        "mmlu": load_mmlu,
        "arc_challenge": load_arc_challenge,
        "boolq": load_boolq,
        "piqa": load_piqa,
    }

    unknown = set(BENCHMARKS_TO_RUN) - set(loaders)
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")

    benchmark_data: Dict[str, BenchmarkData] = {}
    for name in BENCHMARKS_TO_RUN:
        print("=" * 100)
        print(f"Loading benchmark: {name}")
        benchmark_data[name] = loaders[name](limits.get(name))
        print(
            f"Loaded {name}: split={benchmark_data[name].split}, "
            f"n={len(benchmark_data[name].examples)}"
        )
    return benchmark_data


# ============================================================
# Model and scoring helpers
# ============================================================

def validate_required_files(model_spec: ModelSpec) -> None:
    required = [
        model_spec.feature_csv_path,
        model_spec.best_checkpoint_path,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required files for {model_spec.model_name}:\n{formatted}"
        )


def resolve_single_token_label_ids(
    tokenizer,
    labels: Iterable[str],
) -> Dict[str, int]:
    """Resolve each answer label to one next-token ID.

    Leading-space forms are preferred because every benchmark prompt ends at
    "Answer:" and the natural continuation is usually " A", " Yes", etc.
    The script stops with a clear error if a label is not one token, rather
    than silently using a non-equivalent scoring method.
    """
    mapping: Dict[str, int] = {}
    for label in sorted(set(str(x) for x in labels)):
        candidates = [f" {label}", label]
        resolved = None
        for candidate in candidates:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) == 1:
                resolved = int(token_ids[0])
                break
        if resolved is None:
            encodings = {
                candidate: tokenizer.encode(candidate, add_special_tokens=False)
                for candidate in candidates
            }
            raise ValueError(
                f"Answer label {label!r} is not a single token for this tokenizer: "
                f"{encodings}. Use continuation log-likelihood instead."
            )
        mapping[label] = resolved
    return mapping


def collect_all_candidate_labels(benchmark_data: Dict[str, BenchmarkData]) -> List[str]:
    labels = set()
    for data in benchmark_data.values():
        for example in data.examples:
            labels.update(example.candidate_labels)
    return sorted(labels)


@torch.inference_mode()
def get_final_full_logits(
    finetuner: SAEFeatureFinetuner,
    prompts: List[str],
    condition: str,
) -> torch.Tensor:
    if condition == "vanilla":
        _, full_logits, _, _ = finetuner.forward_clean(
            prompts,
            return_full_logits=True,
        )
        return full_logits

    if condition == "steered":
        _, full_logits, _, _ = finetuner.forward_with_interventions(
            prompts,
            return_full_logits=True,
            final_token_only=STEERING_FINAL_TOKEN_ONLY,
        )
        return full_logits

    raise ValueError(f"Unknown condition: {condition}")


def prediction_output_path(
    model_spec: ModelSpec,
    benchmark_name: str,
    condition: str,
) -> str:
    return os.path.join(
        OUTPUT_DIR,
        "condition_predictions",
        model_spec.safe_name,
        f"{benchmark_name}_{condition}.csv",
    )


def validate_resumed_predictions(
    df: pd.DataFrame,
    examples: Sequence[BenchmarkExample],
    model_name: str,
    benchmark_name: str,
    condition: str,
) -> bool:
    required = {
        "model",
        "benchmark",
        "condition",
        "example_uid",
        "gold_index",
        "pred_index",
        "correct",
    }
    if not required.issubset(df.columns):
        return False
    if len(df) != len(examples):
        return False
    if not (df["model"] == model_name).all():
        return False
    if not (df["benchmark"] == benchmark_name).all():
        return False
    if not (df["condition"] == condition).all():
        return False
    expected_uids = [x.example_uid for x in examples]
    return df["example_uid"].astype(str).tolist() == expected_uids


def score_condition(
    finetuner: SAEFeatureFinetuner,
    model_spec: ModelSpec,
    benchmark: BenchmarkData,
    condition: str,
    label_to_token_id: Dict[str, int],
) -> pd.DataFrame:
    output_path = prediction_output_path(
        model_spec=model_spec,
        benchmark_name=benchmark.name,
        condition=condition,
    )

    if SKIP_COMPLETED_CONDITIONS and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if validate_resumed_predictions(
            existing,
            benchmark.examples,
            model_spec.model_name,
            benchmark.name,
            condition,
        ):
            print(f"Using completed result: {output_path}")
            return existing
        print(f"Ignoring incompatible existing result: {output_path}")

    rows: List[Dict] = []
    batch_size = int(model_spec.eval_batch_size)
    examples = benchmark.examples

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc=f"{model_spec.safe_name} {benchmark.name} {condition}",
    ):
        batch_examples = examples[start:start + batch_size]
        prompts = [x.prompt for x in batch_examples]
        full_logits = get_final_full_logits(
            finetuner=finetuner,
            prompts=prompts,
            condition=condition,
        )
        full_logits = full_logits.detach().float().cpu()

        for local_idx, example in enumerate(batch_examples):
            candidate_token_ids = [
                label_to_token_id[label]
                for label in example.candidate_labels
            ]
            candidate_scores = full_logits[
                local_idx,
                torch.tensor(candidate_token_ids, dtype=torch.long),
            ].numpy()
            pred_index = int(np.argmax(candidate_scores))
            pred_label = example.candidate_labels[pred_index]
            row = {
                "model": model_spec.model_name,
                "benchmark": benchmark.name,
                "split": benchmark.split,
                "condition": condition,
                "example_uid": example.example_uid,
                "group": example.group,
                "gold_index": int(example.gold_index),
                "pred_index": pred_index,
                "gold_label": example.gold_label,
                "pred_label": pred_label,
                "correct": int(pred_index == example.gold_index),
                "candidate_labels_json": json.dumps(example.candidate_labels),
                "candidate_scores_json": json.dumps(
                    [float(x) for x in candidate_scores.tolist()]
                ),
                "question_preview": example.question_preview,
            }
            if SAVE_PROMPT_TEXT:
                row["prompt"] = example.prompt
            rows.append(row)

        del full_logits

    result_df = pd.DataFrame(rows)
    safe_save_csv(result_df, output_path)
    return result_df


def accuracy_and_se(correct_values: Sequence[int]) -> Tuple[float, float]:
    values = np.asarray(correct_values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    accuracy = float(values.mean())
    standard_error = float(math.sqrt(accuracy * (1.0 - accuracy) / values.size))
    return accuracy, standard_error


def summarize_condition_predictions(pred_df: pd.DataFrame) -> Dict[str, float]:
    accuracy, standard_error = accuracy_and_se(pred_df["correct"].astype(int).values)
    return {
        "n": int(len(pred_df)),
        "accuracy": accuracy,
        "accuracy_standard_error": standard_error,
    }


def paired_comparison(
    vanilla_df: pd.DataFrame,
    steered_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    vanilla = vanilla_df.sort_values("example_uid").reset_index(drop=True)
    steered = steered_df.sort_values("example_uid").reset_index(drop=True)

    if vanilla["example_uid"].tolist() != steered["example_uid"].tolist():
        raise RuntimeError("Vanilla and steered example ordering differs.")
    if vanilla["gold_index"].astype(int).tolist() != steered["gold_index"].astype(int).tolist():
        raise RuntimeError("Vanilla and steered gold labels differ.")

    paired = vanilla[[
        "model",
        "benchmark",
        "split",
        "example_uid",
        "group",
        "gold_index",
        "gold_label",
        "question_preview",
    ]].copy()
    paired["vanilla_pred_index"] = vanilla["pred_index"].astype(int)
    paired["steered_pred_index"] = steered["pred_index"].astype(int)
    paired["vanilla_pred_label"] = vanilla["pred_label"]
    paired["steered_pred_label"] = steered["pred_label"]
    paired["vanilla_correct"] = vanilla["correct"].astype(int)
    paired["steered_correct"] = steered["correct"].astype(int)
    paired["prediction_changed"] = (
        paired["vanilla_pred_index"] != paired["steered_pred_index"]
    ).astype(int)
    paired["correct_to_incorrect"] = (
        (paired["vanilla_correct"] == 1)
        & (paired["steered_correct"] == 0)
    ).astype(int)
    paired["incorrect_to_correct"] = (
        (paired["vanilla_correct"] == 0)
        & (paired["steered_correct"] == 1)
    ).astype(int)

    vanilla_accuracy = float(paired["vanilla_correct"].mean())
    steered_accuracy = float(paired["steered_correct"].mean())
    c2i = int(paired["correct_to_incorrect"].sum())
    i2c = int(paired["incorrect_to_correct"].sum())
    discordant = c2i + i2c

    if binomtest is not None and discordant > 0:
        mcnemar_p = float(binomtest(min(c2i, i2c), discordant, 0.5).pvalue)
    else:
        mcnemar_p = float("nan")

    summary = {
        "n": int(len(paired)),
        "vanilla_accuracy": vanilla_accuracy,
        "steered_accuracy": steered_accuracy,
        "delta_accuracy": steered_accuracy - vanilla_accuracy,
        "prediction_flip_rate": float(paired["prediction_changed"].mean()),
        "correct_to_incorrect_count": c2i,
        "correct_to_incorrect_rate": float(c2i / max(len(paired), 1)),
        "incorrect_to_correct_count": i2c,
        "incorrect_to_correct_rate": float(i2c / max(len(paired), 1)),
        "mcnemar_exact_p": mcnemar_p,
    }
    return summary, paired


def summarize_groups(
    pred_df: pd.DataFrame,
    model_name: str,
    benchmark_name: str,
    condition: str,
) -> List[Dict]:
    rows = []
    for group_name, group_df in pred_df.groupby("group", sort=True):
        accuracy, standard_error = accuracy_and_se(
            group_df["correct"].astype(int).values
        )
        rows.append(
            {
                "model": model_name,
                "benchmark": benchmark_name,
                "condition": condition,
                "group": group_name,
                "n": int(len(group_df)),
                "accuracy": accuracy,
                "accuracy_standard_error": standard_error,
            }
        )
    return rows


def cleanup_after_model(finetuner: Optional[SAEFeatureFinetuner]) -> None:
    if finetuner is not None:
        del finetuner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# Main evaluation
# ============================================================

def evaluate_model(
    model_spec: ModelSpec,
    benchmark_data: Dict[str, BenchmarkData],
    long_metric_rows: List[Dict],
    comparison_rows: List[Dict],
    group_metric_rows: List[Dict],
) -> None:
    validate_required_files(model_spec)

    print("=" * 100)
    print(f"Model: {model_spec.model_name}")
    print(f"Feature CSV: {model_spec.feature_csv_path}")
    print(f"Best checkpoint: {model_spec.best_checkpoint_path}")
    print(f"Evaluation batch size: {model_spec.eval_batch_size}")
    print("=" * 100)

    finetuner: Optional[SAEFeatureFinetuner] = None
    try:
        finetuner = SAEFeatureFinetuner(
            model_name=model_spec.model_name,
            feature_csv_path=model_spec.feature_csv_path,
            emotions=list(DEFAULT_EMOTIONS),
            sae_size=model_spec.sae_size,
            cache_dir=CACHE_DIR,
            device=DEVICE,
            device_map=DEVICE_MAP,
            torch_dtype=TORCH_DTYPE,
            max_length=MAX_LENGTH,
        )
        finetuner.tokenizer.truncation_side = TRUNCATION_SIDE

        all_labels = collect_all_candidate_labels(benchmark_data)
        label_to_token_id = resolve_single_token_label_ids(
            finetuner.tokenizer,
            all_labels,
        )
        print("Answer-label token IDs:")
        for label, token_id in label_to_token_id.items():
            print(
                f"  {label!r}: {token_id} "
                f"({finetuner.tokenizer.decode([token_id])!r})"
            )

        # Loading the learned multipliers does not affect vanilla forwards,
        # because vanilla evaluation registers no SAE hooks.
        finetuner.load_state(model_spec.best_checkpoint_path)

        for benchmark_name in BENCHMARKS_TO_RUN:
            benchmark = benchmark_data[benchmark_name]
            print(
                f"\nBenchmark: {benchmark.name}, split={benchmark.split}, "
                f"n={len(benchmark.examples)}"
            )

            vanilla_df = score_condition(
                finetuner=finetuner,
                model_spec=model_spec,
                benchmark=benchmark,
                condition="vanilla",
                label_to_token_id=label_to_token_id,
            )
            steered_df = score_condition(
                finetuner=finetuner,
                model_spec=model_spec,
                benchmark=benchmark,
                condition="steered",
                label_to_token_id=label_to_token_id,
            )

            for condition, pred_df in [
                ("vanilla", vanilla_df),
                ("steered", steered_df),
            ]:
                metric = summarize_condition_predictions(pred_df)
                long_metric_rows.append(
                    {
                        "model": model_spec.model_name,
                        "benchmark": benchmark.name,
                        "split": benchmark.split,
                        "condition": condition,
                        **metric,
                    }
                )
                group_metric_rows.extend(
                    summarize_groups(
                        pred_df=pred_df,
                        model_name=model_spec.model_name,
                        benchmark_name=benchmark.name,
                        condition=condition,
                    )
                )

            comparison, paired_df = paired_comparison(vanilla_df, steered_df)
            comparison_rows.append(
                {
                    "model": model_spec.model_name,
                    "benchmark": benchmark.name,
                    "split": benchmark.split,
                    **comparison,
                }
            )

            paired_output_path = os.path.join(
                OUTPUT_DIR,
                "paired_predictions",
                model_spec.safe_name,
                f"{benchmark.name}_paired.csv",
            )
            safe_save_csv(paired_df, paired_output_path)

            safe_save_csv(
                pd.DataFrame(long_metric_rows),
                os.path.join(OUTPUT_DIR, "general_capability_results_long.csv"),
            )
            safe_save_csv(
                pd.DataFrame(comparison_rows),
                os.path.join(OUTPUT_DIR, "general_capability_comparison.csv"),
            )
            safe_save_csv(
                pd.DataFrame(group_metric_rows),
                os.path.join(OUTPUT_DIR, "general_capability_group_results_long.csv"),
            )

            print(
                f"[{benchmark.name}] "
                f"vanilla={comparison['vanilla_accuracy']:.4f}, "
                f"steered={comparison['steered_accuracy']:.4f}, "
                f"delta={comparison['delta_accuracy']:+.4f}, "
                f"correct->incorrect={comparison['correct_to_incorrect_rate']:.4f}, "
                f"incorrect->correct={comparison['incorrect_to_correct_rate']:.4f}"
            )

    finally:
        cleanup_after_model(finetuner)


def create_final_aggregate_tables(
    long_metric_rows: List[Dict],
    comparison_rows: List[Dict],
    group_metric_rows: List[Dict],
) -> None:
    long_df = pd.DataFrame(long_metric_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    group_df = pd.DataFrame(group_metric_rows)

    safe_save_csv(
        long_df,
        os.path.join(OUTPUT_DIR, "general_capability_results_long.csv"),
    )
    safe_save_csv(
        comparison_df,
        os.path.join(OUTPUT_DIR, "general_capability_comparison.csv"),
    )
    safe_save_csv(
        group_df,
        os.path.join(OUTPUT_DIR, "general_capability_group_results_long.csv"),
    )

    if len(comparison_df) > 0:
        model_average = (
            comparison_df.groupby("model", as_index=False)
            .agg(
                num_benchmarks=("benchmark", "nunique"),
                vanilla_accuracy_macro_average=("vanilla_accuracy", "mean"),
                steered_accuracy_macro_average=("steered_accuracy", "mean"),
                delta_accuracy_macro_average=("delta_accuracy", "mean"),
                mean_prediction_flip_rate=("prediction_flip_rate", "mean"),
                mean_correct_to_incorrect_rate=("correct_to_incorrect_rate", "mean"),
                mean_incorrect_to_correct_rate=("incorrect_to_correct_rate", "mean"),
            )
        )
        safe_save_csv(
            model_average,
            os.path.join(OUTPUT_DIR, "general_capability_model_average.csv"),
        )

        benchmark_average = (
            comparison_df.groupby("benchmark", as_index=False)
            .agg(
                num_models=("model", "nunique"),
                vanilla_accuracy_mean=("vanilla_accuracy", "mean"),
                steered_accuracy_mean=("steered_accuracy", "mean"),
                delta_accuracy_mean=("delta_accuracy", "mean"),
                mean_prediction_flip_rate=("prediction_flip_rate", "mean"),
                mean_correct_to_incorrect_rate=("correct_to_incorrect_rate", "mean"),
                mean_incorrect_to_correct_rate=("incorrect_to_correct_rate", "mean"),
            )
        )
        safe_save_csv(
            benchmark_average,
            os.path.join(OUTPUT_DIR, "general_capability_benchmark_average.csv"),
        )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_spec in MODEL_SPECS:
        validate_required_files(model_spec)

    benchmark_data = load_benchmarks()

    run_config = {
        "run_mode": RUN_MODE,
        "benchmarks": BENCHMARKS_TO_RUN,
        "limits": current_limits(),
        "sample_seed": SAMPLE_SEED,
        "output_dir": OUTPUT_DIR,
        "feature_csv_name": FEATURE_CSV_NAME,
        "steering_run_dir_name": STEERING_RUN_DIR_NAME,
        "max_length": MAX_LENGTH,
        "truncation_side": TRUNCATION_SIDE,
        "steering_final_token_only": STEERING_FINAL_TOKEN_ONLY,
        "scoring_method": "zero_shot_next_token_answer_label",
        "models": [asdict(x) for x in MODEL_SPECS],
        "loaded_benchmarks": {
            name: {
                "split": data.split,
                "n": len(data.examples),
            }
            for name, data in benchmark_data.items()
        },
    }
    safe_save_json(run_config, os.path.join(OUTPUT_DIR, "run_config.json"))

    total_questions = sum(len(x.examples) for x in benchmark_data.values())
    total_question_evaluations = total_questions * len(MODEL_SPECS) * 2
    print("=" * 100)
    print(f"Run mode: {RUN_MODE}")
    print(f"Questions per model: {total_questions:,}")
    print(
        "Total model-question evaluations "
        f"(all models, vanilla + steered): {total_question_evaluations:,}"
    )
    print("=" * 100)

    long_metric_rows: List[Dict] = []
    comparison_rows: List[Dict] = []
    group_metric_rows: List[Dict] = []
    failures: List[str] = []

    for model_spec in MODEL_SPECS:
        try:
            evaluate_model(
                model_spec=model_spec,
                benchmark_data=benchmark_data,
                long_metric_rows=long_metric_rows,
                comparison_rows=comparison_rows,
                group_metric_rows=group_metric_rows,
            )
        except Exception as exc:
            error = f"FAILED {model_spec.model_name}: {repr(exc)}"
            print(error)
            traceback.print_exc()
            failures.append(error)
            cleanup_after_model(None)

    create_final_aggregate_tables(
        long_metric_rows=long_metric_rows,
        comparison_rows=comparison_rows,
        group_metric_rows=group_metric_rows,
    )

    if failures:
        safe_save_json(
            {"failures": failures},
            os.path.join(OUTPUT_DIR, "failures.json"),
        )
        print("\nSome model evaluations failed:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("\nAll general-capability evaluations finished successfully.")


if __name__ == "__main__":
    main()
