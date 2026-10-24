"""CPU preprocessing exported from the supplied label_correlated_ngram notebook."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score


# ============================================================
# Configuration
# ============================================================

INPUT_CSV_PATH = "data/self_reported_ekman_zeroshot_preds.csv"
OUTPUT_DIR = Path("data/lexical_shortcut_control")

TEXT_COL = "text"
LABEL_COL = "emotion"

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "surprise",
    "sadness",
]

# Lexical shortcut selection
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.95
TOP_K_PER_EMOTION = 20

# A term must occur proportionally this many times more often
# in the target emotion than in all other emotions.
MIN_PREVALENCE_RATIO = 3.0

# Minimum number of target-emotion examples containing the term.
MIN_TARGET_DOCUMENTS = 3

RANDOM_SEED = 42


# ============================================================
# Identify label-correlated n-grams
# ============================================================

def identify_label_correlated_ngrams(
    df,
    text_col,
    label_col,
    emotions,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    top_k_per_emotion=20,
    min_prevalence_ratio=3.0,
    min_target_documents=3,
):
    """
    Identify unigrams and bigrams that are disproportionately associated
    with each emotion label.

    Terms are ranked using the chi-square statistic and filtered using
    target-vs-other document prevalence ratios.
    """

    vectorizer = CountVectorizer(
        lowercase=True,
        binary=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b\w+\b",
    )

    texts = df[text_col].fillna("").astype(str)
    labels = df[label_col].astype(str).str.lower().to_numpy()

    X = vectorizer.fit_transform(texts)
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    all_results = []

    for emotion in emotions:
        target_mask = labels == emotion
        other_mask = labels != emotion

        num_target = target_mask.sum()
        num_other = other_mask.sum()

        if num_target == 0:
            print(f"Warning: no examples found for {emotion}")
            continue

        binary_targets = target_mask.astype(int)

        chi2_scores, p_values = chi2(X, binary_targets)
        chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0)

        target_document_counts = np.asarray(
            X[target_mask].sum(axis=0)
        ).ravel()

        other_document_counts = np.asarray(
            X[other_mask].sum(axis=0)
        ).ravel()

        target_prevalence = target_document_counts / max(num_target, 1)
        other_prevalence = other_document_counts / max(num_other, 1)

        # Small smoothing constant prevents division by zero.
        smoothing = 1.0 / max(len(df), 1)

        prevalence_ratio = (
            target_prevalence + smoothing
        ) / (
            other_prevalence + smoothing
        )

        emotion_results = pd.DataFrame(
            {
                "emotion": emotion,
                "ngram": feature_names,
                "chi2_score": chi2_scores,
                "p_value": p_values,
                "target_document_count": target_document_counts,
                "other_document_count": other_document_counts,
                "target_prevalence": target_prevalence,
                "other_prevalence": other_prevalence,
                "prevalence_ratio": prevalence_ratio,
            }
        )

        emotion_results = emotion_results[
            (
                emotion_results["target_document_count"]
                >= min_target_documents
            )
            & (
                emotion_results["prevalence_ratio"]
                >= min_prevalence_ratio
            )
        ]

        emotion_results = (
            emotion_results
            .sort_values(
                ["chi2_score", "prevalence_ratio"],
                ascending=False,
            )
            .head(top_k_per_emotion)
            .reset_index(drop=True)
        )

        emotion_results["rank"] = (
            np.arange(len(emotion_results)) + 1
        )

        all_results.append(emotion_results)

    if not all_results:
        raise ValueError(
            "No label-correlated n-grams were found. "
            "Try reducing MIN_DF or MIN_PREVALENCE_RATIO."
        )

    shortcut_df = pd.concat(all_results, ignore_index=True)

    return shortcut_df, vectorizer, X


# ============================================================
# Remove shortcut-containing examples
# ============================================================

def filter_shortcut_examples(
    df,
    X,
    vectorizer,
    shortcut_df,
    label_col,
    mode="global",
):
    """
    mode='global':
        Remove an example if it contains any selected shortcut n-gram.

    mode='label_specific':
        Remove an example only if it contains a shortcut n-gram associated
        with its own gold label.
    """

    vocabulary = vectorizer.vocabulary_
    labels = df[label_col].astype(str).str.lower().to_numpy()

    drop_mask = np.zeros(len(df), dtype=bool)

    if mode == "global":
        selected_terms = shortcut_df["ngram"].drop_duplicates().tolist()

        selected_columns = [
            vocabulary[term]
            for term in selected_terms
            if term in vocabulary
        ]

        if selected_columns:
            shortcut_counts = np.asarray(
                X[:, selected_columns].sum(axis=1)
            ).ravel()

            drop_mask = shortcut_counts > 0

    elif mode == "label_specific":
        for emotion, emotion_terms_df in shortcut_df.groupby("emotion"):
            row_indices = np.where(labels == emotion)[0]

            selected_columns = [
                vocabulary[term]
                for term in emotion_terms_df["ngram"]
                if term in vocabulary
            ]

            if len(row_indices) == 0 or not selected_columns:
                continue

            shortcut_counts = np.asarray(
                X[row_indices][:, selected_columns].sum(axis=1)
            ).ravel()

            drop_mask[row_indices] = shortcut_counts > 0

    else:
        raise ValueError(
            "mode must be either 'global' or 'label_specific'"
        )

    filtered_df = df.loc[~drop_mask].copy().reset_index(drop=True)

    removal_df = df.copy()
    removal_df["_removed_as_lexical_shortcut"] = drop_mask

    return filtered_df, removal_df


# ============================================================
# Filtering summary
# ============================================================

def create_filtering_summary(
    removal_df,
    label_col,
):
    summary = (
        removal_df
        .groupby(label_col)
        .agg(
            original_count=(
                "_removed_as_lexical_shortcut",
                "size",
            ),
            removed_count=(
                "_removed_as_lexical_shortcut",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["remaining_count"] = (
        summary["original_count"] - summary["removed_count"]
    )

    summary["removal_rate"] = (
        summary["removed_count"] / summary["original_count"]
    )

    return summary


# ============================================================
# Balance labels to control label priors
# ============================================================

def create_balanced_dataset(
    df,
    label_col,
    emotions,
    random_seed=42,
):
    available_df = df[
        df[label_col].astype(str).str.lower().isin(emotions)
    ].copy()

    class_counts = (
        available_df[label_col]
        .astype(str)
        .str.lower()
        .value_counts()
    )

    missing_emotions = [
        emotion
        for emotion in emotions
        if emotion not in class_counts
    ]

    if missing_emotions:
        raise ValueError(
            f"No remaining examples for: {missing_emotions}"
        )

    samples_per_emotion = int(
        min(class_counts[emotion] for emotion in emotions)
    )

    balanced_parts = []

    for emotion in emotions:
        emotion_df = available_df[
            available_df[label_col].astype(str).str.lower()
            == emotion
        ]

        sampled_df = emotion_df.sample(
            n=samples_per_emotion,
            random_state=random_seed,
        )

        balanced_parts.append(sampled_df)

    balanced_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )

    return balanced_df


# ============================================================
# Full pipeline
# ============================================================

def run_lexical_shortcut_control():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV_PATH)

    required_columns = {TEXT_COL, LABEL_COL}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[df[LABEL_COL].isin(EMOTIONS)].reset_index(drop=True)

    print("Original class counts:")
    print(df[LABEL_COL].value_counts().reindex(EMOTIONS))
    print()

    shortcut_df, vectorizer, X = identify_label_correlated_ngrams(
        df=df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        emotions=EMOTIONS,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        top_k_per_emotion=TOP_K_PER_EMOTION,
        min_prevalence_ratio=MIN_PREVALENCE_RATIO,
        min_target_documents=MIN_TARGET_DOCUMENTS,
    )

    shortcut_path = OUTPUT_DIR / "label_correlated_ngrams.csv"
    shortcut_df.to_csv(shortcut_path, index=False)

    # Strongest control: remove any selected shortcut term.
    global_filtered_df, global_removal_df = filter_shortcut_examples(
        df=df,
        X=X,
        vectorizer=vectorizer,
        shortcut_df=shortcut_df,
        label_col=LABEL_COL,
        mode="global",
    )

    global_summary_df = create_filtering_summary(
        global_removal_df,
        LABEL_COL,
    )

    global_filtered_path = (
        OUTPUT_DIR / "dataset_without_global_lexical_shortcuts.csv"
    )
    global_summary_path = (
        OUTPUT_DIR / "global_filtering_summary.csv"
    )

    global_filtered_df.to_csv(global_filtered_path, index=False)
    global_summary_df.to_csv(global_summary_path, index=False)

    # Less aggressive control: remove shortcuts associated with gold label.
    label_filtered_df, label_removal_df = filter_shortcut_examples(
        df=df,
        X=X,
        vectorizer=vectorizer,
        shortcut_df=shortcut_df,
        label_col=LABEL_COL,
        mode="label_specific",
    )

    label_summary_df = create_filtering_summary(
        label_removal_df,
        LABEL_COL,
    )

    label_filtered_path = (
        OUTPUT_DIR / "dataset_without_label_specific_shortcuts.csv"
    )
    label_summary_path = (
        OUTPUT_DIR / "label_specific_filtering_summary.csv"
    )

    label_filtered_df.to_csv(label_filtered_path, index=False)
    label_summary_df.to_csv(label_summary_path, index=False)

    # Balanced version controls label priors.
    balanced_global_df = create_balanced_dataset(
        df=global_filtered_df,
        label_col=LABEL_COL,
        emotions=EMOTIONS,
        random_seed=RANDOM_SEED,
    )

    balanced_global_path = (
        OUTPUT_DIR
        / "balanced_dataset_without_global_lexical_shortcuts.csv"
    )

    balanced_global_df.to_csv(
        balanced_global_path,
        index=False,
    )

    print("Selected shortcut terms:")
    for emotion in EMOTIONS:
        terms = shortcut_df.loc[
            shortcut_df["emotion"] == emotion,
            "ngram",
        ].tolist()

        print(f"{emotion}: {terms}")

    print("\nGlobal filtering summary:")
    print(global_summary_df.to_string(index=False))

    print("\nBalanced filtered class counts:")
    print(
        balanced_global_df[LABEL_COL]
        .value_counts()
        .reindex(EMOTIONS)
    )

    print("\nSaved outputs:")
    print(shortcut_path)
    print(global_filtered_path)
    print(global_summary_path)
    print(label_filtered_path)
    print(label_summary_path)
    print(balanced_global_path)

    return {
        "shortcut_terms": shortcut_df,
        "global_filtered": global_filtered_df,
        "label_specific_filtered": label_filtered_df,
        "balanced_global_filtered": balanced_global_df,
        "balanced_global_path": str(balanced_global_path),
    }


if __name__ == "__main__":
    results = run_lexical_shortcut_control()

