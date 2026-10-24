import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.mixture import GaussianMixture


EMOTIONS = ["joy", "surprise", "disgust", "anger", "sadness", "fear"]


def get_logit_cols(emotions=EMOTIONS):
    return [f"logit_{e}" for e in emotions]


def load_logits_df(path):
    df = pd.read_csv(path)
    return df


def filter_valid_logit_rows(
    df,
    emotions=EMOTIONS,
    min_total_abs_logit=1e-4,
    nonzero_threshold=1e-8,
    min_nonzero_logits=2,
    verbose=True,
):
    """
    Remove rows with too weak total logit effect or too few affected emotions.
    """
    df = df.copy()
    logit_cols = get_logit_cols(emotions)

    W = df[logit_cols].abs().to_numpy()
    total_abs_logit = W.sum(axis=1)
    num_nonzero_logits = (W > nonzero_threshold).sum(axis=1)

    valid_mask = (
        (total_abs_logit > min_total_abs_logit)
        & (num_nonzero_logits >= min_nonzero_logits)
    )

    if verbose:
        print("Original rows:", len(df))
        print("Valid rows:", int(valid_mask.sum()))
        print("Excluded rows:", int((~valid_mask).sum()))

    df = df.loc[valid_mask].copy()

    return df


def add_entropy_columns(
    df,
    emotions=EMOTIONS,
    eps=1e-12,
    nonzero_threshold=1e-8,
):
    """
    Compute normalized entropy over absolute emotion logit effects.
    """
    df = df.copy()
    logit_cols = get_logit_cols(emotions)

    W = df[logit_cols].abs().to_numpy()

    P = W / (W.sum(axis=1, keepdims=True) + eps)

    entropy = -(P * np.log(P + eps)).sum(axis=1)
    normalized_entropy = entropy / np.log(len(emotions))

    df["total_abs_logit"] = W.sum(axis=1)
    df["num_nonzero_logits"] = (W > nonzero_threshold).sum(axis=1)
    df["entropy"] = entropy
    df["normalized_entropy"] = normalized_entropy

    return df


def plot_entropy_histogram(
    df,
    bins=40,
    figsize=(7, 4),
    title="Distribution of Emotion-Logit Entropy",
    save_path=None,
    show=True,
):
    plt.figure(figsize=figsize)

    plt.hist(
        df["normalized_entropy"],
        bins=bins,
    )

    plt.xlabel("Normalized Entropy")
    plt.ylabel("Count")
    plt.title(title)
    plt.xlim(0, 1)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def estimate_entropy_threshold_gmm(
    df,
    n_components=2,
    random_state=0,
    num_points=10000,
    verbose=True,
):
    """
    Fit a 2-component GMM to normalized entropy values and return the
    intersection threshold between the two Gaussian components.
    """
    vals = df["normalized_entropy"].values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
    )
    gmm.fit(vals)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    order = np.argsort(means)

    m1, m2 = means[order]
    s1, s2 = stds[order]
    w1, w2 = weights[order]

    xs = np.linspace(0, 1, num_points)

    pdf1 = w1 * norm.pdf(xs, m1, s1)
    pdf2 = w2 * norm.pdf(xs, m2, s2)

    intersection_idx = np.argmin(np.abs(pdf1 - pdf2))
    threshold = xs[intersection_idx]

    if verbose:
        print("GMM entropy threshold:", threshold)
        print("Low-entropy component mean:", m1)
        print("High-entropy component mean:", m2)

    gmm_info = {
        "threshold": threshold,
        "means": means,
        "stds": stds,
        "weights": weights,
        "sorted_means": (m1, m2),
        "sorted_stds": (s1, s2),
        "sorted_weights": (w1, w2),
        "xs": xs,
        "pdf1": pdf1,
        "pdf2": pdf2,
    }

    return threshold, gmm_info


def plot_entropy_gmm(
    df,
    gmm_info,
    bins=40,
    figsize=(7, 4),
    title="GMM Fit on Normalized Entropy",
    save_path=None,
    show=True,
):
    xs = gmm_info["xs"]
    pdf1 = gmm_info["pdf1"]
    pdf2 = gmm_info["pdf2"]
    threshold = gmm_info["threshold"]

    vals = df["normalized_entropy"].values

    plt.figure(figsize=figsize)

    plt.hist(
        vals,
        bins=bins,
        density=True,
        alpha=0.5,
    )

    plt.plot(xs, pdf1, label="Low-entropy component")
    plt.plot(xs, pdf2, label="High-entropy component")

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.4f}",
    )

    plt.xlabel("Normalized Entropy")
    plt.ylabel("Density")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def add_entropy_group(
    df,
    entropy_threshold,
    specific_label="emotion_specific",
    broad_label="shared_broad",
):
    """
    Add entropy group labels.
    Low entropy means the feature mostly affects a small number of emotions.
    High entropy means the feature broadly affects many emotions.
    """
    df = df.copy()

    df["entropy_group"] = np.where(
        df["normalized_entropy"] < entropy_threshold,
        specific_label,
        broad_label,
    )

    return df


def add_max_abs_logit(
    df,
    emotions=EMOTIONS,
):
    df = df.copy()
    logit_cols = get_logit_cols(emotions)

    df["max_abs_logit"] = df[logit_cols].abs().max(axis=1)

    return df


def add_max_positive_logit(
    df,
    emotions=EMOTIONS,
):
    """
    Add max positive signed logit effect.

    Negative logit effects are ignored.
    """
    df = df.copy()
    logit_cols = get_logit_cols(emotions)

    W_signed = df[logit_cols].to_numpy()
    W_positive = np.maximum(W_signed, 0)

    df["max_positive_logit"] = W_positive.max(axis=1)

    return df


def get_emotion_specific_positive_features(
    df,
    entropy_group_col="entropy_group",
    specific_label="emotion_specific",
    require_positive=True,
    emotions=EMOTIONS,
):
    df_specific = df[df[entropy_group_col] == specific_label].copy()
    df_specific = add_max_positive_logit(df_specific, emotions=emotions)

    if require_positive:
        df_specific = df_specific[df_specific["max_positive_logit"] > 0].copy()

    return df_specific


def plot_max_positive_logit_histogram(
    df_specific,
    bins=50,
    figsize=(7, 4),
    title="Distribution of Max Positive Logit Effect\n(Emotion-Specific Features Only)",
    save_path=None,
    show=True,
):
    plt.figure(figsize=figsize)

    plt.hist(
        df_specific["max_positive_logit"],
        bins=bins,
    )

    plt.xlabel("Max Positive Logit Effect")
    plt.ylabel("Count")
    plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def run_entropy_classification_pipeline(
    input_path,
    output_path,
    emotions=EMOTIONS,
    entropy_threshold=None,
    min_total_abs_logit=1e-4,
    nonzero_threshold=1e-8,
    min_nonzero_logits=2,
    plot_dir=None,
    show_plots=True,
    verbose=True,
):
    """
    Full pipeline:
    1. Load feature logit file
    2. Filter degenerate rows
    3. Compute entropy
    4. Estimate threshold with GMM if threshold is not provided
    5. Assign entropy groups
    6. Add max_abs_logit and max_positive_logit
    7. Save output CSV
    """

    df = load_logits_df(input_path)

    df = filter_valid_logit_rows(
        df,
        emotions=emotions,
        min_total_abs_logit=min_total_abs_logit,
        nonzero_threshold=nonzero_threshold,
        min_nonzero_logits=min_nonzero_logits,
        verbose=verbose,
    )

    df = add_entropy_columns(
        df,
        emotions=emotions,
        nonzero_threshold=nonzero_threshold,
    )

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

        entropy_hist_path = os.path.join(
            plot_dir,
            "normalized_entropy_histogram.png",
        )
    else:
        entropy_hist_path = None

    plot_entropy_histogram(
        df,
        save_path=entropy_hist_path,
        show=show_plots,
    )

    if entropy_threshold is None:
        entropy_threshold, gmm_info = estimate_entropy_threshold_gmm(
            df,
            verbose=verbose,
        )

        if plot_dir is not None:
            gmm_plot_path = os.path.join(
                plot_dir,
                "normalized_entropy_gmm.png",
            )
        else:
            gmm_plot_path = None

        plot_entropy_gmm(
            df,
            gmm_info,
            save_path=gmm_plot_path,
            show=show_plots,
        )
    else:
        gmm_info = None

        if verbose:
            print("Using provided entropy threshold:", entropy_threshold)

    df = add_entropy_group(
        df,
        entropy_threshold=entropy_threshold,
    )

    df = add_max_abs_logit(
        df,
        emotions=emotions,
    )

    df = add_max_positive_logit(
        df,
        emotions=emotions,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    if verbose:
        print("Saved:", output_path)
        print(df["entropy_group"].value_counts())

        df_specific = get_emotion_specific_positive_features(
            df,
            emotions=emotions,
        )

        print("\nEmotion-specific positive max logit stats:")
        print(df_specific["max_positive_logit"].describe())

    if plot_dir is not None:
        df_specific = get_emotion_specific_positive_features(
            df,
            emotions=emotions,
        )

        positive_logit_plot_path = os.path.join(
            plot_dir,
            "emotion_specific_max_positive_logit_histogram.png",
        )

        plot_max_positive_logit_histogram(
            df_specific,
            save_path=positive_logit_plot_path,
            show=show_plots,
        )

    return df, entropy_threshold