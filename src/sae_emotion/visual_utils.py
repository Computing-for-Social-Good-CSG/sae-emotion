"""
Reusable Plotly visualization utilities for emotion feature graphs.

Main entry point:
    build_emotion_feature_graph(...)

Example:
    from sae_emotion.visual_utils import build_emotion_feature_graph

    fig, features, edges = build_emotion_feature_graph(
        feature_path="data/google-gemma-2-2b-feature-act/auto_selected_unique_features_with_affected_emotions.csv",
        edge_path="data/google-gemma-2-2b-feature-act/emotion_feature_edges/all_affected_feature_edges.csv",
        output_html="data/google-gemma-2-2b-feature-act/visuals/core_features.html",
        show=True,
    )
"""

import os
import re
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

VISUAL_UTILS_VERSION = "nan-safe-plus-publishable-toggle-semantic-2026-05-17-v4"


DEFAULT_EMOTIONS = [
    "joy",
    "surprise",
    "disgust",
    "anger",
    "sadness",
    "fear",
]

DEFAULT_EMOTION_COLORS = {
    "joy": "#F58518",
    "surprise": "#72B7B2",
    "disgust": "#54A24B",
    "anger": "#E45756",
    "sadness": "#4C78A8",
    "fear": "#B279A2",
}


def parse_affected_emotions(x, emotions: Optional[Sequence[str]] = None) -> List[str]:
    """Parse the affected_emotions column into an ordered list of emotion names."""
    if emotions is None:
        emotions = DEFAULT_EMOTIONS
    if pd.isna(x):
        return []

    x = str(x).strip()
    x = (
        x.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    if x == "":
        return []

    parsed = [e.strip() for e in x.split(",")]
    return [e for e in emotions if e in parsed]


def _require_columns(df: pd.DataFrame, required_cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _prepare_features(
    features: pd.DataFrame,
    emotions: Sequence[str],
) -> pd.DataFrame:
    _require_columns(
        features,
        ["layer_idx", "feature_idx", "affected_emotions"],
        "features",
    )

    features = features[features["affected_emotions"].notna()].copy()

    features["feature_idx"] = features["feature_idx"].astype(int)
    features["layer_idx"] = features["layer_idx"].astype(int)

    features["node_id"] = (
        "L"
        + features["layer_idx"].astype(str)
        + "-F"
        + features["feature_idx"].astype(str)
    )

    features["affected_emotion_list"] = features["affected_emotions"].apply(
        lambda x: parse_affected_emotions(x, emotions)
    )

    features["num_affected_emotions"] = features["affected_emotion_list"].apply(len)
    features["is_shared"] = features["num_affected_emotions"] >= 2
    features["affected_emotions_clean"] = features["affected_emotion_list"].apply(
        lambda xs: ", ".join(xs)
    )

    logit_cols = [f"logit_{e}" for e in emotions if f"logit_{e}" in features.columns]
    if len(logit_cols) == 0:
        raise ValueError("No logit_* columns found in features.")

    if "max_abs_logit" not in features.columns:
        features["max_abs_logit"] = features[logit_cols].abs().max(axis=1)

    def get_top_two_logit_emotions(row: pd.Series) -> Tuple[str, str]:
        vals = []
        for emo in emotions:
            col = f"logit_{emo}"
            if col in row.index:
                vals.append((emo, abs(row[col])))

        vals = sorted(vals, key=lambda x: x[1], reverse=True)
        top_emo = vals[0][0]
        second_emo = vals[1][0] if len(vals) > 1 else top_emo
        return top_emo, second_emo

    features[["top_logit_emotion", "second_logit_emotion"]] = features.apply(
        lambda row: pd.Series(get_top_two_logit_emotions(row)),
        axis=1,
    )

    return features


def _prepare_edges(edges: pd.DataFrame, valid_nodes: set) -> pd.DataFrame:
    _require_columns(
        edges,
        [
            "source_layer",
            "source_feature",
            "target_layer",
            "target_feature",
            "mean_effect",
            "clean_target_mean",
        ],
        "edges",
    )

    edges = edges.copy()

    # Some resumed or cross-model edge files can contain blank / NaN effect rows.
    # Plotly line.width cannot be NaN, so sanitize all numeric columns used in
    # width, dash, and hover construction before plotting.
    int_cols = ["source_feature", "target_feature", "source_layer", "target_layer"]
    for col in int_cols:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")

    edges = edges.dropna(subset=int_cols).copy()

    for col in int_cols:
        edges[col] = edges[col].astype(int)

    numeric_cols = [
        "mean_effect",
        "clean_target_mean",
        "ablated_target_mean",
        "normalized_effect",
        "drop_percent",
    ]
    for col in numeric_cols:
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")
            edges[col] = edges[col].replace([np.inf, -np.inf], np.nan)

    edges["source_node_id"] = (
        "L" + edges["source_layer"].astype(str) + "-F" + edges["source_feature"].astype(str)
    )
    edges["target_node_id"] = (
        "L" + edges["target_layer"].astype(str) + "-F" + edges["target_feature"].astype(str)
    )

    edges = edges[
        edges["source_node_id"].isin(valid_nodes)
        & edges["target_node_id"].isin(valid_nodes)
    ].copy()

    edges["mean_effect"] = edges["mean_effect"].fillna(0.0)
    edges["clean_target_mean"] = edges["clean_target_mean"].fillna(0.0)
    edges["abs_effect"] = edges["mean_effect"].abs()

    if "ablated_target_mean" not in edges.columns:
        edges["ablated_target_mean"] = edges["clean_target_mean"] - edges["mean_effect"]
    else:
        edges["ablated_target_mean"] = edges["ablated_target_mean"].fillna(
            edges["clean_target_mean"] - edges["mean_effect"]
        )

    if "normalized_effect" in edges.columns:
        edges["drop_percent"] = edges["normalized_effect"] * 100
    else:
        denom = edges["clean_target_mean"].abs() + 1e-8
        edges["drop_percent"] = edges["mean_effect"] / denom * 100

    edges["drop_percent"] = (
        pd.to_numeric(edges["drop_percent"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    edges["abs_drop_percent"] = edges["drop_percent"].abs()

    return edges


def _safe_float(x, default: float = 0.0) -> float:
    """Return a finite Python float. Plotly rejects NaN/inf numeric values."""
    try:
        val = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(val):
        return float(default)
    return float(val)


def _safe_fmt(x, digits: int = 3, default: float = 0.0) -> str:
    val = _safe_float(x, default=default)
    return f"{val:.{digits}f}"


def _format_logit_weights(row: pd.Series, emotions: Sequence[str]) -> str:
    lines = []
    for emo in emotions:
        col = f"logit_{emo}"
        if col in row.index:
            lines.append(f"{emo}: {_safe_fmt(row[col], digits=4)}")
    return "<br>".join(lines)


def _format_edges(edges_list: List[Tuple[str, int, float, float, float, float]]) -> str:
    if len(edges_list) == 0:
        return "None"

    lines = []
    for other, layer, eff, clean, ablated, drop in edges_list:
        eff = _safe_float(eff)
        clean = _safe_float(clean)
        ablated = _safe_float(ablated)
        drop = _safe_float(drop)
        sign = "+" if eff >= 0 else ""
        lines.append(
            f"{other} (L{layer}): "
            f"clean {clean:.3f} → {ablated:.3f}, "
            f"Δ {sign}{eff:.3f}, "
            f"drop {drop:.1f}%"
        )

    return "<br>".join(lines)


def _add_layout_positions(
    features: pd.DataFrame,
    emotions: Sequence[str],
    emotion_band_gap: float,
    shared_secondary_pull: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    features = features.sort_values(["layer_idx", "feature_idx"]).reset_index(drop=True)

    emotion_y_base = {emo: i * emotion_band_gap for i, emo in enumerate(emotions)}

    def get_y_base(row: pd.Series) -> float:
        primary = row["top_logit_emotion"]
        secondary = row["second_logit_emotion"]

        primary_y = emotion_y_base[primary]
        if not row["is_shared"]:
            return primary_y

        secondary_y = emotion_y_base[secondary]
        return primary_y * (1.0 - shared_secondary_pull) + secondary_y * shared_secondary_pull

    def get_layout_group(row: pd.Series) -> str:
        primary = row["top_logit_emotion"]
        if row["is_shared"]:
            secondary = row["second_logit_emotion"]
            return f"{primary}_shared_with_{secondary}"
        return primary

    features["layout_group"] = features.apply(get_layout_group, axis=1)
    features["y_base"] = features.apply(get_y_base, axis=1)

    features["within_group_idx"] = features.groupby(["layout_group", "layer_idx"]).cumcount()
    features["group_count"] = features.groupby(["layout_group", "layer_idx"])[
        "node_id"
    ].transform("count")

    features["y_pos"] = (
        features["y_base"]
        + features["within_group_idx"]
        - (features["group_count"] - 1) / 2
    )

    rng = np.random.default_rng(random_seed)
    features["x_pos"] = features["layer_idx"] + rng.uniform(-0.06, 0.06, len(features))
    features["y_pos"] = features["y_pos"] + rng.uniform(-0.06, 0.06, len(features))

    return features, emotion_y_base


def _add_node_sizes(
    features: pd.DataFrame,
    node_min_size: float,
    node_max_size: float,
) -> pd.DataFrame:
    features = features.copy()

    v = pd.to_numeric(features["max_abs_logit"], errors="coerce").to_numpy(dtype=float)
    v = np.nan_to_num(v, nan=1e-6, posinf=1e-6, neginf=1e-6)
    v = np.clip(v, 1e-6, None)
    v_log = np.log(v)

    denom = v_log.max() - v_log.min()
    if not np.isfinite(denom) or denom <= 1e-8:
        v_norm = np.zeros_like(v_log)
    else:
        v_norm = (v_log - v_log.min()) / denom

    features["node_size"] = node_min_size + v_norm * (node_max_size - node_min_size)

    return features


def build_emotion_feature_graph(
    feature_path: str,
    edge_path: str,
    output_html: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    emotion_colors: Optional[Dict[str, str]] = None,
    node_min_size: float = 18,
    node_max_size: float = 48,
    edge_opacity: float = 0.18,
    random_seed: int = 42,
    emotion_band_gap: float = 14.0,
    shared_secondary_pull: float = 0.15,
    width: int = 1600,
    height: int = 1600,
    title: Optional[str] = None,
    show: bool = False,
    verbose: bool = True,
) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Build the emotion feature graph figure.

    Args:
        feature_path: CSV containing feature rows. Must include layer_idx, feature_idx,
            affected_emotions, and logit_* columns.
        edge_path: CSV containing edge rows. Must include source_layer, source_feature,
            target_layer, target_feature, mean_effect, and clean_target_mean.
        output_html: Optional path to save the interactive HTML.
        emotions: Emotion order used for parsing, layout, and plotting.
        emotion_colors: Mapping from emotion to color.
        node_min_size: Minimum node marker size.
        node_max_size: Maximum node marker size.
        edge_opacity: Edge opacity.
        random_seed: Random seed for layout jitter.
        emotion_band_gap: Vertical spacing between emotion bands.
        shared_secondary_pull: How much shared nodes are pulled toward the second emotion band.
        width: Figure width.
        height: Figure height.
        title: Optional Plotly figure title.
        show: Whether to call fig.show().
        verbose: Whether to print basic node and edge counts.

    Returns:
        fig, features, edges
    """
    if emotion_colors is None:
        emotion_colors = DEFAULT_EMOTION_COLORS

    features = pd.read_csv(feature_path)
    edges = pd.read_csv(edge_path)

    features = _prepare_features(features, emotions)
    valid_nodes = set(features["node_id"])
    edges = _prepare_edges(edges, valid_nodes)

    if verbose:
        print(f"Nodes: {len(features)}")
        print(f"Edges: {len(edges)}")
        print("Shared nodes:", int(features["is_shared"].sum()))

    features, emotion_y_base = _add_layout_positions(
        features=features,
        emotions=emotions,
        emotion_band_gap=emotion_band_gap,
        shared_secondary_pull=shared_secondary_pull,
        random_seed=random_seed,
    )
    features = _add_node_sizes(features, node_min_size, node_max_size)

    pos = {
        row["node_id"]: (row["x_pos"], row["y_pos"])
        for _, row in features.iterrows()
    }

    incoming = {node_id: [] for node_id in features["node_id"]}
    outgoing = {node_id: [] for node_id in features["node_id"]}

    for _, row in edges.iterrows():
        s = row["source_node_id"]
        t = row["target_node_id"]

        if s not in pos or t not in pos:
            continue

        effect = row["mean_effect"]
        clean = row["clean_target_mean"]
        ablated = row["ablated_target_mean"]
        drop = row["drop_percent"]

        outgoing[s].append((t, row["target_layer"], effect, clean, ablated, drop))
        incoming[t].append((s, row["source_layer"], effect, clean, ablated, drop))

    for k in incoming:
        incoming[k].sort(key=lambda x: x[1])
    for k in outgoing:
        outgoing[k].sort(key=lambda x: x[1])

    edge_traces = []
    max_drop = _safe_float(edges["abs_drop_percent"].max() if len(edges) > 0 else 1.0, default=1.0)
    if max_drop <= 0:
        max_drop = 1.0

    for _, row in edges.iterrows():
        s = row["source_node_id"]
        t = row["target_node_id"]

        if s not in pos or t not in pos:
            continue

        x0, y0 = pos[s]
        x1, y1 = pos[t]

        drop = abs(_safe_float(row.get("drop_percent", 0.0), default=0.0))
        effect = _safe_float(row.get("mean_effect", 0.0), default=0.0)

        edge_width = _safe_float(0.4 + 4.0 * (drop / (max_drop + 1e-8)), default=0.4)
        if edge_width < 0:
            edge_width = 0.4

        dash = "solid" if effect >= 0 else "dot"
        color = emotion_colors.get(row.get("conditioning_emotion", ""), "#777")

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=float(edge_width), dash=dash, color=color),
                opacity=edge_opacity,
                hoverinfo="text",
                text=(
                    f"{s} → {t}<br>"
                    f"Conditioning emotion: {row.get('conditioning_emotion', '')}<br>"
                    f"clean {_safe_fmt(row.get('clean_target_mean', 0.0), 3)} "
                    f"→ {_safe_fmt(row.get('ablated_target_mean', 0.0), 3)}<br>"
                    f"Δ {_safe_fmt(row.get('mean_effect', 0.0), 3)}, "
                    f"drop {_safe_fmt(row.get('drop_percent', 0.0), 2)}%"
                ),
                showlegend=False,
            )
        )

    node_traces = []

    for emotion in emotions:
        sub = features[
            (features["num_affected_emotions"] == 1)
            & (features["affected_emotion_list"].apply(lambda xs: emotion in xs))
        ]

        if len(sub) == 0:
            continue

        hover = []
        for _, row in sub.iterrows():
            node_id = row["node_id"]
            desc = "<br>".join(
                textwrap.wrap(str(row.get("description", "")), width=80)
            )
            logit_weights = _format_logit_weights(row, emotions)

            hover.append(
                f"<b>{node_id}</b><br>"
                f"Layer: {row['layer_idx']}<br>"
                f"Feature: {row['feature_idx']}<br>"
                f"<br><b>Affected emotions:</b> {row['affected_emotions_clean']}<br>"
                f"<br><b>Shared node:</b> No<br>"
                f"<br><b>Logit weights:</b><br>{logit_weights}<br>"
                f"<br><b>Description:</b><br>{desc}<br>"
                f"<br><b>Incoming:</b><br>{_format_edges(incoming[node_id])}<br>"
                f"<br><b>Outgoing:</b><br>{_format_edges(outgoing[node_id])}"
            )

        node_traces.append(
            go.Scatter(
                x=sub["x_pos"],
                y=sub["y_pos"],
                mode="markers+text",
                text=sub["node_id"],
                textposition="middle center",
                textfont=dict(size=7, color="white"),
                hoverinfo="text",
                hovertext=hover,
                marker=dict(
                    size=sub["node_size"],
                    color=emotion_colors[emotion],
                    line=dict(width=1, color="black"),
                ),
                name=emotion,
            )
        )

    shared_nodes = features[features["is_shared"]].copy()
    shared_traces = []

    if len(shared_nodes) > 0:
        shared_hover = []

        for _, row in shared_nodes.iterrows():
            node_id = row["node_id"]
            desc = "<br>".join(
                textwrap.wrap(str(row.get("description", "")), width=80)
            )
            logit_weights = _format_logit_weights(row, emotions)

            shared_hover.append(
                f"<b>{node_id} (SHARED)</b><br>"
                f"Layer: {row['layer_idx']}<br>"
                f"Feature: {row['feature_idx']}<br>"
                f"<br><b>Affected emotions:</b> {row['affected_emotions_clean']}<br>"
                f"<b># affected emotions:</b> {row['num_affected_emotions']}<br>"
                f"<br><b>Primary emotion:</b> {row['top_logit_emotion']}<br>"
                f"<b>Border emotion:</b> {row['second_logit_emotion']}<br>"
                f"<br><b>Logit weights:</b><br>{logit_weights}<br>"
                f"<br><b>Description:</b><br>{desc}<br>"
                f"<br><b>Incoming:</b><br>{_format_edges(incoming[node_id])}<br>"
                f"<br><b>Outgoing:</b><br>{_format_edges(outgoing[node_id])}"
            )

        shared_traces.append(
            go.Scatter(
                x=shared_nodes["x_pos"],
                y=shared_nodes["y_pos"],
                mode="markers+text",
                text=shared_nodes["node_id"],
                textposition="middle center",
                textfont=dict(size=7, color="white"),
                hoverinfo="text",
                hovertext=shared_hover,
                marker=dict(
                    size=shared_nodes["node_size"],
                    color=shared_nodes["top_logit_emotion"].map(emotion_colors),
                    line=dict(
                        width=3,
                        color=shared_nodes["second_logit_emotion"].map(emotion_colors),
                    ),
                ),
                name="shared features",
                showlegend=True,
            )
        )

    annotations = []
    for emotion in emotions:
        annotations.append(
            dict(
                x=1.02,
                y=emotion_y_base[emotion],
                xref="paper",
                yref="y",
                text=f"<b>{emotion}</b>",
                showarrow=False,
                font=dict(size=14, color=emotion_colors[emotion]),
                xanchor="left",
            )
        )

    if title is None:
        title = (
            "All Emotion Features (All Edges)<br>"
            "<sup>"
            "Shared nodes use fill color for the strongest absolute logit emotion "
            "and border color for the second strongest."
            "</sup>"
        )

    fig = go.Figure(data=edge_traces + node_traces + shared_traces)
    fig.update_layout(
        title=title,
        xaxis=dict(title="Layer"),
        yaxis=dict(showticklabels=False),
        width=width,
        height=height,
        annotations=annotations,
    )

    if show:
        fig.show()

    if output_html is not None:
        output_dir = os.path.dirname(output_html)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.write_html(output_html)
        if verbose:
            print("Saved to:", output_html)

    return fig, features, edges


# ============================================================
# Publishable analysis utilities
# ============================================================

FEATURE_ID_COLS = ["layer_idx", "feature_idx"]
DESCRIPTION_CANDIDATES = [
    "description",
    "feature_description",
    "explanation",
    "neuronpedia_description",
    "llm_description",
]

EMOTION_KEYWORDS = {
    "joy": [
        "joy", "happy", "happiness", "excited", "excitement",
        "delight", "delighted", "celebration", "celebrate",
        "pleasure", "cheerful", "glad", "positive",
    ],
    "surprise": [
        "surprise", "surprised", "unexpected", "shock", "shocked",
        "astonishment", "amazement", "amazed", "sudden",
    ],
    "disgust": [
        "disgust", "disgusting", "revulsion", "gross",
        "offensive", "repulsed", "repulsion",
    ],
    "anger": [
        "anger", "angry", "rage", "frustration", "frustrated",
        "hostility", "resentment", "furious", "annoyance",
        "irritation",
    ],
    "sadness": [
        "sad", "sadness", "grief", "mourning", "loss",
        "depression", "depressed", "sorrow", "heartbreak",
        "loneliness", "lonely", "disappointment",
    ],
    "fear": [
        "fear", "afraid", "scared", "panic", "terror",
        "anxiety", "anxious", "threat", "danger", "worry",
        "worried",
    ],
}


def _ensure_output_dir(output_dir: Optional[str]) -> Optional[str]:
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_csv_if_requested(df: pd.DataFrame, output_dir: Optional[str], filename: str) -> None:
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, filename), index=False)


def _save_matrix_csv_if_requested(df: pd.DataFrame, output_dir: Optional[str], filename: str) -> None:
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, filename))


def find_description_col(df: pd.DataFrame, candidates: Sequence[str] = DESCRIPTION_CANDIDATES) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Cannot find description column. Tried: {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def find_logit_cols(
    df: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
) -> Dict[str, str]:
    cols = {}

    for e in emotions:
        candidates = [
            f"logit_{e}",
            f"{e}_logit",
            f"logit_effect_{e}",
            f"{e}_logit_effect",
        ]
        found = [c for c in candidates if c in df.columns]
        if found:
            cols[e] = found[0]

    if len(cols) == 0:
        raise ValueError("No emotion logit columns found.")

    return cols


def safe_feature_key(df: pd.DataFrame, feature_id_cols: Sequence[str] = FEATURE_ID_COLS) -> pd.Series:
    missing = [c for c in feature_id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature id columns: {missing}")

    layer_col, feature_col = feature_id_cols
    return (
        df[layer_col].astype(int).astype(str)
        + "-F"
        + df[feature_col].astype(int).astype(str)
    )


def normalized_entropy(values, eps: float = 1e-12) -> float:
    values = np.asarray(values, dtype=float)
    values = np.abs(values)
    total = values.sum()

    if total <= eps or len(values) == 0:
        return np.nan

    p = values / (total + eps)
    ent = -(p * np.log(p + eps)).sum()
    return float(ent / np.log(len(values))) if len(values) > 1 else 0.0


def gini(values) -> float:
    values = np.asarray(values, dtype=float)
    values = np.abs(values)

    if len(values) == 0:
        return np.nan
    if np.all(values == 0):
        return 0.0

    values = np.sort(values)
    n = len(values)
    return float(
        (2 * np.arange(1, n + 1) @ values) / (n * values.sum())
        - (n + 1) / n
    )


def get_primary_emotion(row: pd.Series, logit_cols: Dict[str, str]) -> str:
    vals = {e: abs(_safe_float(row[col], default=0.0)) for e, col in logit_cols.items()}
    return max(vals.items(), key=lambda x: x[1])[0]


def run_emotion_concentration_analysis(
    df: pd.DataFrame,
    logit_cols: Optional[Dict[str, str]] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    output_dir = _ensure_output_dir(output_dir)
    if logit_cols is None:
        logit_cols = find_logit_cols(df, emotions=emotions)

    rows = []

    for layer_idx, layer_df in df.groupby("layer_idx"):
        for emotion, col in logit_cols.items():
            tmp = layer_df.copy()
            tmp["abs_logit"] = pd.to_numeric(tmp[col], errors="coerce").abs().fillna(0.0)
            tmp = tmp[tmp["abs_logit"] > 0].sort_values("abs_logit", ascending=False)

            n_features = len(tmp)
            total = tmp["abs_logit"].sum()

            if n_features == 0 or total == 0:
                rows.append({
                    "layer_idx": layer_idx,
                    "emotion": emotion,
                    "n_features": 0,
                    "total_abs_logit": 0.0,
                    "features_for_50pct": np.nan,
                    "features_for_80pct": np.nan,
                    "features_for_90pct": np.nan,
                    "top1_share": np.nan,
                    "top3_share": np.nan,
                    "top5_share": np.nan,
                    "gini": np.nan,
                    "normalized_entropy": np.nan,
                })
                continue

            vals = tmp["abs_logit"].to_numpy()
            cum = np.cumsum(vals) / total

            def k_for(p: float) -> int:
                return int(np.searchsorted(cum, p) + 1)

            rows.append({
                "layer_idx": layer_idx,
                "emotion": emotion,
                "n_features": n_features,
                "total_abs_logit": float(total),
                "features_for_50pct": k_for(0.50),
                "features_for_80pct": k_for(0.80),
                "features_for_90pct": k_for(0.90),
                "top1_share": float(vals[:1].sum() / total),
                "top3_share": float(vals[:3].sum() / total),
                "top5_share": float(vals[:5].sum() / total),
                "gini": gini(vals),
                "normalized_entropy": normalized_entropy(vals),
            })

    out = pd.DataFrame(rows)
    _save_csv_if_requested(out, output_dir, "emotion_concentration_by_layer.csv")
    return out


def plot_concentration(
    conc_df: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
    show: bool = True,
):
    output_dir = _ensure_output_dir(output_dir)
    figs = []

    fig = plt.figure(figsize=(10, 6))
    for emotion in emotions:
        sub = conc_df[conc_df["emotion"] == emotion]
        if len(sub) > 0:
            plt.plot(sub["layer_idx"], sub["features_for_80pct"], marker="o", label=emotion)
    plt.xlabel("Layer")
    plt.ylabel("# Features for 80%")
    plt.title("Emotion Concentration by Layer")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "features_for_80pct_by_layer.png"), dpi=300)
    if show:
        plt.show()
    figs.append(fig)

    fig = plt.figure(figsize=(10, 6))
    for emotion in emotions:
        sub = conc_df[conc_df["emotion"] == emotion]
        if len(sub) > 0:
            plt.plot(sub["layer_idx"], sub["gini"], marker="o", label=emotion)
    plt.xlabel("Layer")
    plt.ylabel("Gini")
    plt.title("Logit Concentration Gini by Layer")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "gini_by_layer.png"), dpi=300)
    if show:
        plt.show()
    figs.append(fig)

    return figs


def run_emotion_overlap_matrix_from_affected(
    df: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
):
    output_dir = _ensure_output_dir(output_dir)
    df = df.copy()
    df["feature_key"] = safe_feature_key(df)

    if "affected_emotions" not in df.columns:
        raise ValueError("CSV must contain an affected_emotions column.")

    emotion_sets = {e: set() for e in emotions}

    for _, row in df.iterrows():
        affected = parse_affected_emotions(row["affected_emotions"], emotions=emotions)
        for e in affected:
            emotion_sets[e].add(row["feature_key"])

    rows = []
    for e1 in emotions:
        for e2 in emotions:
            s1 = emotion_sets[e1]
            s2 = emotion_sets[e2]
            inter = s1 & s2
            union = s1 | s2
            rows.append({
                "emotion_1": e1,
                "emotion_2": e2,
                "n_emotion_1": len(s1),
                "n_emotion_2": len(s2),
                "n_overlap": len(inter),
                "jaccard": len(inter) / len(union) if len(union) > 0 else np.nan,
                "overlap_coefficient": (
                    len(inter) / min(len(s1), len(s2))
                    if min(len(s1), len(s2)) > 0
                    else np.nan
                ),
            })

    overlap_df = pd.DataFrame(rows)
    jaccard_mat = overlap_df.pivot(index="emotion_1", columns="emotion_2", values="jaccard")
    overlap_coef_mat = overlap_df.pivot(index="emotion_1", columns="emotion_2", values="overlap_coefficient")
    count_mat = overlap_df.pivot(index="emotion_1", columns="emotion_2", values="n_overlap")

    _save_csv_if_requested(overlap_df, output_dir, "emotion_overlap_from_affected_long.csv")
    _save_matrix_csv_if_requested(jaccard_mat, output_dir, "emotion_overlap_jaccard_matrix.csv")
    _save_matrix_csv_if_requested(overlap_coef_mat, output_dir, "emotion_overlap_coefficient_matrix.csv")
    _save_matrix_csv_if_requested(count_mat, output_dir, "emotion_overlap_count_matrix.csv")

    return overlap_df, jaccard_mat, overlap_coef_mat, count_mat


def plot_overlap_matrix(
    mat: pd.DataFrame,
    title: str,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
    show: bool = True,
):
    output_dir = _ensure_output_dir(output_dir)
    mat = mat.reindex(index=emotions, columns=emotions)

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(emotions)), emotions, rotation=45, ha="right")
    plt.yticks(range(len(emotions)), emotions)

    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            val = mat.loc[e1, e2]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center")

    plt.title(title)
    plt.colorbar()
    plt.tight_layout()

    if output_dir is not None:
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
    if show:
        plt.show()

    return fig


def assign_feature_type(description, emotion_keywords: Dict[str, List[str]] = EMOTION_KEYWORDS) -> str:
    text = str(description).lower()

    for _, keywords in emotion_keywords.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return "emotion_semantics"

    return "other_semantics"


def run_semantic_category_analysis(
    df: pd.DataFrame,
    logit_cols: Optional[Dict[str, str]] = None,
    desc_col: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
):
    output_dir = _ensure_output_dir(output_dir)
    df = df.copy()

    if logit_cols is None:
        logit_cols = find_logit_cols(df, emotions=emotions)
    if desc_col is None:
        desc_col = find_description_col(df)

    df["feature_key"] = safe_feature_key(df)
    df["primary_emotion"] = df.apply(lambda r: get_primary_emotion(r, logit_cols), axis=1)
    df["semantic_type"] = df[desc_col].apply(assign_feature_type)

    summary = df.groupby(["primary_emotion", "semantic_type"]).size().reset_index(name="count")
    total = summary.groupby("primary_emotion")["count"].transform("sum")
    summary["proportion"] = summary["count"] / total

    by_layer = df.groupby(["layer_idx", "primary_emotion", "semantic_type"]).size().reset_index(name="count")
    by_layer_total = by_layer.groupby(["layer_idx", "primary_emotion"])["count"].transform("sum")
    by_layer["proportion"] = by_layer["count"] / by_layer_total

    _save_csv_if_requested(df, output_dir, "features_with_emotion_vs_other_semantics.csv")
    _save_csv_if_requested(summary, output_dir, "emotion_vs_other_semantics.csv")
    _save_csv_if_requested(by_layer, output_dir, "emotion_vs_other_semantics_by_layer.csv")

    return df, summary, by_layer


def plot_semantic_category_by_emotion(
    summary: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
    show: bool = True,
):
    output_dir = _ensure_output_dir(output_dir)
    mat = summary.pivot(index="primary_emotion", columns="semantic_type", values="proportion").fillna(0)
    mat = mat.reindex(emotions).fillna(0)

    for col in ["emotion_semantics", "other_semantics"]:
        if col not in mat.columns:
            mat[col] = 0.0
    mat = mat[["emotion_semantics", "other_semantics"]]

    fig = plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(mat))
    for semantic_type in mat.columns:
        vals = mat[semantic_type].to_numpy()
        plt.bar(mat.index, vals, bottom=bottom, label=semantic_type)
        bottom += vals

    plt.ylabel("Proportion")
    plt.title("Emotion Semantics vs Other Semantics")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "emotion_vs_other_semantics.png"), dpi=300)
    if show:
        plt.show()

    return fig


def plot_semantic_category_by_layer(
    by_layer: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
    show: bool = True,
):
    output_dir = _ensure_output_dir(output_dir)
    figs = []

    for emotion in emotions:
        sub = by_layer[by_layer["primary_emotion"] == emotion]
        if len(sub) == 0:
            continue

        mat = sub.pivot(index="layer_idx", columns="semantic_type", values="proportion").fillna(0).sort_index()
        for col in ["emotion_semantics", "other_semantics"]:
            if col not in mat.columns:
                mat[col] = 0.0

        fig = plt.figure(figsize=(8, 5))
        plt.plot(mat.index, mat["emotion_semantics"], marker="o", label="emotion_semantics")
        plt.plot(mat.index, mat["other_semantics"], marker="o", label="other_semantics")
        plt.xlabel("Layer")
        plt.ylabel("Proportion")
        plt.title(f"Semantic Type by Layer: {emotion}")
        plt.legend()
        plt.tight_layout()

        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"semantic_type_by_layer_{emotion}.png"), dpi=300)
        if show:
            plt.show()
        figs.append(fig)

    return figs


def build_paper_summary(
    conc_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    semantic_summary: pd.DataFrame,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    output_dir = _ensure_output_dir(output_dir)
    rows = []

    for emotion in emotions:
        sub = conc_df[conc_df["emotion"] == emotion]
        if len(sub) == 0:
            continue

        late_layer_cutoff = sub["layer_idx"].quantile(0.75)
        late = sub[sub["layer_idx"] >= late_layer_cutoff]

        sem_sub = semantic_summary[semantic_summary["primary_emotion"] == emotion]
        sem_dict = dict(zip(sem_sub["semantic_type"], sem_sub["proportion"]))

        overlap_sub = (
            overlap_df[(overlap_df["emotion_1"] == emotion) & (overlap_df["emotion_2"] != emotion)]
            .sort_values("jaccard", ascending=False)
            .head(3)
        )

        rows.append({
            "emotion": emotion,
            "mean_features_for_80pct_all_layers": sub["features_for_80pct"].mean(),
            "mean_features_for_80pct_late_layers": late["features_for_80pct"].mean(),
            "mean_gini_all_layers": sub["gini"].mean(),
            "mean_gini_late_layers": late["gini"].mean(),
            "emotion_semantics_proportion": sem_dict.get("emotion_semantics", 0.0),
            "other_semantics_proportion": sem_dict.get("other_semantics", 0.0),
            "top_jaccard_overlaps": "; ".join(
                f"{r['emotion_2']}={r['jaccard']:.2f}" for _, r in overlap_sub.iterrows()
            ),
            "top_overlap_coefficients": "; ".join(
                f"{r['emotion_2']}={r['overlap_coefficient']:.2f}" for _, r in overlap_sub.iterrows()
            ),
        })

    out = pd.DataFrame(rows)
    _save_csv_if_requested(out, output_dir, "paper_summary_table.csv")
    return out


def run_publishable_feature_analysis(
    input_path: str,
    output_dir: str,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    show: bool = True,
    run_semantic_analysis: bool = True,
    build_summary: bool = True,
):
    """
    Run publishable feature analyses.

    Parameters
    ----------
    input_path:
        Path to the selected/core feature CSV.
    output_dir:
        Directory where CSV and figure outputs are saved.
    emotions:
        Emotion order to use throughout the analysis.
    show:
        Whether to display matplotlib figures.
    run_semantic_analysis:
        If True, run description-keyword semantic categorization and semantic plots.
        If False, skip all description-dependent semantic analysis.
    build_summary:
        If True, build paper_summary_table.csv. When run_semantic_analysis=False,
        semantic proportion columns are filled with 0.0 placeholders.
    """
    output_dir = _ensure_output_dir(output_dir)
    df = pd.read_csv(input_path)

    logit_cols = find_logit_cols(df, emotions=emotions)

    conc_df = run_emotion_concentration_analysis(
        df,
        logit_cols=logit_cols,
        emotions=emotions,
        output_dir=output_dir,
    )
    plot_concentration(conc_df, emotions=emotions, output_dir=output_dir, show=show)

    overlap_df, jaccard_mat, overlap_coef_mat, count_mat = run_emotion_overlap_matrix_from_affected(
        df,
        emotions=emotions,
        output_dir=output_dir,
    )
    plot_overlap_matrix(jaccard_mat, "Emotion Overlap Jaccard Matrix", emotions=emotions, output_dir=output_dir, show=show)
    plot_overlap_matrix(overlap_coef_mat, "Emotion Overlap Coefficient Matrix", emotions=emotions, output_dir=output_dir, show=show)

    sem_df = None
    sem_summary = None
    sem_by_layer = None

    if run_semantic_analysis:
        desc_col = find_description_col(df)

        sem_df, sem_summary, sem_by_layer = run_semantic_category_analysis(
            df,
            logit_cols=logit_cols,
            desc_col=desc_col,
            emotions=emotions,
            output_dir=output_dir,
        )
        plot_semantic_category_by_emotion(
            sem_summary,
            emotions=emotions,
            output_dir=output_dir,
            show=show,
        )
        plot_semantic_category_by_layer(
            sem_by_layer,
            emotions=emotions,
            output_dir=output_dir,
            show=show,
        )
    else:
        sem_summary = pd.DataFrame(
            columns=["primary_emotion", "semantic_type", "count", "proportion"]
        )

    paper_summary = None
    if build_summary:
        paper_summary = build_paper_summary(
            conc_df,
            overlap_df,
            sem_summary,
            emotions=emotions,
            output_dir=output_dir,
        )

    return {
        "features": df,
        "concentration": conc_df,
        "overlap_long": overlap_df,
        "jaccard_matrix": jaccard_mat,
        "overlap_coefficient_matrix": overlap_coef_mat,
        "overlap_count_matrix": count_mat,
        "semantic_features": sem_df,
        "semantic_summary": sem_summary,
        "semantic_by_layer": sem_by_layer,
        "paper_summary": paper_summary,
    }


# ============================================================
# Ablation and activation patching / swap visualization utilities
# ============================================================

VISUAL_UTILS_VERSION = "nan-safe-plus-publishable-patching-plots-2026-05-17-v5"


def _make_plot_output_dir(output_dir: Optional[str]) -> Optional[str]:
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_current_matplotlib_fig(
    output_dir: Optional[str],
    filename: str,
    show: bool = True,
    dpi: int = 300,
):
    fig = plt.gcf()
    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print("Saved:", path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _pick_first_existing_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
    name: str = "column",
) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            f"Cannot find {name}. Tried: {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def plot_ablation_summary(
    summary_path: Optional[str] = None,
    summary_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    layer_col: str = "layer_idx",
    emotion_col: str = "target_emotion",
    flip_col: Optional[str] = None,
    logit_col: Optional[str] = None,
    nfeat_col: Optional[str] = None,
    show: bool = True,
    save: bool = True,
    dpi: int = 300,
):
    """
    Visualize per-layer per-emotion zero-ablation summary results.

    This function covers:
      1. flip rate by layer
      2. target logit drop/change by layer
      3. feature count by layer, if available
      4. flip-rate heatmap

    Parameters
    ----------
    summary_path:
        Path to per_layer_per_emotion_ablation_summary.csv.
    summary_df:
        Optional preloaded summary dataframe. If provided, summary_path is ignored.
    output_dir:
        Optional directory for PNG outputs.
    emotions:
        Emotion order.
    layer_col:
        Layer column name.
    emotion_col:
        Emotion column name, usually target_emotion.
    flip_col:
        Optional explicit flip metric column.
    logit_col:
        Optional explicit logit effect column.
    nfeat_col:
        Optional explicit feature-count column.
    show:
        Whether to display figures.
    save:
        Whether to save PNGs when output_dir is provided.
    dpi:
        Save resolution.

    Returns
    -------
    dict with dataframe, chosen columns, pivot table, and matplotlib figures.
    """
    if summary_df is None:
        if summary_path is None:
            raise ValueError("Either summary_path or summary_df must be provided.")
        df = pd.read_csv(summary_path)
    else:
        df = summary_df.copy()

    if layer_col not in df.columns:
        raise ValueError(f"Missing layer column: {layer_col}")
    if emotion_col not in df.columns:
        raise ValueError(f"Missing emotion column: {emotion_col}")

    if flip_col is None:
        flip_col = _pick_first_existing_col(
            df,
            [
                "flip_away_from_source_rate_clean_source_only",
                "flip_away_from_true_rate",
                "flip_rate",
                "prediction_flip_rate",
            ],
            name="flip-rate column",
        )

    if logit_col is None:
        logit_col = _pick_first_existing_col(
            df,
            [
                "avg_target_logit_drop",
                "avg_target_logit_change",
                "mean_signed_effect",
            ],
            name="logit-effect column",
        )

    if nfeat_col is None:
        nfeat_col = _pick_first_existing_col(
            df,
            [
                "n_ablated_features",
                "n_features",
                "num_features",
            ],
            required=False,
            name="feature-count column",
        )

    print("Using:")
    print("FLIP_COL:", flip_col)
    print("LOGIT_COL:", logit_col)
    print("NFEAT_COL:", nfeat_col)

    df = df.copy()
    df[layer_col] = pd.to_numeric(df[layer_col], errors="coerce")
    df = df.dropna(subset=[layer_col]).copy()
    df[layer_col] = df[layer_col].astype(int)

    for col in [flip_col, logit_col, nfeat_col]:
        if col is not None and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values([emotion_col, layer_col]).reset_index(drop=True)

    save_dir = _make_plot_output_dir(output_dir) if save else None
    figs = {}

    plt.figure(figsize=(10, 6))
    for emotion in emotions:
        sub = df[df[emotion_col] == emotion]
        if len(sub) == 0:
            continue
        plt.plot(
            sub[layer_col],
            sub[flip_col],
            marker="o",
            linewidth=2,
            label=emotion,
        )
    plt.xlabel("Layer")
    plt.ylabel("Prediction Flip Rate")
    plt.title("Layer-wise Emotion Ensemble Ablation Flip Rate")
    plt.grid(alpha=0.3)
    plt.legend()
    figs["flip_rate_by_layer"] = _save_current_matplotlib_fig(
        save_dir,
        "ablation_flip_rate_by_layer.png",
        show=show,
        dpi=dpi,
    )

    plt.figure(figsize=(10, 6))
    for emotion in emotions:
        sub = df[df[emotion_col] == emotion]
        if len(sub) == 0:
            continue
        plt.plot(
            sub[layer_col],
            sub[logit_col],
            marker="o",
            linewidth=2,
            label=emotion,
        )
    plt.xlabel("Layer")
    plt.ylabel("Average Target Logit Change")
    plt.title("Layer-wise Emotion Ensemble Ablation Logit Effect")
    plt.grid(alpha=0.3)
    plt.legend()
    figs["logit_effect_by_layer"] = _save_current_matplotlib_fig(
        save_dir,
        "ablation_logit_effect_by_layer.png",
        show=show,
        dpi=dpi,
    )

    if nfeat_col is not None:
        plt.figure(figsize=(10, 6))
        for emotion in emotions:
            sub = df[df[emotion_col] == emotion]
            if len(sub) == 0:
                continue
            plt.plot(
                sub[layer_col],
                sub[nfeat_col],
                marker="o",
                linewidth=2,
                label=emotion,
            )
        plt.xlabel("Layer")
        plt.ylabel("Number of Features")
        plt.title("Emotion Feature Count Across Layers")
        plt.grid(alpha=0.3)
        plt.legend()
        figs["feature_count_by_layer"] = _save_current_matplotlib_fig(
            save_dir,
            "ablation_feature_count_by_layer.png",
            show=show,
            dpi=dpi,
        )

    pivot = df.pivot_table(index=emotion_col, columns=layer_col, values=flip_col)
    pivot = pivot.reindex(index=emotions)

    plt.figure(figsize=(12, 4))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Flip Rate")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("Layer")
    plt.ylabel("Emotion")
    plt.title("Emotion Ensemble Ablation Flip Rate Heatmap")
    figs["flip_rate_heatmap"] = _save_current_matplotlib_fig(
        save_dir,
        "ablation_flip_rate_heatmap.png",
        show=show,
        dpi=dpi,
    )

    return {
        "df": df,
        "figures": figs,
        "pivot_flip_rate": pivot,
        "flip_col": flip_col,
        "logit_col": logit_col,
        "nfeat_col": nfeat_col,
    }


def _plot_pair_layer_curves_from_swap_df(
    df: pd.DataFrame,
    source_emotion: str,
    target_emotion: str,
    flip_away_col: str,
    flip_target_col: str,
    output_dir: Optional[str],
    show: bool,
    dpi: int,
):
    sub = df[
        (df["source_emotion"] == source_emotion)
        & (df["target_emotion"] == target_emotion)
    ].copy()

    if len(sub) == 0:
        print(f"No data for {source_emotion}->{target_emotion}")
        return None

    sub = sub.sort_values("layer_idx")

    plt.figure(figsize=(9, 5))
    plt.plot(
        sub["layer_idx"],
        sub[flip_away_col],
        marker="o",
        linewidth=2,
        label="flip away from source",
    )
    plt.plot(
        sub["layer_idx"],
        sub[flip_target_col],
        marker="o",
        linewidth=2,
        label="flip to target",
    )
    plt.xlabel("Layer")
    plt.ylabel("Rate")
    plt.title(f"Layer-wise Swap Effect: {source_emotion} → {target_emotion}")
    plt.grid(alpha=0.3)
    plt.legend()

    return _save_current_matplotlib_fig(
        output_dir,
        f"layer_curve_{source_emotion}_to_{target_emotion}.png",
        show=show,
        dpi=dpi,
    )


def plot_swap_summary(
    summary_path: Optional[str] = None,
    summary_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    emotions: Sequence[str] = DEFAULT_EMOTIONS,
    flip_away_col: str = "flip_away_from_source_rate_clean_source_only",
    flip_target_col: str = "flip_to_target_rate_clean_source_only",
    target_logit_col: str = "avg_target_logit_change_clean_source_only",
    source_logit_col: str = "avg_source_logit_change_clean_source_only",
    pair_curves: Optional[Sequence[Tuple[str, str]]] = None,
    top_k: int = 20,
    show: bool = True,
    save: bool = True,
    dpi: int = 300,
):
    """
    Visualize cross-emotion layer-local activation patching / swap summary results.

    This function covers:
      1. best-layer flip-to-target heatmap
      2. best-layer flip-away-from-source heatmap
      3. target specificity heatmap
      4. target emotion controllability bar chart
      5. source emotion disruptability bar chart
      6. layer-wise average flip-to-target by target emotion
      7. layer-wise average flip-away by source emotion
      8. top-k source-target-layer conversions
      9. optional pair-specific layer curves

    Returns a dict with the processed dataframe, best-layer tables, pivots, and figures.
    """
    if summary_df is None:
        if summary_path is None:
            raise ValueError("Either summary_path or summary_df must be provided.")
        df = pd.read_csv(summary_path)
    else:
        df = summary_df.copy()

    required = [
        "source_emotion",
        "target_emotion",
        "layer_idx",
        flip_away_col,
        flip_target_col,
        target_logit_col,
        source_logit_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce")
    df = df.dropna(subset=["layer_idx"]).copy()
    df["layer_idx"] = df["layer_idx"].astype(int)

    for col in [flip_away_col, flip_target_col, target_logit_col, source_logit_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    save_dir = _make_plot_output_dir(output_dir) if save else None
    figs = {}

    best_target = (
        df.sort_values(flip_target_col, ascending=False)
        .groupby(["source_emotion", "target_emotion"], as_index=False)
        .first()
    )

    pivot_target = best_target.pivot(
        index="source_emotion",
        columns="target_emotion",
        values=flip_target_col,
    ).reindex(index=emotions, columns=emotions)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_target.values, aspect="auto")
    plt.colorbar(im, label="Max Flip-to-Target Rate")
    plt.xticks(np.arange(len(emotions)), emotions, rotation=45, ha="right")
    plt.yticks(np.arange(len(emotions)), emotions)
    plt.xlabel("Target Emotion")
    plt.ylabel("Source Emotion")
    plt.title("Max Layer-wise Swap Conversion Rate")
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            val = pivot_target.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    figs["heatmap_max_flip_to_target_rate"] = _save_current_matplotlib_fig(
        save_dir,
        "heatmap_max_flip_to_target_rate.png",
        show=show,
        dpi=dpi,
    )

    best_away = (
        df.sort_values(flip_away_col, ascending=False)
        .groupby(["source_emotion", "target_emotion"], as_index=False)
        .first()
    )

    pivot_away = best_away.pivot(
        index="source_emotion",
        columns="target_emotion",
        values=flip_away_col,
    ).reindex(index=emotions, columns=emotions)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_away.values, aspect="auto")
    plt.colorbar(im, label="Max Flip-away-from-source Rate")
    plt.xticks(np.arange(len(emotions)), emotions, rotation=45, ha="right")
    plt.yticks(np.arange(len(emotions)), emotions)
    plt.xlabel("Target Emotion")
    plt.ylabel("Source Emotion")
    plt.title("Max Layer-wise Swap Disruption Rate")
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            val = pivot_away.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    figs["heatmap_max_flip_away_from_source_rate"] = _save_current_matplotlib_fig(
        save_dir,
        "heatmap_max_flip_away_from_source_rate.png",
        show=show,
        dpi=dpi,
    )

    specificity = best_away.merge(
        best_target[["source_emotion", "target_emotion", flip_target_col]],
        on=["source_emotion", "target_emotion"],
        how="left",
        suffixes=("", "_target_best"),
    )
    specificity["target_specificity"] = specificity[flip_target_col] / (
        specificity[flip_away_col] + 1e-8
    )

    pivot_spec = specificity.pivot(
        index="source_emotion",
        columns="target_emotion",
        values="target_specificity",
    ).reindex(index=emotions, columns=emotions)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_spec.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Target Specificity")
    plt.xticks(np.arange(len(emotions)), emotions, rotation=45, ha="right")
    plt.yticks(np.arange(len(emotions)), emotions)
    plt.xlabel("Target Emotion")
    plt.ylabel("Source Emotion")
    plt.title("Swap Target Specificity")
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            val = pivot_spec.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    figs["heatmap_target_specificity"] = _save_current_matplotlib_fig(
        save_dir,
        "heatmap_target_specificity.png",
        show=show,
        dpi=dpi,
    )

    target_summary = (
        best_target.groupby("target_emotion", as_index=False)
        .agg(
            mean_max_flip_to_target=(flip_target_col, "mean"),
            mean_max_flip_away=(flip_away_col, "mean"),
            mean_target_logit_change=(target_logit_col, "mean"),
        )
        .set_index("target_emotion")
        .reindex(emotions)
        .reset_index()
    )

    x = np.arange(len(target_summary))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, target_summary["mean_max_flip_to_target"], width, label="flip to target")
    plt.bar(x + width / 2, target_summary["mean_max_flip_away"], width, label="flip away from source")
    plt.xticks(x, target_summary["target_emotion"], rotation=45, ha="right")
    plt.ylabel("Average Best-layer Rate")
    plt.xlabel("Target Emotion")
    plt.title("How Controllable Is Each Target Emotion?")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    figs["bar_target_emotion_controllability"] = _save_current_matplotlib_fig(
        save_dir,
        "bar_target_emotion_controllability.png",
        show=show,
        dpi=dpi,
    )

    source_summary = (
        best_target.groupby("source_emotion", as_index=False)
        .agg(
            mean_max_flip_to_target=(flip_target_col, "mean"),
            mean_max_flip_away=(flip_away_col, "mean"),
            mean_source_logit_change=(source_logit_col, "mean"),
        )
        .set_index("source_emotion")
        .reindex(emotions)
        .reset_index()
    )

    x = np.arange(len(source_summary))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, source_summary["mean_max_flip_to_target"], width, label="flip to target")
    plt.bar(x + width / 2, source_summary["mean_max_flip_away"], width, label="flip away from source")
    plt.xticks(x, source_summary["source_emotion"], rotation=45, ha="right")
    plt.ylabel("Average Best-layer Rate")
    plt.xlabel("Source Emotion")
    plt.title("How Easy Is Each Source Emotion to Disrupt?")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    figs["bar_source_emotion_disruptability"] = _save_current_matplotlib_fig(
        save_dir,
        "bar_source_emotion_disruptability.png",
        show=show,
        dpi=dpi,
    )

    plt.figure(figsize=(10, 6))
    for target_emotion in emotions:
        sub = df[df["target_emotion"] == target_emotion]
        if len(sub) == 0:
            continue
        layer_avg = (
            sub.groupby("layer_idx", as_index=False)[flip_target_col]
            .mean()
            .sort_values("layer_idx")
        )
        plt.plot(
            layer_avg["layer_idx"],
            layer_avg[flip_target_col],
            marker="o",
            linewidth=2,
            label=target_emotion,
        )
    plt.xlabel("Layer")
    plt.ylabel("Average Flip-to-Target Rate")
    plt.title("Layer-wise Swap Conversion by Target Emotion")
    plt.grid(alpha=0.3)
    plt.legend()
    figs["line_layerwise_flip_to_target_by_target"] = _save_current_matplotlib_fig(
        save_dir,
        "line_layerwise_flip_to_target_by_target.png",
        show=show,
        dpi=dpi,
    )

    plt.figure(figsize=(10, 6))
    for source_emotion in emotions:
        sub = df[df["source_emotion"] == source_emotion]
        if len(sub) == 0:
            continue
        layer_avg = (
            sub.groupby("layer_idx", as_index=False)[flip_away_col]
            .mean()
            .sort_values("layer_idx")
        )
        plt.plot(
            layer_avg["layer_idx"],
            layer_avg[flip_away_col],
            marker="o",
            linewidth=2,
            label=source_emotion,
        )
    plt.xlabel("Layer")
    plt.ylabel("Average Flip-away-from-source Rate")
    plt.title("Layer-wise Swap Disruption by Source Emotion")
    plt.grid(alpha=0.3)
    plt.legend()
    figs["line_layerwise_flip_away_by_source"] = _save_current_matplotlib_fig(
        save_dir,
        "line_layerwise_flip_away_by_source.png",
        show=show,
        dpi=dpi,
    )

    top = df.sort_values(flip_target_col, ascending=False).head(top_k).copy()
    top["pair_layer"] = (
        top["source_emotion"]
        + "→"
        + top["target_emotion"]
        + " L"
        + top["layer_idx"].astype(str)
    )
    top = top.sort_values(flip_target_col, ascending=True)

    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    plt.barh(top["pair_layer"], top[flip_target_col])
    plt.xlabel("Flip-to-Target Rate")
    plt.ylabel("Source→Target Layer")
    plt.title(f"Top {top_k} Layer-local Emotion Swap Conversions")
    plt.grid(axis="x", alpha=0.3)
    figs["bar_top_swap_conversions"] = _save_current_matplotlib_fig(
        save_dir,
        f"bar_top_{top_k}_swap_conversions.png",
        show=show,
        dpi=dpi,
    )

    if pair_curves is None:
        pair_curves = []

    for source_emotion, target_emotion in pair_curves:
        fig = _plot_pair_layer_curves_from_swap_df(
            df=df,
            source_emotion=source_emotion,
            target_emotion=target_emotion,
            flip_away_col=flip_away_col,
            flip_target_col=flip_target_col,
            output_dir=save_dir,
            show=show,
            dpi=dpi,
        )
        if fig is not None:
            figs[f"layer_curve_{source_emotion}_to_{target_emotion}"] = fig

    return {
        "df": df,
        "figures": figs,
        "best_target": best_target,
        "best_away": best_away,
        "specificity": specificity,
        "pivot_target": pivot_target,
        "pivot_away": pivot_away,
        "pivot_specificity": pivot_spec,
        "target_summary": target_summary,
        "source_summary": source_summary,
        "top_conversions": top,
        "flip_away_col": flip_away_col,
        "flip_target_col": flip_target_col,
        "target_logit_col": target_logit_col,
        "source_logit_col": source_logit_col,
    }



def run_all_model_publishable_feature_aggregate_plot(
    models,
    base_data_dir="data",
    output_dir="data/all_model_emotion_feature_analysis",
    skip_models=None,
    emotions=DEFAULT_EMOTIONS,
    show=True,
):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if skip_models is None:
        skip_models = set()

    os.makedirs(output_dir, exist_ok=True)

    def get_sae_sizes_for_model(model_name):
        if model_name.startswith("google/gemma-2"):
            return [None]

        if model_name.startswith("meta-llama/"):
            return [None]

        if model_name.startswith("google/gemma-3"):
            return ["small", "big"]

        return [None]

    def get_model_slug(model_name):
        return model_name.replace("/", "-")

    def get_model_label(model_name, sae_size):
        slug = get_model_slug(model_name)
        if sae_size is None:
            return slug
        return f"{slug}-{sae_size}"

    def get_feature_dir(model_name, sae_size):
        slug = get_model_slug(model_name)
        if sae_size is None:
            return f"{base_data_dir}/{slug}-feature-act"
        return f"{base_data_dir}/{slug}-feature-act-{sae_size}"

    paper_summary_dfs = []
    concentration_dfs = []
    overlap_dfs = []

    run_log = []

    for model_name in models:
        if model_name in skip_models:
            run_log.append({
                "model_name": model_name,
                "sae_size": None,
                "status": "skipped_model",
                "path": "",
            })
            continue

        for sae_size in get_sae_sizes_for_model(model_name):
            model_label = get_model_label(model_name, sae_size)
            feature_dir = get_feature_dir(model_name, sae_size)
            analysis_dir = os.path.join(feature_dir, "emotion_feature_analysis")

            paper_summary_path = os.path.join(analysis_dir, "paper_summary_table.csv")
            concentration_path = os.path.join(analysis_dir, "emotion_concentration_by_layer.csv")
            overlap_path = os.path.join(analysis_dir, "emotion_overlap_from_affected_long.csv")

            if os.path.exists(paper_summary_path):
                df = pd.read_csv(paper_summary_path)
                df["model_name"] = model_name
                df["sae_size"] = sae_size if sae_size is not None else "none"
                df["model_label"] = model_label
                paper_summary_dfs.append(df)
                print("Loaded:", paper_summary_path)
            else:
                print("Missing:", paper_summary_path)

            if os.path.exists(concentration_path):
                df = pd.read_csv(concentration_path)
                df["model_name"] = model_name
                df["sae_size"] = sae_size if sae_size is not None else "none"
                df["model_label"] = model_label
                concentration_dfs.append(df)
                print("Loaded:", concentration_path)
            else:
                print("Missing:", concentration_path)

            if os.path.exists(overlap_path):
                df = pd.read_csv(overlap_path)
                df["model_name"] = model_name
                df["sae_size"] = sae_size if sae_size is not None else "none"
                df["model_label"] = model_label
                overlap_dfs.append(df)
                print("Loaded:", overlap_path)
            else:
                print("Missing:", overlap_path)

            run_log.append({
                "model_name": model_name,
                "sae_size": sae_size if sae_size is not None else "none",
                "status": "checked",
                "path": analysis_dir,
            })

    paper_summary_all = (
        pd.concat(paper_summary_dfs, ignore_index=True)
        if len(paper_summary_dfs) > 0
        else pd.DataFrame()
    )

    concentration_all = (
        pd.concat(concentration_dfs, ignore_index=True)
        if len(concentration_dfs) > 0
        else pd.DataFrame()
    )

    overlap_all = (
        pd.concat(overlap_dfs, ignore_index=True)
        if len(overlap_dfs) > 0
        else pd.DataFrame()
    )

    run_log_df = pd.DataFrame(run_log)

    paper_summary_all.to_csv(
        os.path.join(output_dir, "all_model_paper_summary_table.csv"),
        index=False,
    )
    concentration_all.to_csv(
        os.path.join(output_dir, "all_model_emotion_concentration_by_layer.csv"),
        index=False,
    )
    overlap_all.to_csv(
        os.path.join(output_dir, "all_model_emotion_overlap_from_affected_long.csv"),
        index=False,
    )
    run_log_df.to_csv(
        os.path.join(output_dir, "all_model_aggregate_run_log.csv"),
        index=False,
    )

    figs = {}

    # ============================================================
    # Plot 1: heatmap of late-layer features_for_80pct
    # ============================================================

    if len(paper_summary_all) > 0:
        metric = "mean_features_for_80pct_late_layers"

        if metric in paper_summary_all.columns:
            pivot = paper_summary_all.pivot_table(
                index="model_label",
                columns="emotion",
                values=metric,
                aggfunc="mean",
            )
            pivot = pivot.reindex(columns=[e for e in emotions if e in pivot.columns])

            plt.figure(figsize=(13, max(5, 0.45 * len(pivot))))
            im = plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(im, label=metric)
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            plt.xlabel("Emotion")
            plt.ylabel("Model")
            plt.title("Late-layer Emotion Feature Concentration Across Models\nLower = more concentrated")

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

            plt.tight_layout()
            path = os.path.join(output_dir, "all_model_late_layer_features_for_80pct_heatmap.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

            figs["late_layer_features_for_80pct_heatmap"] = path
            pivot.to_csv(os.path.join(output_dir, "all_model_late_layer_features_for_80pct_pivot.csv"))

    # ============================================================
    # Plot 2: heatmap of late-layer gini
    # ============================================================

    if len(paper_summary_all) > 0:
        metric = "mean_gini_late_layers"

        if metric in paper_summary_all.columns:
            pivot = paper_summary_all.pivot_table(
                index="model_label",
                columns="emotion",
                values=metric,
                aggfunc="mean",
            )
            pivot = pivot.reindex(columns=[e for e in emotions if e in pivot.columns])

            plt.figure(figsize=(13, max(5, 0.45 * len(pivot))))
            im = plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(im, label=metric)
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            plt.xlabel("Emotion")
            plt.ylabel("Model")
            plt.title("Late-layer Emotion Feature Gini Across Models\nHigher = more concentrated")

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

            plt.tight_layout()
            path = os.path.join(output_dir, "all_model_late_layer_gini_heatmap.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

            figs["late_layer_gini_heatmap"] = path
            pivot.to_csv(os.path.join(output_dir, "all_model_late_layer_gini_pivot.csv"))

    # ============================================================
    # Plot 3: average late-layer concentration by emotion
    # ============================================================

    if len(paper_summary_all) > 0:
        metric = "mean_features_for_80pct_late_layers"

        if metric in paper_summary_all.columns:
            avg_df = (
                paper_summary_all
                .groupby("emotion", as_index=False)[metric]
                .mean()
            )

            avg_df["emotion"] = pd.Categorical(
                avg_df["emotion"],
                categories=emotions,
                ordered=True,
            )
            avg_df = avg_df.sort_values("emotion")

            plt.figure(figsize=(8, 5))
            plt.bar(avg_df["emotion"].astype(str), avg_df[metric])
            plt.xlabel("Emotion")
            plt.ylabel("Mean # Features for 80%")
            plt.title("Average Late-layer Emotion Feature Concentration Across Models\nLower = more concentrated")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            path = os.path.join(output_dir, "average_late_layer_features_for_80pct_by_emotion.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

            figs["average_late_layer_features_for_80pct_by_emotion"] = path
            avg_df.to_csv(
                os.path.join(output_dir, "average_late_layer_features_for_80pct_by_emotion.csv"),
                index=False,
            )

    # ============================================================
    # Plot 4: layer-wise concentration curves, averaged over models
    # ============================================================

    if len(concentration_all) > 0:
        metric = "features_for_80pct"

        if metric in concentration_all.columns:
            avg_layer_df = (
                concentration_all
                .groupby(["emotion", "layer_idx"], as_index=False)[metric]
                .mean()
            )

            plt.figure(figsize=(10, 6))
            for emotion in emotions:
                sub = avg_layer_df[avg_layer_df["emotion"] == emotion].sort_values("layer_idx")
                if len(sub) == 0:
                    continue

                plt.plot(
                    sub["layer_idx"],
                    sub[metric],
                    marker="o",
                    linewidth=2,
                    label=emotion,
                )

            plt.xlabel("Layer")
            plt.ylabel("Mean # Features for 80%")
            plt.title("Average Layer-wise Emotion Feature Concentration Across Models")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            path = os.path.join(output_dir, "average_layerwise_features_for_80pct_across_models.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

            figs["average_layerwise_features_for_80pct_across_models"] = path
            avg_layer_df.to_csv(
                os.path.join(output_dir, "average_layerwise_features_for_80pct_across_models.csv"),
                index=False,
            )

    # ============================================================
    # Plot 5: overlap heatmap averaged over models
    # ============================================================

    if len(overlap_all) > 0 and "jaccard" in overlap_all.columns:
        avg_overlap = (
            overlap_all
            .groupby(["emotion_1", "emotion_2"], as_index=False)["jaccard"]
            .mean()
        )

        mat = avg_overlap.pivot(
            index="emotion_1",
            columns="emotion_2",
            values="jaccard",
        )
        mat = mat.reindex(index=emotions, columns=emotions)

        plt.figure(figsize=(7, 6))
        im = plt.imshow(mat.values, aspect="auto")
        plt.colorbar(im, label="Mean Jaccard")
        plt.xticks(np.arange(len(emotions)), emotions, rotation=45, ha="right")
        plt.yticks(np.arange(len(emotions)), emotions)
        plt.xlabel("Emotion 2")
        plt.ylabel("Emotion 1")
        plt.title("Average Emotion Feature Overlap Across Models")

        for i in range(len(emotions)):
            for j in range(len(emotions)):
                val = mat.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()

        path = os.path.join(output_dir, "average_emotion_overlap_jaccard_heatmap.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        figs["average_emotion_overlap_jaccard_heatmap"] = path
        mat.to_csv(os.path.join(output_dir, "average_emotion_overlap_jaccard_matrix.csv"))

    print("\nSaved aggregate outputs to:", output_dir)
    for name, path in figs.items():
        print(name, ":", path)

    return {
        "paper_summary_all": paper_summary_all,
        "concentration_all": concentration_all,
        "overlap_all": overlap_all,
        "run_log": run_log_df,
        "figures": figs,
    }



def plot_all_model_ablation_summary(
    models,
    base_data_dir="data",
    output_dir="data/all_model_ablation_summary",
    skip_models=None,
    emotions=DEFAULT_EMOTIONS,
    show=True,
    save=True,
    dpi=300,
):
    """
    Aggregate per-layer per-emotion ablation summaries across models.

    Expected per-model input:
        data/{model_slug}-feature-act/per_layer_per_emotion_feature_ablation/per_layer_per_emotion_ablation_summary.csv

    For Gemma-3 small/big:
        data/{model_slug}-feature-act-small/per_layer_per_emotion_feature_ablation/per_layer_per_emotion_ablation_summary.csv
        data/{model_slug}-feature-act-big/per_layer_per_emotion_feature_ablation/per_layer_per_emotion_ablation_summary.csv

    Returns
    -------
    dict containing:
        df
        run_log
        figures
        pivot tables
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if skip_models is None:
        skip_models = set()

    os.makedirs(output_dir, exist_ok=True)

    def get_sae_sizes_for_model(model_name):
        if model_name.startswith("google/gemma-2"):
            return [None]

        if model_name.startswith("meta-llama/"):
            return [None]

        if model_name.startswith("google/gemma-3"):
            return ["small", "big"]

        return [None]

    def get_model_slug(model_name):
        return model_name.replace("/", "-")

    def get_model_label(model_name, sae_size):
        slug = get_model_slug(model_name)
        if sae_size is None:
            return slug
        return f"{slug}-{sae_size}"

    def get_feature_dir(model_name, sae_size):
        slug = get_model_slug(model_name)
        if sae_size is None:
            return f"{base_data_dir}/{slug}-feature-act"
        return f"{base_data_dir}/{slug}-feature-act-{sae_size}"

    def save_current_fig(filename):
        fig = plt.gcf()
        plt.tight_layout()
        path = os.path.join(output_dir, filename)

        if save:
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            print("Saved:", path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return path

    all_dfs = []
    run_log = []

    for model_name in models:
        if model_name in skip_models:
            run_log.append({
                "model_name": model_name,
                "sae_size": "none",
                "model_label": get_model_label(model_name, None),
                "status": "skipped_model",
                "summary_path": "",
            })
            continue

        for sae_size in get_sae_sizes_for_model(model_name):
            model_label = get_model_label(model_name, sae_size)
            feature_dir = get_feature_dir(model_name, sae_size)

            summary_path = os.path.join(
                feature_dir,
                "per_layer_per_emotion_feature_ablation",
                "per_layer_per_emotion_ablation_summary.csv",
            )

            if not os.path.exists(summary_path):
                print("Missing:", summary_path)
                run_log.append({
                    "model_name": model_name,
                    "sae_size": sae_size if sae_size is not None else "none",
                    "model_label": model_label,
                    "status": "missing_summary",
                    "summary_path": summary_path,
                })
                continue

            df = pd.read_csv(summary_path)
            df["model_name"] = model_name
            df["sae_size"] = sae_size if sae_size is not None else "none"
            df["model_label"] = model_label

            all_dfs.append(df)

            print("Loaded:", summary_path)
            run_log.append({
                "model_name": model_name,
                "sae_size": sae_size if sae_size is not None else "none",
                "model_label": model_label,
                "status": "loaded",
                "summary_path": summary_path,
            })

    if len(all_dfs) == 0:
        print("No ablation summaries found.")
        return {
            "df": pd.DataFrame(),
            "run_log": pd.DataFrame(run_log),
            "figures": {},
        }

    df = pd.concat(all_dfs, ignore_index=True)
    run_log_df = pd.DataFrame(run_log)

    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce")
    df = df.dropna(subset=["layer_idx"]).copy()
    df["layer_idx"] = df["layer_idx"].astype(int)

    if "target_emotion" not in df.columns:
        raise ValueError(f"Missing target_emotion column. Available columns: {list(df.columns)}")

    numeric_candidates = [
        "prediction_flip_rate",
        "flip_away_from_true_rate",
        "avg_target_logit_drop",
        "std_target_logit_drop",
        "n_ablated_features",
        "n_instances",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    aggregate_csv_path = os.path.join(output_dir, "all_model_ablation_summary.csv")
    run_log_path = os.path.join(output_dir, "all_model_ablation_run_log.csv")

    df.to_csv(aggregate_csv_path, index=False)
    run_log_df.to_csv(run_log_path, index=False)

    print("Saved:", aggregate_csv_path)
    print("Saved:", run_log_path)

    figures = {}

    # ============================================================
    # Plot 1: max flip-away rate heatmap
    # ============================================================

    if "flip_away_from_true_rate" in df.columns:
        metric = "flip_away_from_true_rate"

        best_df = (
            df.sort_values(metric, ascending=False)
            .groupby(["model_label", "target_emotion"], as_index=False)
            .first()
        )

        pivot = best_df.pivot_table(
            index="model_label",
            columns="target_emotion",
            values=metric,
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=[e for e in emotions if e in pivot.columns])

        plt.figure(figsize=(13, max(5, 0.45 * len(pivot))))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, label="Max Flip-away Rate")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel("Emotion")
        plt.ylabel("Model")
        plt.title("Max Layer-wise Ablation Flip-away Rate Across Models")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        figures["heatmap_max_flip_away_rate"] = save_current_fig(
            "heatmap_max_ablation_flip_away_rate.png"
        )

        pivot.to_csv(os.path.join(output_dir, "pivot_max_ablation_flip_away_rate.csv"))

    # ============================================================
    # Plot 2: max prediction flip rate heatmap
    # ============================================================

    if "prediction_flip_rate" in df.columns:
        metric = "prediction_flip_rate"

        best_df = (
            df.sort_values(metric, ascending=False)
            .groupby(["model_label", "target_emotion"], as_index=False)
            .first()
        )

        pivot = best_df.pivot_table(
            index="model_label",
            columns="target_emotion",
            values=metric,
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=[e for e in emotions if e in pivot.columns])

        plt.figure(figsize=(13, max(5, 0.45 * len(pivot))))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, label="Max Prediction Flip Rate")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel("Emotion")
        plt.ylabel("Model")
        plt.title("Max Layer-wise Ablation Prediction Flip Rate Across Models")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        figures["heatmap_max_prediction_flip_rate"] = save_current_fig(
            "heatmap_max_ablation_prediction_flip_rate.png"
        )

        pivot.to_csv(os.path.join(output_dir, "pivot_max_ablation_prediction_flip_rate.csv"))

    # ============================================================
    # Plot 3: max target logit drop heatmap
    # ============================================================

    if "avg_target_logit_drop" in df.columns:
        metric = "avg_target_logit_drop"

        best_df = (
            df.sort_values(metric, ascending=False)
            .groupby(["model_label", "target_emotion"], as_index=False)
            .first()
        )

        pivot = best_df.pivot_table(
            index="model_label",
            columns="target_emotion",
            values=metric,
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=[e for e in emotions if e in pivot.columns])

        plt.figure(figsize=(13, max(5, 0.45 * len(pivot))))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, label="Max Avg Target Logit Drop")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel("Emotion")
        plt.ylabel("Model")
        plt.title("Max Layer-wise Target Logit Drop Across Models")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        figures["heatmap_max_target_logit_drop"] = save_current_fig(
            "heatmap_max_ablation_target_logit_drop.png"
        )

        pivot.to_csv(os.path.join(output_dir, "pivot_max_ablation_target_logit_drop.csv"))

    # ============================================================
    # Plot 4: average layer-wise flip-away curve across models
    # ============================================================

    if "flip_away_from_true_rate" in df.columns:
        metric = "flip_away_from_true_rate"

        avg_layer_df = (
            df.groupby(["target_emotion", "layer_idx"], as_index=False)[metric]
            .mean()
            .sort_values(["target_emotion", "layer_idx"])
        )

        plt.figure(figsize=(10, 6))
        for emotion in emotions:
            sub = avg_layer_df[avg_layer_df["target_emotion"] == emotion]
            if len(sub) == 0:
                continue

            plt.plot(
                sub["layer_idx"],
                sub[metric],
                marker="o",
                linewidth=2,
                label=emotion,
            )

        plt.xlabel("Layer")
        plt.ylabel("Mean Flip-away Rate")
        plt.title("Average Layer-wise Ablation Flip-away Rate Across Models")
        plt.grid(alpha=0.3)
        plt.legend()

        figures["line_average_layerwise_flip_away_rate"] = save_current_fig(
            "line_average_layerwise_ablation_flip_away_rate.png"
        )

        avg_layer_df.to_csv(
            os.path.join(output_dir, "average_layerwise_ablation_flip_away_rate.csv"),
            index=False,
        )

    # ============================================================
    # Plot 5: average layer-wise target logit drop across models
    # ============================================================

    if "avg_target_logit_drop" in df.columns:
        metric = "avg_target_logit_drop"

        avg_layer_df = (
            df.groupby(["target_emotion", "layer_idx"], as_index=False)[metric]
            .mean()
            .sort_values(["target_emotion", "layer_idx"])
        )

        plt.figure(figsize=(10, 6))
        for emotion in emotions:
            sub = avg_layer_df[avg_layer_df["target_emotion"] == emotion]
            if len(sub) == 0:
                continue

            plt.plot(
                sub["layer_idx"],
                sub[metric],
                marker="o",
                linewidth=2,
                label=emotion,
            )

        plt.xlabel("Layer")
        plt.ylabel("Mean Target Logit Drop")
        plt.title("Average Layer-wise Ablation Target Logit Drop Across Models")
        plt.grid(alpha=0.3)
        plt.legend()

        figures["line_average_layerwise_target_logit_drop"] = save_current_fig(
            "line_average_layerwise_ablation_target_logit_drop.png"
        )

        avg_layer_df.to_csv(
            os.path.join(output_dir, "average_layerwise_ablation_target_logit_drop.csv"),
            index=False,
        )

    # ============================================================
    # Plot 6: average best-layer flip-away by emotion
    # ============================================================

    if "flip_away_from_true_rate" in df.columns:
        metric = "flip_away_from_true_rate"

        best_df = (
            df.sort_values(metric, ascending=False)
            .groupby(["model_label", "target_emotion"], as_index=False)
            .first()
        )

        avg_df = (
            best_df.groupby("target_emotion", as_index=False)[metric]
            .mean()
        )

        avg_df["target_emotion"] = pd.Categorical(
            avg_df["target_emotion"],
            categories=emotions,
            ordered=True,
        )
        avg_df = avg_df.sort_values("target_emotion")

        plt.figure(figsize=(8, 5))
        plt.bar(avg_df["target_emotion"].astype(str), avg_df[metric])
        plt.xlabel("Emotion")
        plt.ylabel("Mean Best-layer Flip-away Rate")
        plt.title("Average Best-layer Ablation Necessity Across Models")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)

        figures["bar_average_best_layer_flip_away_by_emotion"] = save_current_fig(
            "bar_average_best_layer_ablation_flip_away_by_emotion.png"
        )

        avg_df.to_csv(
            os.path.join(output_dir, "average_best_layer_ablation_flip_away_by_emotion.csv"),
            index=False,
        )

    # ============================================================
    # Plot 7: average best-layer logit drop by emotion
    # ============================================================

    if "avg_target_logit_drop" in df.columns:
        metric = "avg_target_logit_drop"

        best_df = (
            df.sort_values(metric, ascending=False)
            .groupby(["model_label", "target_emotion"], as_index=False)
            .first()
        )

        avg_df = (
            best_df.groupby("target_emotion", as_index=False)[metric]
            .mean()
        )

        avg_df["target_emotion"] = pd.Categorical(
            avg_df["target_emotion"],
            categories=emotions,
            ordered=True,
        )
        avg_df = avg_df.sort_values("target_emotion")

        plt.figure(figsize=(8, 5))
        plt.bar(avg_df["target_emotion"].astype(str), avg_df[metric])
        plt.xlabel("Emotion")
        plt.ylabel("Mean Best-layer Target Logit Drop")
        plt.title("Average Best-layer Ablation Logit Effect Across Models")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)

        figures["bar_average_best_layer_logit_drop_by_emotion"] = save_current_fig(
            "bar_average_best_layer_ablation_logit_drop_by_emotion.png"
        )

        avg_df.to_csv(
            os.path.join(output_dir, "average_best_layer_ablation_logit_drop_by_emotion.csv"),
            index=False,
        )

    # ============================================================
    # Plot 8: top model-emotion-layer ablation effects
    # ============================================================

    if "flip_away_from_true_rate" in df.columns:
        metric = "flip_away_from_true_rate"

        top_df = df.sort_values(metric, ascending=False).head(30).copy()
        top_df["label"] = (
            top_df["model_label"]
            + " | "
            + top_df["target_emotion"].astype(str)
            + " L"
            + top_df["layer_idx"].astype(str)
        )

        top_df = top_df.sort_values(metric, ascending=True)

        plt.figure(figsize=(10, max(5, 0.28 * len(top_df))))
        plt.barh(top_df["label"], top_df[metric])
        plt.xlabel("Flip-away Rate")
        plt.ylabel("Model | Emotion Layer")
        plt.title("Top Layer-wise Ablation Effects Across Models")
        plt.grid(axis="x", alpha=0.3)

        figures["bar_top_ablation_flip_away_effects"] = save_current_fig(
            "bar_top_ablation_flip_away_effects.png"
        )

        top_df.to_csv(
            os.path.join(output_dir, "top_ablation_flip_away_effects.csv"),
            index=False,
        )

    print("\nSaved aggregate ablation visuals to:", output_dir)
    for name, path in figures.items():
        print(name, ":", path)

    return {
        "df": df,
        "run_log": run_log_df,
        "figures": figures,
    }

# ============================================================
# All-model activation patching / swap aggregate visualization
# ============================================================

VISUAL_UTILS_VERSION = "nan-safe-plus-publishable-patching-plots-aggregate-swap-2026-05-18-v6"


def plot_all_model_swap_summary(
    models,
    base_data_dir="data",
    output_dir="data/all_model_cross_emotion_layer_feature_swap",
    skip_models=None,
    emotions=DEFAULT_EMOTIONS,
    pair_curves=None,
    make_per_model_plots=True,
    show=True,
    save=True,
    dpi=300,
    top_k=40,
    flip_away_col="flip_away_from_source_rate_clean_source_only",
    flip_target_col="flip_to_target_rate_clean_source_only",
    target_logit_col="avg_target_logit_change_clean_source_only",
    source_logit_col="avg_source_logit_change_clean_source_only",
):
    """
    Plot cross-emotion layer-local activation swap results for all models and aggregate them.

    Expected per-run input:
        data/{model_slug}-feature-act/cross_emotion_layer_feature_swap/cross_emotion_layer_feature_swap_summary.csv

    For Gemma-3 small/big:
        data/{model_slug}-feature-act-small/cross_emotion_layer_feature_swap/cross_emotion_layer_feature_swap_summary.csv
        data/{model_slug}-feature-act-big/cross_emotion_layer_feature_swap/cross_emotion_layer_feature_swap_summary.csv

    This function does two things:
      1. Optionally calls plot_swap_summary for every available model/SAE-size run.
      2. Builds aggregate CSVs and aggregate figures across all available runs.

    Returns
    -------
    dict containing:
        df: all loaded summary rows
        run_log: loaded/missing status for each model/SAE size
        best_pair_layer: best layer per model/source/target pair by flip-to-target
        figures: output figure paths
        aggregate tables used for plotting
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if skip_models is None:
        skip_models = set()
    else:
        skip_models = set(skip_models)

    if pair_curves is None:
        pair_curves = []

    os.makedirs(output_dir, exist_ok=True)

    def get_sae_sizes_for_model(model_name):
        if model_name.startswith("google/gemma-2"):
            return [None]
        if model_name.startswith("meta-llama/"):
            return [None]
        if model_name.startswith("google/gemma-3"):
            return ["small", "big"]
        return [None]

    def get_model_slug(model_name):
        return model_name.replace("/", "-")

    def get_model_label(model_name, sae_size):
        slug = get_model_slug(model_name)
        if sae_size is None:
            return slug
        return f"{slug}-{sae_size}"

    def get_feature_dir(model_name, sae_size):
        slug = get_model_slug(model_name)
        if sae_size is None:
            return f"{base_data_dir}/{slug}-feature-act"
        return f"{base_data_dir}/{slug}-feature-act-{sae_size}"

    def save_current_fig(filename):
        fig = plt.gcf()
        plt.tight_layout()
        path = os.path.join(output_dir, filename)

        if save:
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            print("Saved:", path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return path

    def sanitize_numeric(df, cols):
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def draw_heatmap(pivot, title, cbar_label, filename, fmt=".2f", figsize=None):
        if pivot is None or len(pivot) == 0:
            return None

        if figsize is None:
            figsize = (max(8, 0.42 * len(pivot.columns)), max(5, 0.45 * len(pivot.index)))

        plt.figure(figsize=figsize)
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, label=cbar_label)
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.title(title)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

        return save_current_fig(filename)

    all_dfs = []
    run_log = []

    for model_name in models:
        if model_name in skip_models:
            run_log.append({
                "model_name": model_name,
                "sae_size": "none",
                "model_label": get_model_label(model_name, None),
                "status": "skipped_model",
                "summary_path": "",
            })
            continue

        for sae_size in get_sae_sizes_for_model(model_name):
            model_label = get_model_label(model_name, sae_size)
            feature_dir = get_feature_dir(model_name, sae_size)
            swap_dir = os.path.join(feature_dir, "cross_emotion_layer_feature_swap")
            summary_path = os.path.join(swap_dir, "cross_emotion_layer_feature_swap_summary.csv")

            if not os.path.exists(summary_path):
                print("Missing:", summary_path)
                run_log.append({
                    "model_name": model_name,
                    "sae_size": sae_size if sae_size is not None else "none",
                    "model_label": model_label,
                    "status": "missing_summary",
                    "summary_path": summary_path,
                })
                continue

            df_one = pd.read_csv(summary_path)
            df_one["model_name"] = model_name
            df_one["sae_size"] = sae_size if sae_size is not None else "none"
            df_one["model_label"] = model_label
            all_dfs.append(df_one)

            print("Loaded:", summary_path)
            run_log.append({
                "model_name": model_name,
                "sae_size": sae_size if sae_size is not None else "none",
                "model_label": model_label,
                "status": "loaded",
                "summary_path": summary_path,
            })

            if make_per_model_plots:
                per_model_plot_dir = os.path.join(swap_dir, "plots")
                print("Plotting per-model swap summary:", model_label)
                plot_swap_summary(
                    summary_df=df_one,
                    output_dir=per_model_plot_dir,
                    emotions=emotions,
                    flip_away_col=flip_away_col,
                    flip_target_col=flip_target_col,
                    target_logit_col=target_logit_col,
                    source_logit_col=source_logit_col,
                    pair_curves=pair_curves,
                    top_k=min(top_k, 30),
                    show=show,
                    save=save,
                    dpi=dpi,
                )

    run_log_df = pd.DataFrame(run_log)
    run_log_path = os.path.join(output_dir, "all_model_swap_run_log.csv")
    run_log_df.to_csv(run_log_path, index=False)
    print("Saved:", run_log_path)

    if len(all_dfs) == 0:
        print("No swap summaries found.")
        return {
            "df": pd.DataFrame(),
            "run_log": run_log_df,
            "figures": {},
        }

    df = pd.concat(all_dfs, ignore_index=True)

    required = [
        "model_label",
        "source_emotion",
        "target_emotion",
        "layer_idx",
        flip_away_col,
        flip_target_col,
        target_logit_col,
        source_logit_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce")
    df = df.dropna(subset=["layer_idx"]).copy()
    df["layer_idx"] = df["layer_idx"].astype(int)
    df = sanitize_numeric(df, [flip_away_col, flip_target_col, target_logit_col, source_logit_col])

    df["pair"] = df["source_emotion"].astype(str) + "→" + df["target_emotion"].astype(str)
    df["model_pair"] = df["model_label"].astype(str) + " | " + df["pair"]

    all_summary_path = os.path.join(output_dir, "all_model_swap_summary.csv")
    df.to_csv(all_summary_path, index=False)
    print("Saved:", all_summary_path)

    figures = {}

    # Best layer for every model/source/target pair.
    best_pair_layer = (
        df.sort_values(flip_target_col, ascending=False)
        .groupby(["model_label", "source_emotion", "target_emotion"], as_index=False)
        .first()
    )
    best_pair_layer["pair"] = (
        best_pair_layer["source_emotion"].astype(str)
        + "→"
        + best_pair_layer["target_emotion"].astype(str)
    )
    best_pair_layer.to_csv(
        os.path.join(output_dir, "best_layer_per_model_source_target_pair.csv"),
        index=False,
    )

    # ============================================================
    # Plot 1: model x pair heatmap, best-layer flip-to-target.
    # ============================================================
    pivot_model_pair_target = best_pair_layer.pivot_table(
        index="model_label",
        columns="pair",
        values=flip_target_col,
        aggfunc="mean",
    )
    ordered_pairs = [f"{s}→{t}" for s in emotions for t in emotions if s != t]
    pivot_model_pair_target = pivot_model_pair_target.reindex(
        columns=[p for p in ordered_pairs if p in pivot_model_pair_target.columns]
    )
    figures["heatmap_model_pair_max_flip_to_target"] = draw_heatmap(
        pivot_model_pair_target,
        "Best-layer Swap Conversion Rate by Model and Emotion Pair",
        "Max Flip-to-Target Rate",
        "heatmap_model_pair_max_flip_to_target.png",
        figsize=(max(12, 0.42 * len(pivot_model_pair_target.columns)), max(5, 0.45 * len(pivot_model_pair_target.index))),
    )
    pivot_model_pair_target.to_csv(
        os.path.join(output_dir, "pivot_model_pair_max_flip_to_target.csv")
    )

    # ============================================================
    # Plot 2: model x target controllability, averaged over sources.
    # ============================================================
    target_by_model = (
        best_pair_layer
        .groupby(["model_label", "target_emotion"], as_index=False)
        .agg(
            mean_best_flip_to_target=(flip_target_col, "mean"),
            mean_best_flip_away=(flip_away_col, "mean"),
            mean_target_logit_change=(target_logit_col, "mean"),
        )
    )
    pivot_model_target = target_by_model.pivot_table(
        index="model_label",
        columns="target_emotion",
        values="mean_best_flip_to_target",
        aggfunc="mean",
    ).reindex(columns=[e for e in emotions if e in target_by_model["target_emotion"].unique()])
    figures["heatmap_model_target_controllability"] = draw_heatmap(
        pivot_model_target,
        "Target Emotion Controllability Across Models",
        "Mean Best-layer Flip-to-Target Rate",
        "heatmap_model_target_controllability.png",
    )
    target_by_model.to_csv(os.path.join(output_dir, "target_controllability_by_model.csv"), index=False)
    pivot_model_target.to_csv(os.path.join(output_dir, "pivot_model_target_controllability.csv"))

    # ============================================================
    # Plot 3: model x source disruptability, averaged over targets.
    # ============================================================
    source_by_model = (
        best_pair_layer
        .groupby(["model_label", "source_emotion"], as_index=False)
        .agg(
            mean_best_flip_to_target=(flip_target_col, "mean"),
            mean_best_flip_away=(flip_away_col, "mean"),
            mean_source_logit_change=(source_logit_col, "mean"),
        )
    )
    pivot_model_source = source_by_model.pivot_table(
        index="model_label",
        columns="source_emotion",
        values="mean_best_flip_away",
        aggfunc="mean",
    ).reindex(columns=[e for e in emotions if e in source_by_model["source_emotion"].unique()])
    figures["heatmap_model_source_disruptability"] = draw_heatmap(
        pivot_model_source,
        "Source Emotion Disruptability Across Models",
        "Mean Best-layer Flip-away Rate",
        "heatmap_model_source_disruptability.png",
    )
    source_by_model.to_csv(os.path.join(output_dir, "source_disruptability_by_model.csv"), index=False)
    pivot_model_source.to_csv(os.path.join(output_dir, "pivot_model_source_disruptability.csv"))

    # ============================================================
    # Plot 4: average target controllability bar chart across models.
    # ============================================================
    target_avg = (
        target_by_model
        .groupby("target_emotion", as_index=False)
        .agg(
            mean_best_flip_to_target=("mean_best_flip_to_target", "mean"),
            std_best_flip_to_target=("mean_best_flip_to_target", "std"),
            mean_best_flip_away=("mean_best_flip_away", "mean"),
            mean_target_logit_change=("mean_target_logit_change", "mean"),
        )
    )
    target_avg["target_emotion"] = pd.Categorical(target_avg["target_emotion"], categories=emotions, ordered=True)
    target_avg = target_avg.sort_values("target_emotion")

    x = np.arange(len(target_avg))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, target_avg["mean_best_flip_to_target"], width, label="flip to target")
    plt.bar(x + width / 2, target_avg["mean_best_flip_away"], width, label="flip away from source")
    plt.xticks(x, target_avg["target_emotion"].astype(str), rotation=45, ha="right")
    plt.xlabel("Target Emotion")
    plt.ylabel("Mean Best-layer Rate")
    plt.title("Average Swap Target Controllability Across Models")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    figures["bar_average_target_controllability"] = save_current_fig(
        "bar_average_target_controllability.png"
    )
    target_avg.to_csv(os.path.join(output_dir, "average_target_controllability_across_models.csv"), index=False)

    # ============================================================
    # Plot 5: average source disruptability bar chart across models.
    # ============================================================
    source_avg = (
        source_by_model
        .groupby("source_emotion", as_index=False)
        .agg(
            mean_best_flip_to_target=("mean_best_flip_to_target", "mean"),
            mean_best_flip_away=("mean_best_flip_away", "mean"),
            std_best_flip_away=("mean_best_flip_away", "std"),
            mean_source_logit_change=("mean_source_logit_change", "mean"),
        )
    )
    source_avg["source_emotion"] = pd.Categorical(source_avg["source_emotion"], categories=emotions, ordered=True)
    source_avg = source_avg.sort_values("source_emotion")

    x = np.arange(len(source_avg))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, source_avg["mean_best_flip_to_target"], width, label="flip to target")
    plt.bar(x + width / 2, source_avg["mean_best_flip_away"], width, label="flip away from source")
    plt.xticks(x, source_avg["source_emotion"].astype(str), rotation=45, ha="right")
    plt.xlabel("Source Emotion")
    plt.ylabel("Mean Best-layer Rate")
    plt.title("Average Source Emotion Disruptability Across Models")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    figures["bar_average_source_disruptability"] = save_current_fig(
        "bar_average_source_disruptability.png"
    )
    source_avg.to_csv(os.path.join(output_dir, "average_source_disruptability_across_models.csv"), index=False)

    # ============================================================
    # Plot 6: average layer-wise flip-to-target by target emotion.
    # ============================================================
    layer_target_avg = (
        df.groupby(["target_emotion", "layer_idx"], as_index=False)[flip_target_col]
        .mean()
        .sort_values(["target_emotion", "layer_idx"])
    )
    plt.figure(figsize=(10, 6))
    for emotion in emotions:
        sub = layer_target_avg[layer_target_avg["target_emotion"] == emotion]
        if len(sub) == 0:
            continue
        plt.plot(sub["layer_idx"], sub[flip_target_col], marker="o", linewidth=2, label=emotion)
    plt.xlabel("Layer")
    plt.ylabel("Mean Flip-to-Target Rate")
    plt.title("Average Layer-wise Swap Conversion Across Models")
    plt.grid(alpha=0.3)
    plt.legend()
    figures["line_average_layerwise_flip_to_target"] = save_current_fig(
        "line_average_layerwise_flip_to_target_across_models.png"
    )
    layer_target_avg.to_csv(os.path.join(output_dir, "average_layerwise_flip_to_target_across_models.csv"), index=False)

    # ============================================================
    # Plot 7: average layer-wise flip-away by source emotion.
    # ============================================================
    layer_source_avg = (
        df.groupby(["source_emotion", "layer_idx"], as_index=False)[flip_away_col]
        .mean()
        .sort_values(["source_emotion", "layer_idx"])
    )
    plt.figure(figsize=(10, 6))
    for emotion in emotions:
        sub = layer_source_avg[layer_source_avg["source_emotion"] == emotion]
        if len(sub) == 0:
            continue
        plt.plot(sub["layer_idx"], sub[flip_away_col], marker="o", linewidth=2, label=emotion)
    plt.xlabel("Layer")
    plt.ylabel("Mean Flip-away-from-source Rate")
    plt.title("Average Layer-wise Swap Disruption Across Models")
    plt.grid(alpha=0.3)
    plt.legend()
    figures["line_average_layerwise_flip_away"] = save_current_fig(
        "line_average_layerwise_flip_away_across_models.png"
    )
    layer_source_avg.to_csv(os.path.join(output_dir, "average_layerwise_flip_away_across_models.csv"), index=False)

    # ============================================================
    # Plot 8: average source-target pair specificity across models.
    # ============================================================
    pair_avg = (
        best_pair_layer
        .groupby(["source_emotion", "target_emotion"], as_index=False)
        .agg(
            mean_best_flip_to_target=(flip_target_col, "mean"),
            mean_best_flip_away=(flip_away_col, "mean"),
            mean_target_logit_change=(target_logit_col, "mean"),
        )
    )
    pair_avg["target_specificity"] = pair_avg["mean_best_flip_to_target"] / (
        pair_avg["mean_best_flip_away"] + 1e-8
    )
    pair_avg.to_csv(os.path.join(output_dir, "average_source_target_pair_effects_across_models.csv"), index=False)

    pivot_pair_target = pair_avg.pivot(
        index="source_emotion",
        columns="target_emotion",
        values="mean_best_flip_to_target",
    ).reindex(index=emotions, columns=emotions)
    figures["heatmap_average_pair_flip_to_target"] = draw_heatmap(
        pivot_pair_target,
        "Average Best-layer Swap Conversion by Emotion Pair",
        "Mean Best-layer Flip-to-Target Rate",
        "heatmap_average_pair_flip_to_target.png",
        figsize=(8, 6),
    )
    pivot_pair_target.to_csv(os.path.join(output_dir, "pivot_average_pair_flip_to_target.csv"))

    pivot_pair_spec = pair_avg.pivot(
        index="source_emotion",
        columns="target_emotion",
        values="target_specificity",
    ).reindex(index=emotions, columns=emotions)
    figures["heatmap_average_pair_target_specificity"] = draw_heatmap(
        pivot_pair_spec,
        "Average Swap Target Specificity by Emotion Pair",
        "Flip-to-target / Flip-away",
        "heatmap_average_pair_target_specificity.png",
        figsize=(8, 6),
    )
    pivot_pair_spec.to_csv(os.path.join(output_dir, "pivot_average_pair_target_specificity.csv"))

    # ============================================================
    # Plot 9: top model/source/target/layer conversions.
    # ============================================================
    top = df.sort_values(flip_target_col, ascending=False).head(top_k).copy()
    top["label"] = (
        top["model_label"].astype(str)
        + " | "
        + top["source_emotion"].astype(str)
        + "→"
        + top["target_emotion"].astype(str)
        + " L"
        + top["layer_idx"].astype(str)
    )
    top = top.sort_values(flip_target_col, ascending=True)

    plt.figure(figsize=(12, max(5, 0.30 * len(top))))
    plt.barh(top["label"], top[flip_target_col])
    plt.xlabel("Flip-to-Target Rate")
    plt.ylabel("Model | Source→Target Layer")
    plt.title(f"Top {top_k} Layer-local Swap Conversions Across Models")
    plt.grid(axis="x", alpha=0.3)
    figures["bar_top_swap_conversions_across_models"] = save_current_fig(
        f"bar_top_{top_k}_swap_conversions_across_models.png"
    )
    top.to_csv(os.path.join(output_dir, f"top_{top_k}_swap_conversions_across_models.csv"), index=False)

    # Save aggregate tables after all derived columns are ready.
    best_pair_layer.to_csv(os.path.join(output_dir, "best_pair_layer_with_metrics.csv"), index=False)

    print("\nSaved aggregate swap visuals to:", output_dir)
    for name, path in figures.items():
        print(name, ":", path)

    return {
        "df": df,
        "run_log": run_log_df,
        "best_pair_layer": best_pair_layer,
        "target_by_model": target_by_model,
        "source_by_model": source_by_model,
        "target_avg": target_avg,
        "source_avg": source_avg,
        "layer_target_avg": layer_target_avg,
        "layer_source_avg": layer_source_avg,
        "pair_avg": pair_avg,
        "figures": figures,
    }
