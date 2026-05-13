#!/usr/bin/env python3
"""Streamlit app for Q2b network exploration using exported CSV files."""

from __future__ import annotations

import math
import re
from typing import Dict, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CATEGORY_COLORS = {
    "center": "#4A08A7",
    "genre": "#F98219",
    "artist": "#39C39C",
}

RELATION_COLORS = {
    "influence": "#284CC5",
    "collaboration": "#F2BD4B",
    "influence+collaboration": "#7CCEF7",
}


def normalize_plotly_color(color: str, fallback: str = "#999999") -> str:
    """Accept common color inputs; convert 8-digit hex (#RRGGBBAA) to rgba()."""
    if not isinstance(color, str):
        return fallback
    c = color.strip()
    m = re.fullmatch(r"#([0-9a-fA-F]{8})", c)
    if m:
        raw = m.group(1)
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
        a = int(raw[6:8], 16) / 255.0
        return f"rgba({r},{g},{b},{a:.3f})"
    return c or fallback


@st.cache_data(show_spinner=False)
def load_exported_tables(nodes_path: str, edges_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Normalize dtypes for filtering.
    for col in ["influence_weight", "collaboration_weight", "total_signal", "size"]:
        if col in nodes_df.columns:
            nodes_df[col] = pd.to_numeric(nodes_df[col], errors="coerce").fillna(0.0)

    for col in ["influence_weight", "collaboration_weight", "weight"]:
        if col in edges_df.columns:
            edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce").fillna(0.0)

    nodes_df["id"] = nodes_df["id"].astype(str)
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)

    return nodes_df, edges_df


def filter_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    selected_categories: list[str],
    selected_relations: list[str],
    weight_range: Tuple[float, float],
    edge_influence_range: Tuple[float, float],
    min_node_signal: float,
    min_node_influence: float,
    max_nodes: int,
    node_size_metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Category/signal filter on nodes.
    ndf = nodes_df.copy()
    ndf = ndf[ndf["category"].isin(selected_categories)]
    ndf = ndf[ndf["total_signal"] >= min_node_signal]
    ndf = ndf[ndf["influence_weight"] >= min_node_influence]

    # Relation/weight filter on edges.
    edf = edges_df.copy()
    edf = edf[edf["relation"].isin(selected_relations)]
    edf = edf[(edf["weight"] >= weight_range[0]) & (edf["weight"] <= weight_range[1])]
    edf = edf[
        (edf["influence_weight"] >= edge_influence_range[0])
        & (edf["influence_weight"] <= edge_influence_range[1])
    ]

    # Keep only connected IDs that survive node filter.
    valid_ids = set(ndf["id"])
    edf = edf[edf["source"].isin(valid_ids) & edf["target"].isin(valid_ids)]

    # Recompute valid IDs from filtered edges, plus center if present.
    connected_ids = set(edf["source"]).union(set(edf["target"]))
    center_ids = {nid for nid in valid_ids if nid.startswith("center::")}
    connected_ids |= center_ids
    ndf = ndf[ndf["id"].isin(connected_ids)]

    # Optional cap on node count (always keep center node if present).
    if len(ndf) > max_nodes:
        center_df = ndf[ndf["category"] == "center"]
        sort_col = "influence_weight" if node_size_metric == "influence_weight" else "total_signal"
        non_center_df = ndf[ndf["category"] != "center"].sort_values(sort_col, ascending=False)
        remaining = max(0, max_nodes - len(center_df))
        ndf = pd.concat([center_df, non_center_df.head(remaining)], ignore_index=True)
        keep_ids = set(ndf["id"])
        edf = edf[edf["source"].isin(keep_ids) & edf["target"].isin(keep_ids)]

    return ndf, edf


def build_plotly_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    show_labels: bool,
    layout_k: float,
    node_size_scale: float,
    edge_width_scale: float,
    edge_opacity: float,
    node_size_metric: str,
) -> go.Figure:
    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(
            row["id"],
            label=row.get("label", row["id"]),
            category=row.get("category", "artist"),
            original_type=row.get("original_type", ""),
            influence_weight=float(row.get("influence_weight", 0.0)),
            collaboration_weight=float(row.get("collaboration_weight", 0.0)),
            total_signal=float(row.get("total_signal", 0.0)),
            size=float(row.get("size", 10.0)),
        )

    for _, row in edges_df.iterrows():
        if row["source"] not in G.nodes or row["target"] not in G.nodes:
            continue
        G.add_edge(
            row["source"],
            row["target"],
            edge_kind=row.get("edge_kind", ""),
            relation=row.get("relation", "influence"),
            influence_weight=float(row.get("influence_weight", 0.0)),
            collaboration_weight=float(row.get("collaboration_weight", 0.0)),
            weight=float(row.get("weight", 0.0)),
        )

    if G.number_of_nodes() == 0:
        return go.Figure()

    # Force layout for exploratory readability.
    pos = nx.spring_layout(G, seed=42, k=layout_k, iterations=120)

    fig = go.Figure()

    # Edge traces grouped by relation for legend control.
    for relation in sorted(set(nx.get_edge_attributes(G, "relation").values())):
        xs = []
        ys = []
        weights = []
        custom = []

        for u, v, attrs in G.edges(data=True):
            if attrs.get("relation") != relation:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            weights.append(max(attrs.get("weight", 0.0), 0.0))
            custom.append(
                [
                    attrs.get("edge_kind", ""),
                    attrs.get("relation", ""),
                    float(attrs.get("influence_weight", 0.0)),
                    float(attrs.get("collaboration_weight", 0.0)),
                    float(attrs.get("weight", 0.0)),
                ]
            )

        if not xs:
            continue

        mean_weight = (sum(weights) / len(weights)) if weights else 1.0
        line_width = min(7.0, edge_width_scale * (1.0 + 0.28 * math.sqrt(mean_weight)))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(
                    color=normalize_plotly_color(RELATION_COLORS.get(relation, "#999999")),
                    width=line_width,
                ),
                opacity=edge_opacity,
                name=f"Edge: {relation}",
                hoverinfo="skip",
            )
        )

    # Node traces grouped by category.
    group_to_nodes: Dict[str, list] = {}
    group_to_color: Dict[str, str] = {}
    for n, attrs in G.nodes(data=True):
        category = str(attrs.get("category", "artist"))
        key = f"category_{category}"
        group_to_nodes.setdefault(key, []).append(n)
        group_to_color[key] = CATEGORY_COLORS.get(category, "#999999")

    for group_key, node_ids in sorted(group_to_nodes.items(), key=lambda kv: len(kv[1]), reverse=True):
        if not node_ids:
            continue

        xs = [pos[n][0] for n in node_ids]
        ys = [pos[n][1] for n in node_ids]
        labels = [G.nodes[n].get("label", n) for n in node_ids]
        size_values = []
        for n in node_ids:
            d = G.nodes[n]
            metric_value = (
                float(d.get("influence_weight", 0.0))
                if node_size_metric == "influence_weight"
                else float(d.get("total_signal", 0.0))
            )
            size_values.append(max(0.0, metric_value))

        group_max = max(size_values) if size_values else 1.0
        group_max = max(group_max, 1.0)
        sizes = [
            max(6.0, min(46.0, (9.0 + 34.0 * math.sqrt(v / group_max)) * node_size_scale))
            for v in size_values
        ]

        hover_data = []
        for n in node_ids:
            d = G.nodes[n]
            hover_data.append(
                [
                    d.get("label", n),
                    d.get("category", ""),
                    d.get("original_type", ""),
                    float(d.get("influence_weight", 0.0)),
                    float(d.get("collaboration_weight", 0.0)),
                    float(d.get("total_signal", 0.0)),
                    (
                        float(d.get("influence_weight", 0.0))
                        if node_size_metric == "influence_weight"
                        else float(d.get("total_signal", 0.0))
                    ),
                ]
            )

        text_vals = labels if show_labels else None

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if show_labels else "markers",
                text=text_vals,
                textposition="top center",
                textfont=dict(size=9, color="#1F2937"),
                marker=dict(
                    size=sizes,
                    color=normalize_plotly_color(group_to_color[group_key]),
                    line=dict(color="#FFFFFF", width=0.8),
                    opacity=0.93,
                ),
                name=f"Node: {group_key.split('_', 1)[-1]}",
                customdata=hover_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Category: %{customdata[1]}<br>"
                    "Type: %{customdata[2]}<br>"
                    "Influence weight: %{customdata[3]:.1f}<br>"
                    "Collaboration weight: %{customdata[4]:.1f}<br>"
                    "Total signal: %{customdata[5]:.1f}<br>"
                    f"Node size metric ({node_size_metric}): "
                    "%{customdata[6]:.1f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Q2b: Oceanus Folk Influence-Collaboration Network",
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
        height=840,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", x=0, y=1.02, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def main() -> None:
    st.set_page_config(page_title="Q2b Interactive", layout="wide")
    st.title("Q2b Interactive Dashboard")

    with st.sidebar:
        st.header("Data")
        nodes_path = st.text_input("Nodes CSV", value="outputs/influence_network_nodes.csv")
        edges_path = st.text_input("Edges CSV", value="outputs/influence_network_edges.csv")

    try:
        nodes_df, edges_df = load_exported_tables(nodes_path, edges_path)
    except Exception as exc:
        st.error(f"Failed to load exported CSV files: {exc}")
        st.stop()

    if nodes_df.empty or edges_df.empty:
        st.warning("Node or edge table is empty.")
        st.stop()

    categories = sorted(nodes_df["category"].dropna().unique().tolist())
    relations = sorted(edges_df["relation"].dropna().unique().tolist())
    w_min = float(edges_df["weight"].min())
    w_max = float(edges_df["weight"].max())
    edge_inf_min = float(edges_df["influence_weight"].min())
    edge_inf_max = float(edges_df["influence_weight"].max())
    node_inf_max = float(nodes_df["influence_weight"].max())
    max_nodes_limit = int(len(nodes_df))

    with st.sidebar:
        st.header("Filters")
        selected_categories = st.multiselect("Node categories", options=categories, default=categories)
        selected_relations = st.multiselect("Edge relations", options=relations, default=relations)
        weight_range = st.slider(
            "Edge weight range",
            min_value=w_min,
            max_value=w_max,
            value=(w_min, w_max),
            step=1.0,
        )
        if edge_inf_max > edge_inf_min:
            edge_influence_range = st.slider(
                "Edge influence_weight range",
                min_value=edge_inf_min,
                max_value=edge_inf_max,
                value=(edge_inf_min, edge_inf_max),
                step=1.0,
            )
        else:
            edge_influence_range = (edge_inf_min, edge_inf_max)
            st.caption(f"Edge influence_weight range fixed at {edge_inf_min:.1f}")
        min_node_signal = st.slider(
            "Min node total_signal",
            min_value=0.0,
            max_value=float(nodes_df["total_signal"].max()),
            value=0.0,
            step=1.0,
        )
        if node_inf_max > 0:
            min_node_influence = st.slider(
                "Min node influence_weight",
                min_value=0.0,
                max_value=node_inf_max,
                value=0.0,
                step=1.0,
            )
        else:
            min_node_influence = 0.0
            st.caption("Min node influence_weight fixed at 0.0")

        max_nodes = st.slider(
            "Max displayed nodes",
            min_value=20,
            max_value=max_nodes_limit,
            value=min(160, max_nodes_limit),
            step=1,
        )

        show_labels = st.toggle("Show labels", value=False)
        layout_k = st.slider("Layout spacing (spring k)", min_value=0.05, max_value=1.0, value=0.28, step=0.01)
        node_size_scale = st.slider("Node size scale", min_value=0.25, max_value=1.2, value=0.55, step=0.01)
        node_size_metric = st.selectbox(
            "Node size metric",
            options=["total_signal", "influence_weight"],
            index=0,
        )
        edge_width_scale = st.slider("Edge width scale", min_value=0.6, max_value=3.0, value=1.7, step=0.1)
        edge_opacity = st.slider("Edge opacity", min_value=0.2, max_value=1.0, value=0.72, step=0.02)

    if not selected_categories or not selected_relations:
        st.warning("Please select at least one category and one relation.")
        st.stop()

    ndf, edf = filter_network(
        nodes_df=nodes_df,
        edges_df=edges_df,
        selected_categories=selected_categories,
        selected_relations=selected_relations,
        weight_range=weight_range,
        edge_influence_range=edge_influence_range,
        min_node_signal=min_node_signal,
        min_node_influence=min_node_influence,
        max_nodes=max_nodes,
        node_size_metric=node_size_metric,
    )

    if ndf.empty or edf.empty:
        st.warning("No nodes/edges left after filtering. Relax the filters.")
        st.stop()

    fig = build_plotly_network(
        ndf,
        edf,
        show_labels=show_labels,
        layout_k=layout_k,
        node_size_scale=node_size_scale,
        edge_width_scale=edge_width_scale,
        edge_opacity=edge_opacity,
        node_size_metric=node_size_metric,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "reset",
            "responsive": True,
            "modeBarButtonsToAdd": ["autoScale2d", "resetScale2d"],
        },
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes (shown)", int(len(ndf)))
    c2.metric("Edges (shown)", int(len(edf)))
    c3.metric("Node categories", int(ndf["category"].nunique()))
    c4.metric("Edge relations", int(edf["relation"].nunique()))

    top_cols = st.columns(2)
    with top_cols[0]:
        st.subheader("Top Genres by influence_weight")
        top_genres = (
            ndf[ndf["category"] == "genre"][["label", "influence_weight", "collaboration_weight", "total_signal"]]
            .sort_values("influence_weight", ascending=False)
            .head(10)
        )
        st.dataframe(top_genres, use_container_width=True, hide_index=True)
    with top_cols[1]:
        st.subheader("Top Artists by influence_weight")
        top_artists = (
            ndf[ndf["category"] == "artist"][["label", "influence_weight", "collaboration_weight", "total_signal"]]
            .sort_values("influence_weight", ascending=False)
            .head(10)
        )
        st.dataframe(top_artists, use_container_width=True, hide_index=True)

    with st.expander("Data preview"):
        st.write("Filtered nodes")
        st.dataframe(ndf, use_container_width=True)
        st.write("Filtered edges")
        st.dataframe(edf, use_container_width=True)


if __name__ == "__main__":
    main()
