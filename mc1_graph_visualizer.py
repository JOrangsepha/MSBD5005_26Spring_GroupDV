import json
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="MC1 Graph Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path("MC1_release/MC1_graph.json")


@st.cache_data
def load_graph(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = pd.DataFrame(data.get("nodes", []))
    edge_key = next((k for k in ["links", "edges", "relationships"] if k in data), None)
    if edge_key is None:
        raise KeyError("No edge list found. Expected one of: links, edges, relationships")
    edges = pd.DataFrame(data.get(edge_key, []))

    if not nodes.empty and "id" in nodes.columns:
        nodes["id"] = pd.to_numeric(nodes["id"], errors="coerce").astype("Int64")
    if not edges.empty:
        edges["source"] = pd.to_numeric(edges["source"], errors="coerce").astype("Int64")
        edges["target"] = pd.to_numeric(edges["target"], errors="coerce").astype("Int64")
        if "Edge Type" not in edges.columns:
            edges["Edge Type"] = "Unknown"

    return data, nodes, edges, edge_key


@st.cache_data
def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    g = nx.DiGraph()
    for _, row in nodes.iterrows():
        node_id = row.get("id")
        if pd.isna(node_id):
            continue
        g.add_node(int(node_id), **row.to_dict())
    for _, row in edges.iterrows():
        src = row.get("source")
        tgt = row.get("target")
        if pd.isna(src) or pd.isna(tgt):
            continue
        g.add_edge(int(src), int(tgt), **row.to_dict())
    return g


@st.cache_data
def make_sankey(nodes: pd.DataFrame, edges: pd.DataFrame):
    node_map = nodes.set_index("id").to_dict(orient="index") if not nodes.empty else {}
    labels = []
    label_to_idx = {}

    def get_label(nid):
        info = node_map.get(nid, {})
        name = info.get("name", f"Node {nid}")
        stg = info.get("stage_name")
        if isinstance(stg, str) and stg.strip():
            return f"{name} ({stg.strip()})"
        return name

    for nid in nodes["id"].dropna().astype(int).tolist()[:1200]:
        lbl = get_label(nid)
        if lbl not in label_to_idx:
            label_to_idx[lbl] = len(labels)
            labels.append(lbl)

    src_idx, tgt_idx, values, colors = [], [], [], []
    edge_subset = edges.head(5000).copy()
    for _, row in edge_subset.iterrows():
        src = row.get("source")
        tgt = row.get("target")
        if pd.isna(src) or pd.isna(tgt):
            continue
        s_lbl = get_label(int(src))
        t_lbl = get_label(int(tgt))
        if s_lbl not in label_to_idx:
            label_to_idx[s_lbl] = len(labels)
            labels.append(s_lbl)
        if t_lbl not in label_to_idx:
            label_to_idx[t_lbl] = len(labels)
            labels.append(t_lbl)
        src_idx.append(label_to_idx[s_lbl])
        tgt_idx.append(label_to_idx[t_lbl])
        values.append(1)
        edge_kind = row.get("Edge Type", "Unknown")
        colors.append("rgba(99,102,241,0.25)" if edge_kind == "InterpolatesFrom" else "rgba(14,165,233,0.15)")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, pad=18, thickness=18),
            link=dict(source=src_idx, target=tgt_idx, value=values, color=colors),
        )
    )
    fig.update_layout(template="plotly_white", height=700, margin=dict(l=10, r=10, t=40, b=10))
    return fig


try:
    data, nodes, edges, edge_key = load_graph(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load graph: {e}")
    st.stop()


graph = build_graph(nodes, edges)

st.title("MC1 Graph Visualizer")
st.caption(f"Loaded from `{DATA_PATH}` using edge key `{edge_key}`")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodes", f"{len(nodes):,}")
c2.metric("Links", f"{len(edges):,}")
c3.metric("Node types", f"{nodes['Node Type'].nunique():,}" if "Node Type" in nodes.columns else "0")
c4.metric("Edge types", f"{edges['Edge Type'].nunique():,}" if "Edge Type" in edges.columns else "0")

left, right = st.columns([1, 1])
with left:
    st.subheader("Node type distribution")
    if "Node Type" in nodes.columns:
        node_type_counts = nodes["Node Type"].fillna("Unknown").value_counts().reset_index()
        node_type_counts.columns = ["Node Type", "Count"]
        st.dataframe(node_type_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No `Node Type` column found.")

with right:
    st.subheader("Edge type distribution")
    if "Edge Type" in edges.columns:
        edge_type_counts = edges["Edge Type"].fillna("Unknown").value_counts().reset_index()
        edge_type_counts.columns = ["Edge Type", "Count"]
        st.dataframe(edge_type_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No `Edge Type` column found.")

st.markdown("---")

filters_left, filters_right, filters_extra = st.columns(3)
with filters_left:
    node_types = sorted(nodes["Node Type"].dropna().astype(str).unique().tolist()) if "Node Type" in nodes.columns else []
    selected_node_types = st.multiselect("Filter node types", node_types, default=node_types[:3] if len(node_types) > 3 else node_types)
with filters_right:
    edge_types = sorted(edges["Edge Type"].dropna().astype(str).unique().tolist()) if "Edge Type" in edges.columns else []
    selected_edge_types = st.multiselect("Filter edge types", edge_types, default=edge_types[:5] if len(edge_types) > 5 else edge_types)
with filters_extra:
    max_nodes = st.number_input("Max nodes in network view", min_value=50, max_value=4000, value=500, step=50)

filtered_nodes = nodes.copy()
if selected_node_types and "Node Type" in filtered_nodes.columns:
    filtered_nodes = filtered_nodes[filtered_nodes["Node Type"].isin(selected_node_types)]

filtered_edges = edges.copy()
if selected_edge_types and "Edge Type" in filtered_edges.columns:
    filtered_edges = filtered_edges[filtered_edges["Edge Type"].isin(selected_edge_types)]

node_ids = set(filtered_nodes["id"].dropna().astype(int).tolist()) if not filtered_nodes.empty else set()
filtered_edges = filtered_edges[
    filtered_edges["source"].isin(node_ids) & filtered_edges["target"].isin(node_ids)
].copy()

st.subheader("Filtered data tables")
tab1, tab2, tab3 = st.tabs(["Nodes", "Edges", "Network"])
with tab1:
    st.dataframe(filtered_nodes.head(5000), use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(filtered_edges.head(5000), use_container_width=True, hide_index=True)
with tab3:
    subgraph_nodes = filtered_nodes.head(int(max_nodes)).copy()
    subgraph_ids = set(subgraph_nodes["id"].dropna().astype(int).tolist())
    subgraph_edges = filtered_edges[
        filtered_edges["source"].isin(subgraph_ids) & filtered_edges["target"].isin(subgraph_ids)
    ].copy()

    if subgraph_nodes.empty:
        st.info("No nodes match the current filters.")
    else:
        fig = make_sankey(subgraph_nodes, subgraph_edges)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Oceanus Folk subgraph (nodes + table)")

if "genre" in nodes.columns and "id" in nodes.columns:
    oceanus_nodes = nodes[nodes["genre"].fillna("").astype(str).str.strip().eq("Oceanus Folk")].copy()
else:
    oceanus_nodes = pd.DataFrame(columns=nodes.columns)

oceanus_ids = set(oceanus_nodes["id"].dropna().astype(int).tolist()) if not oceanus_nodes.empty else set()

oceanus_edge_mask = (
    edges["source"].isin(oceanus_ids) | edges["target"].isin(oceanus_ids)
) if not edges.empty and oceanus_ids else pd.Series([False] * len(edges), index=edges.index)
oceanus_edges = edges[oceanus_edge_mask].copy() if len(edges) else pd.DataFrame(columns=edges.columns)

neighbor_ids = set()
if not oceanus_edges.empty:
    neighbor_ids.update(oceanus_edges["source"].dropna().astype(int).tolist())
    neighbor_ids.update(oceanus_edges["target"].dropna().astype(int).tolist())

oceanus_sub_nodes = nodes[nodes["id"].isin(neighbor_ids)].copy() if neighbor_ids else oceanus_nodes.copy()

m1, m2, m3 = st.columns(3)
m1.metric("Oceanus Folk core nodes", f"{len(oceanus_nodes):,}")
m2.metric("Oceanus-related edges", f"{len(oceanus_edges):,}")
m3.metric("Oceanus subgraph nodes", f"{len(oceanus_sub_nodes):,}")

st.markdown("#### Oceanus Folk share among songs released each year")
if {"Node Type", "release_date", "genre"}.issubset(nodes.columns):
    songs_all = nodes[nodes["Node Type"].astype(str).str.strip().eq("Song")].copy()
    songs_all["release_year"] = pd.to_numeric(songs_all["release_date"], errors="coerce")
    songs_all = songs_all[songs_all["release_year"].notna()].copy()
    songs_all["release_year"] = songs_all["release_year"].astype(int)

    if songs_all.empty:
        st.info("No valid song release years found for ratio calculation.")
    else:
        yearly_total = songs_all.groupby("release_year").size().rename("total_songs")
        yearly_oceanus = songs_all[
            songs_all["genre"].fillna("").astype(str).str.strip().eq("Oceanus Folk")
        ].groupby("release_year").size().rename("oceanus_songs")

        yearly_share = pd.concat([yearly_total, yearly_oceanus], axis=1).fillna(0).reset_index()
        yearly_share["oceanus_songs"] = yearly_share["oceanus_songs"].astype(int)
        yearly_share["share_pct"] = (yearly_share["oceanus_songs"] / yearly_share["total_songs"] * 100).round(4)
        yearly_share = yearly_share.sort_values("release_year")

        ratio_fig = go.Figure()
        ratio_fig.add_trace(
            go.Scatter(
                x=yearly_share["release_year"],
                y=yearly_share["share_pct"],
                mode="lines+markers",
                name="Oceanus Folk share",
                line=dict(color="#000000", width=3),
                marker=dict(size=6, color="#000000"),
                customdata=yearly_share[["oceanus_songs", "total_songs"]].values,
                hovertemplate=(
                    "Year: %{x}<br>"
                    "Oceanus songs: %{customdata[0]}<br>"
                    "Total songs: %{customdata[1]}<br>"
                    "Share: %{y:.2f}%<extra></extra>"
                ),
            )
        )
        min_year = int(yearly_share["release_year"].min())
        max_year = int(yearly_share["release_year"].max())
        split_year = 2028
        ratio_fig.add_vrect(
            x0=min_year,
            x1=split_year,
            fillcolor="rgba(255, 217, 47, 0.18)",
            line_width=0,
            layer="below",
        )
        ratio_fig.add_vrect(
            x0=split_year,
            x1=max_year,
            fillcolor="rgba(231, 138, 195, 0.18)",
            line_width=0,
            layer="below",
        )
        avg_share = float(yearly_share["share_pct"].mean())
        ratio_fig.add_hline(
            y=avg_share,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text=f"Average share: {avg_share:.2f}%",
            annotation_position="top right",
        )
        ratio_fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Release year",
            yaxis_title="Oceanus Folk share (%)",
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(ratio_fig, use_container_width=True, key="oceanus-yearly-share-line")

        with st.expander("View yearly ratio table"):
            st.dataframe(
                yearly_share.rename(
                    columns={
                        "release_year": "Year",
                        "total_songs": "Total songs",
                        "oceanus_songs": "Oceanus Folk songs",
                        "share_pct": "Oceanus share (%)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("Missing one of required columns: `Node Type`, `release_date`, `genre`.")

oceanus_max_nodes = st.number_input(
    "Max nodes in Oceanus subgraph view",
    min_value=50,
    max_value=5000,
    value=600,
    step=50,
    key="oceanus_max_nodes",
)

if oceanus_sub_nodes.empty:
    st.info("No Oceanus Folk nodes found in current dataset.")
else:
    st.markdown("#### Share of songs influenced by Oceanus Folk over time")
    if {"Node Type", "release_date", "id"}.issubset(nodes.columns) and {"source", "target"}.issubset(edges.columns):
        songs_all = nodes[nodes["Node Type"].astype(str).str.strip().eq("Song")].copy()
        songs_all["release_year"] = pd.to_numeric(songs_all["release_date"], errors="coerce")
        songs_all = songs_all[songs_all["release_year"].notna()].copy()
        songs_all["release_year"] = songs_all["release_year"].astype(int)

        if songs_all.empty:
            st.info("No valid song release years found for influence ratio calculation.")
        else:
            influence_edge_types = ["InStyleOf", "InterpolatesFrom", "DirectlySamples", "CoverOf", "LyricalReferenceTo"]
            influence_edges = edges.copy()
            if "Edge Type" in influence_edges.columns:
                influence_edges = influence_edges[influence_edges["Edge Type"].isin(influence_edge_types)].copy()

            oceanus_influence_edges = influence_edges[influence_edges["source"].isin(oceanus_ids)].copy()
            oceanus_target_ids = set(oceanus_influence_edges["target"].dropna().astype(int).tolist())

            influenced_songs = songs_all[songs_all["id"].isin(oceanus_target_ids)].copy()

            yearly_total = songs_all.groupby("release_year").size().rename("total_songs")
            yearly_influenced = influenced_songs.groupby("release_year").size().rename("influenced_songs")

            yearly_influence_share = pd.concat([yearly_total, yearly_influenced], axis=1).fillna(0).reset_index()
            yearly_influence_share["influenced_songs"] = yearly_influence_share["influenced_songs"].astype(int)
            yearly_influence_share["share_pct"] = (
                yearly_influence_share["influenced_songs"] / yearly_influence_share["total_songs"] * 100
            ).round(4)
            yearly_influence_share = yearly_influence_share.sort_values("release_year")

            if yearly_influence_share.empty:
                st.info("No influenced song ratio data available.")
            else:
                fig_influence_ratio = go.Figure()
                fig_influence_ratio.add_trace(
                    go.Scatter(
                        x=yearly_influence_share["release_year"],
                        y=yearly_influence_share["share_pct"],
                        mode="lines+markers",
                        name="Influenced by Oceanus Folk",
                        line=dict(color="#000000", width=3),
                        marker=dict(size=6, color="#000000"),
                        customdata=yearly_influence_share[["influenced_songs", "total_songs"]].values,
                        hovertemplate=(
                            "Year: %{x}<br>"
                            "Influenced songs: %{customdata[0]}<br>"
                            "Total songs: %{customdata[1]}<br>"
                            "Share: %{y:.2f}%<extra></extra>"
                        ),
                    )
                )
                min_year = int(yearly_influence_share["release_year"].min())
                max_year = int(yearly_influence_share["release_year"].max())
                split_year = 2028
                fig_influence_ratio.add_vrect(
                    x0=min_year,
                    x1=split_year,
                    fillcolor="rgba(255, 217, 47, 0.18)",
                    line_width=0,
                    layer="below",
                )
                fig_influence_ratio.add_vrect(
                    x0=split_year,
                    x1=max_year,
                    fillcolor="rgba(231, 138, 195, 0.18)",
                    line_width=0,
                    layer="below",
                )
                avg_share_2 = float(yearly_influence_share["share_pct"].mean())
                fig_influence_ratio.add_hline(
                    y=avg_share_2,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=2,
                    annotation_text=f"Average share: {avg_share_2:.2f}%",
                    annotation_position="top right",
                )
                fig_influence_ratio.update_layout(
                    template="plotly_white",
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Release year",
                    yaxis_title="Influenced-song share (%)",
                    yaxis=dict(ticksuffix="%"),
                )
                st.plotly_chart(fig_influence_ratio, use_container_width=True, key="oceanus-influence-share-line")

                with st.expander("View influenced-song ratio table"):
                    st.dataframe(
                        yearly_influence_share.rename(
                            columns={
                                "release_year": "Year",
                                "total_songs": "Total songs",
                                "influenced_songs": "Songs influenced by Oceanus Folk",
                                "share_pct": "Influenced share (%)",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info("Missing required columns for influence ratio chart.")

    oceanus_plot_nodes = oceanus_sub_nodes.head(int(oceanus_max_nodes)).copy()
    oceanus_plot_ids = set(oceanus_plot_nodes["id"].dropna().astype(int).tolist())
    oceanus_plot_edges = oceanus_edges[
        oceanus_edges["source"].isin(oceanus_plot_ids) & oceanus_edges["target"].isin(oceanus_plot_ids)
    ].copy()

    oceanus_fig = make_sankey(oceanus_plot_nodes, oceanus_plot_edges)
    st.plotly_chart(oceanus_fig, use_container_width=True, key="oceanus-subgraph-sankey")

    t1, t2 = st.tabs(["Oceanus-related nodes table", "Oceanus-related edges table"])
    with t1:
        st.dataframe(oceanus_sub_nodes.head(8000), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(oceanus_edges.head(8000), use_container_width=True, hide_index=True)
