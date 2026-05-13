#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_MODULE_PATH = Path(__file__).with_name("influence_trend_line.py")
_SPEC = importlib.util.spec_from_file_location("two_a_line_module", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load module from {_MODULE_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

EDGE_TYPE_ORDER = _MOD.EDGE_TYPE_ORDER
build_yearly_metrics = _MOD.build_yearly_metrics
load_graph = _MOD.load_graph


EDGE_COLOR = {
    "InStyleOf": "#6C8EBF",
    "InterpolatesFrom": "#C97A40",
    "CoverOf": "#5B9B74",
    "LyricalReferenceTo": "#9A68A6",
    "DirectlySamples": "#A65D5D",
}


@st.cache_data(show_spinner=False)
def load_metrics(graph_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    graph = load_graph(graph_path)
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = graph["links"]
    return build_yearly_metrics(nodes, links)


def build_theme_river(
    edge_df: pd.DataFrame,
    x_range: Tuple[float, float],
    selected_edge_types: List[str],
    normalize_share: bool,
) -> go.Figure:
    min_year = math.floor(x_range[0])
    max_year = math.ceil(x_range[1])

    df = edge_df[edge_df["edge_type"].isin(selected_edge_types)].copy()
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]

    pivot = (
        df.pivot_table(index="year", columns="edge_type", values="count", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    for edge_type in selected_edge_types:
        if edge_type not in pivot.columns:
            pivot[edge_type] = 0.0
    pivot = pivot[selected_edge_types]

    if pivot.empty:
        return go.Figure()

    if normalize_share:
        row_sums = pivot.sum(axis=1).replace(0, 1.0)
        values = pivot.div(row_sums, axis=0)
        y_title = "Share of influence events"
    else:
        values = pivot
        y_title = "Influence events"

    years = values.index.tolist()

    fig = go.Figure()

    for edge_type in selected_edge_types:
        series = values[edge_type].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=years,
                y=series,
                mode="lines",
                line=dict(width=0.8, color=EDGE_COLOR.get(edge_type, "#888888")),
                stackgroup="one",
                groupnorm="fraction" if normalize_share else None,
                opacity=0.86,
                name=edge_type,
                customdata=series,
                hovertemplate=(
                    "Year: %{x}<br>"
                    + f"{edge_type} events: "
                    + ("%{y:.3f}" if normalize_share else "%{y:.0f}")
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Theme River: Composition of Influence Types Over Time",
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin=dict(l=70, r=40, t=65, b=50),
        legend=dict(orientation="h", x=0, y=1.02),
    )
    fig.update_xaxes(title="Year", showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(
        title=y_title,
        rangemode="tozero",
        range=[0, 1] if normalize_share else None,
        zeroline=True,
        zerolinecolor="rgba(148, 163, 184, 0.35)",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.15)",
    )
    return fig


def get_time_presets(years: List[int], peak_year: int) -> Dict[str, Tuple[float, float]]:
    min_year = int(min(years))
    max_year = int(max(years))

    presets: Dict[str, Tuple[float, float]] = {
        "Full": (min_year - 0.2, max_year + 0.2),
    }

    early_end = min(max_year, peak_year - 5)
    if early_end > min_year:
        presets["Early Phase"] = (min_year - 0.2, early_end + 0.2)

    buildup_start = max(min_year, peak_year - 4)
    buildup_end = min(max_year, peak_year - 1)
    if buildup_end > buildup_start:
        presets["Build-up"] = (buildup_start - 0.2, buildup_end + 0.2)

    peak_window_start = max(min_year, peak_year - 1)
    peak_window_end = min(max_year, peak_year + 3)
    if peak_window_end > peak_window_start:
        presets["Peak Window"] = (peak_window_start - 0.2, peak_window_end + 0.2)

    recent_start = min(max_year, peak_year + 4)
    if recent_start < max_year:
        presets["Recent Tail"] = (recent_start - 0.2, max_year + 0.2)

    presets["Custom"] = (min_year - 0.2, max_year + 0.2)
    return presets


def get_edge_visible_indices(edge_mode: str) -> List[int]:
    # Trace index map (same order as build_figure traces).
    # 0 yearly, 1 MA, 2 cumulative, 3 influence events,
    # 4 unique artists, 5 unique genres,
    # 6..10 edge type lines.
    core_idx = [0, 1, 2, 3]
    reach_idx = [4, 5]
    edge_idx = {edge_type: 6 + i for i, edge_type in enumerate(EDGE_TYPE_ORDER)}

    if edge_mode == "Core":
        return core_idx
    if edge_mode == "Core + Reach":
        return core_idx + reach_idx
    if edge_mode == "Style/Sample":
        return core_idx + reach_idx + [
            edge_idx["InStyleOf"],
            edge_idx["InterpolatesFrom"],
            edge_idx["DirectlySamples"],
        ]
    if edge_mode == "Reference/Cover":
        return core_idx + reach_idx + [
            edge_idx["CoverOf"],
            edge_idx["LyricalReferenceTo"],
        ]
    return core_idx + reach_idx + [edge_idx[t] for t in EDGE_TYPE_ORDER]


def detect_change_points(df: pd.DataFrame, top_k: int) -> List[dict]:
    if df.empty or top_k <= 0:
        return []

    working = df.copy()
    working["delta"] = working["influenced_works"].diff()
    working["delta_abs"] = working["delta"].abs()
    working = working.dropna(subset=["delta"])
    if working.empty:
        return []

    selected = working.nlargest(top_k, "delta_abs")
    points: List[dict] = []
    for _, row in selected.iterrows():
        year = int(row["year"])
        delta = float(row["delta"])

        dominant_type = "N/A"
        dominant_count = 0
        for edge_type in EDGE_TYPE_ORDER:
            count = int(row.get(f"events_{edge_type}", 0))
            if count > dominant_count:
                dominant_count = count
                dominant_type = edge_type

        direction = "up" if delta >= 0 else "down"
        points.append(
            {
                "year": year,
                "delta": delta,
                "direction": direction,
                "dominant_type": dominant_type,
                "dominant_count": dominant_count,
            }
        )

    points.sort(key=lambda x: x["year"])
    return points


def build_figure(
    df: pd.DataFrame,
    visible_indices: List[int],
    x_range: Tuple[float, float],
    change_display_mode: str,
    change_point_count: int,
    show_peak_annotation: bool,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Core traces.
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["influenced_works"],
            mode="lines+markers",
            name="Yearly Works",
            line=dict(color="#2F6C9E", width=2.6),
            marker=dict(size=6),
            hovertemplate="Year: %{x}<br>Yearly works: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["works_3yr_ma"],
            mode="lines",
            name="3-Year MA",
            line=dict(color="#1F4E79", width=2, dash="dot"),
            hovertemplate="Year: %{x}<br>3-year MA: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["cumulative_works"],
            mode="lines",
            name="Cumulative Works",
            line=dict(color="#D97706", width=2.8),
            hovertemplate="Year: %{x}<br>Cumulative works: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=df["year"],
            y=df["influence_events"],
            name="Influence Events",
            marker_color="#8EC1D6",
            opacity=0.42,
            hovertemplate="Year: %{x}<br>Influence events: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Reach traces.
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["unique_artists"],
            mode="lines+markers",
            name="Unique Affected Artists",
            line=dict(color="#2E8B57", width=2),
            marker=dict(size=5),
            hovertemplate="Year: %{x}<br>Unique artists: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["unique_genres"],
            mode="lines+markers",
            name="Unique Affected Genres",
            line=dict(color="#A85577", width=2),
            marker=dict(size=5),
            hovertemplate="Year: %{x}<br>Unique genres: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Edge-type traces.
    for edge_type in EDGE_TYPE_ORDER:
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[f"events_{edge_type}"],
                mode="lines",
                name=f"Events - {edge_type}",
                line=dict(color=EDGE_COLOR.get(edge_type, "#888888"), width=1.8),
                hovertemplate=f"Year: %{{x}}<br>{edge_type} events: %{{y}}<extra></extra>",
            ),
            secondary_y=False,
        )

    # Apply visibility preset.
    for idx, trace in enumerate(fig.data):
        trace.visible = idx in visible_indices
        trace.showlegend = idx in visible_indices

    peak_row = df.loc[df["influenced_works"].idxmax()]
    peak_year = int(peak_row["year"])

    fig.update_layout(
        title=dict(text="Q2a: Oceanus Folk Influence Over Time", x=0.02, y=0.97, xanchor="left"),
        template="plotly_white",
        barmode="overlay",
        hovermode="x unified",
        dragmode="zoom",
        height=760,
        margin=dict(l=70, r=70, t=75, b=130),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        xaxis=dict(
            title="Year",
            range=list(x_range),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.2)",
            rangeslider=dict(
                visible=True,
                thickness=0.15,
                bgcolor="rgba(241, 245, 249, 0.9)",
                bordercolor="#CBD5E1",
                borderwidth=1,
                yaxis=dict(rangemode="match"),
            ),
        ),
    )

    fig.update_yaxes(
        title_text="Yearly Metrics (works/events/artists/genres)",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.2)",
        fixedrange=False,
    )
    fig.update_yaxes(
        title_text="Cumulative Influenced Works",
        secondary_y=True,
        fixedrange=False,
    )

    if show_peak_annotation:
        fig.add_annotation(
            x=peak_year,
            y=int(peak_row["influenced_works"]),
            text=f"Peak {peak_year}: {int(peak_row['influenced_works'])}",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-42,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#CBD5E1",
        )

    if change_display_mode != "Off":
        points = detect_change_points(df, top_k=change_point_count)
        marker_x: List[int] = []
        marker_y: List[float] = []
        marker_text: List[str] = []
        for i, pt in enumerate(points):
            year = pt["year"]
            row = df[df["year"] == year]
            if row.empty:
                continue
            y_val = float(row["influenced_works"].iloc[0])
            sign = "+" if pt["delta"] >= 0 else ""
            txt = (
                f"Shift {year}: {sign}{pt['delta']:.0f}<br>"
                f"Top type: {pt['dominant_type']} ({pt['dominant_count']})"
            )

            fig.add_vline(
                x=year,
                line_width=1,
                line_dash="dot",
                line_color="rgba(71, 85, 105, 0.45)",
            )

            marker_x.append(year)
            marker_y.append(y_val)
            marker_text.append(
                f"Shift {year}: {sign}{pt['delta']:.0f}<br>"
                f"Top type: {pt['dominant_type']} ({pt['dominant_count']})"
            )

            if change_display_mode == "Lines + Labels":
                fig.add_annotation(
                    x=year,
                    y=y_val,
                    text=txt,
                    showarrow=True,
                    arrowhead=2,
                    ax=-40 if i % 2 == 0 else 40,
                    ay=-55 if pt["delta"] >= 0 else 55,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#CBD5E1",
                    font=dict(size=11),
                )

        if marker_x:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    name="Change Points",
                    marker=dict(size=9, color="#334155", symbol="diamond"),
                    hovertemplate="%{text}<extra></extra>",
                    text=marker_text,
                ),
                secondary_y=False,
            )
            fig.data[-1].showlegend = change_display_mode != "Off"

    return fig


def main() -> None:
    st.set_page_config(page_title="Q2a Interactive", layout="wide")
    st.title("Q2a Interactive Dashboard")

    with st.sidebar:
        st.header("Controls")
        graph_path = st.text_input("Graph JSON Path", value="MC1_graph.json")

    try:
        df, edge_df = load_metrics(graph_path)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    if df.empty:
        st.warning("No yearly influence data found with current settings.")
        st.stop()

    years = df["year"].tolist()
    min_year = int(min(years))
    max_year = int(max(years))
    peak_year = int(df.loc[df["influenced_works"].idxmax(), "year"])

    presets = get_time_presets(years, peak_year)

    with st.sidebar:
        st.subheader("Time Preset")
        time_preset = st.radio(
            "Select a time window",
            options=list(presets.keys()),
            index=0,
            label_visibility="collapsed",
        )

        if time_preset == "Custom":
            custom_range = st.slider(
                "Custom year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1,
            )
            x_range = (custom_range[0] - 0.2, custom_range[1] + 0.2)
        else:
            x_range = presets[time_preset]

        st.subheader("Edge Detail Preset")
        edge_mode = st.radio(
            "Select detail mode",
            options=["Core", "Core + Reach", "Style/Sample", "Reference/Cover", "All Details"],
            index=0,
            label_visibility="collapsed",
        )
        st.subheader("Theme River")
        river_edge_types = st.multiselect(
            "Influence types",
            options=EDGE_TYPE_ORDER,
            default=EDGE_TYPE_ORDER,
        )
        river_as_share = st.toggle("Normalize as share (0-1)", value=False)
        show_peak_annotation = st.toggle("Show peak annotation", value=True)
        st.subheader("Change-point Annotation")
        change_display_mode = st.radio(
            "Display mode",
            options=["Off", "Lines only", "Lines + Labels"],
            index=1,
        )
        change_point_count = st.slider("Number of change points", min_value=1, max_value=6, value=3, step=1)

    visible_indices = get_edge_visible_indices(edge_mode)
    fig = build_figure(
        df,
        visible_indices,
        x_range,
        change_display_mode=change_display_mode,
        change_point_count=change_point_count,
        show_peak_annotation=show_peak_annotation,
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

    st.subheader("Theme River of Influence Types")
    if river_edge_types:
        river_fig = build_theme_river(
            edge_df=edge_df,
            x_range=x_range,
            selected_edge_types=river_edge_types,
            normalize_share=river_as_share,
        )
        st.plotly_chart(
            river_fig,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "doubleClick": "reset",
                "responsive": True,
            },
        )
    else:
        st.info("Select at least one influence type to render the theme river.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Year", peak_year)
    c2.metric("Peak Yearly Works", int(df["influenced_works"].max()))
    c3.metric("Total Cumulative Works", int(df["cumulative_works"].max()))

    with st.expander("Data Preview"):
        st.write("Yearly metrics")
        st.dataframe(df, use_container_width=True)
        st.write("Edge-type yearly metrics")
        st.dataframe(edge_df, use_container_width=True)


if __name__ == "__main__":
    main()
