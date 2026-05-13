#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INFLUENCE_EDGE_TYPES = {
    "InStyleOf",
    "InterpolatesFrom",
    "CoverOf",
    "LyricalReferenceTo",
    "DirectlySamples",
}

CREATIVE_EDGE_TYPES = {
    "PerformerOf",
    "ComposerOf",
    "LyricistOf",
    "ProducerOf",
}

EDGE_TYPE_ORDER = [
    "InStyleOf",
    "InterpolatesFrom",
    "CoverOf",
    "LyricalReferenceTo",
    "DirectlySamples",
]


def parse_year(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        year = int(raw)
        if 1800 <= year <= 2100:
            return year
        return None

    text = str(raw).strip()
    if not text:
        return None

    # Robust parsing: use explicit 4-digit year tokens only.
    # This avoids accidental parses from mixed numeric strings.
    match = re.search(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)", text)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2100:
            return year
    return None


def pick_year(node: dict) -> Optional[int]:
    for key in ("release_date", "notoriety_date", "written_date"):
        year = parse_year(node.get(key))
        if year is not None:
            return year
    return None


def load_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_oceanus_sets(nodes: Dict[int, dict], links: list) -> Tuple[Set[int], Set[int]]:
    oceanus_work_ids: Set[int] = {
        node_id
        for node_id, node in nodes.items()
        if node.get("Node Type") in {"Song", "Album"} and node.get("genre") == "Oceanus Folk"
    }

    oceanus_artist_ids: Set[int] = set()
    for edge in links:
        edge_type = edge.get("Edge Type")
        src = edge.get("source")
        dst = edge.get("target")

        if edge_type in CREATIVE_EDGE_TYPES and dst in oceanus_work_ids:
            src_type = nodes.get(src, {}).get("Node Type")
            if src_type in {"Person", "MusicalGroup"}:
                oceanus_artist_ids.add(src)

    oceanus_group_ids = {
        node_id
        for node_id in oceanus_artist_ids
        if nodes.get(node_id, {}).get("Node Type") == "MusicalGroup"
    }

    for edge in links:
        if edge.get("Edge Type") == "MemberOf" and edge.get("target") in oceanus_group_ids:
            src = edge.get("source")
            if nodes.get(src, {}).get("Node Type") == "Person":
                oceanus_artist_ids.add(src)

    return oceanus_work_ids, oceanus_artist_ids


def build_source_work_contributors(nodes: Dict[int, dict], links: list) -> Dict[int, Set[int]]:
    contributors: Dict[int, Set[int]] = defaultdict(set)
    for edge in links:
        if edge.get("Edge Type") not in CREATIVE_EDGE_TYPES:
            continue
        src = edge.get("source")
        dst = edge.get("target")
        src_type = nodes.get(src, {}).get("Node Type")
        dst_type = nodes.get(dst, {}).get("Node Type")
        if src_type in {"Person", "MusicalGroup"} and dst_type in {"Song", "Album"}:
            contributors[dst].add(src)
    return contributors


def build_yearly_metrics(nodes: Dict[int, dict], links: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oceanus_work_ids, oceanus_artist_ids = build_oceanus_sets(nodes, links)
    contributors_by_work = build_source_work_contributors(nodes, links)

    works_by_year: Dict[int, Set[int]] = defaultdict(set)
    genres_by_year: Dict[int, Set[str]] = defaultdict(set)
    artists_by_year: Dict[int, Set[int]] = defaultdict(set)
    edge_type_events_by_year: Dict[int, Counter] = defaultdict(Counter)

    for edge in links:
        edge_type = edge.get("Edge Type")
        if edge_type not in INFLUENCE_EDGE_TYPES:
            continue

        src = edge.get("source")
        dst = edge.get("target")
        src_node = nodes.get(src, {})
        dst_node = nodes.get(dst, {})

        if src_node.get("Node Type") not in {"Song", "Album"}:
            continue

        oceanus_influence = False
        if dst in oceanus_work_ids:
            oceanus_influence = True
        elif dst in oceanus_artist_ids and dst_node.get("Node Type") in {"Person", "MusicalGroup"}:
            oceanus_influence = True

        if not oceanus_influence:
            continue

        src_genre = src_node.get("genre")
        if src_genre == "Oceanus Folk":
            continue

        year = pick_year(src_node)
        if year is None:
            continue

        works_by_year[year].add(src)
        edge_type_events_by_year[year][edge_type] += 1

        if src_genre:
            genres_by_year[year].add(src_genre)

        for artist_id in contributors_by_work.get(src, set()):
            if artist_id not in oceanus_artist_ids:
                artists_by_year[year].add(artist_id)

    if not works_by_year:
        return pd.DataFrame(), pd.DataFrame()

    years = list(range(min(works_by_year), max(works_by_year) + 1))

    records = []
    edge_records = []
    running = 0
    for year in years:
        influenced_works = len(works_by_year.get(year, set()))
        running += influenced_works
        influence_events = sum(edge_type_events_by_year.get(year, Counter()).values())
        unique_artists = len(artists_by_year.get(year, set()))
        unique_genres = len(genres_by_year.get(year, set()))

        row = {
            "year": year,
            "influenced_works": influenced_works,
            "cumulative_works": running,
            "influence_events": influence_events,
            "unique_artists": unique_artists,
            "unique_genres": unique_genres,
        }

        for edge_type in EDGE_TYPE_ORDER:
            value = edge_type_events_by_year.get(year, Counter()).get(edge_type, 0)
            row[f"events_{edge_type}"] = value
            edge_records.append({"year": year, "edge_type": edge_type, "count": value})

        records.append(row)

    df = pd.DataFrame(records)
    df["works_3yr_ma"] = df["influenced_works"].rolling(window=3, min_periods=1).mean().round(2)

    edge_df = pd.DataFrame(edge_records)
    return df, edge_df


def build_interactive_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    years = df["year"].tolist()
    min_year = int(min(years))
    max_year = int(max(years))
    peak_row = df.loc[df["influenced_works"].idxmax()]
    peak_year = int(peak_row["year"])

    # Core traces (default visible and shown in legend).
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["influenced_works"],
            mode="lines+markers",
            name="Yearly Works",
            line=dict(color="#2F6C9E", width=2.6),
            marker=dict(size=6),
            hovertemplate="Year: %{x}<br>Yearly works: %{y}<extra></extra>",
            showlegend=True,
            visible=True,
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
            showlegend=True,
            visible=True,
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
            showlegend=True,
            visible=True,
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
            showlegend=True,
            visible=True,
        ),
        secondary_y=False,
    )

    # Optional traces (hidden by default and hidden from legend).
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["unique_artists"],
            mode="lines+markers",
            name="Unique Affected Artists",
            line=dict(color="#2E8B57", width=2),
            marker=dict(size=5),
            hovertemplate="Year: %{x}<br>Unique artists: %{y}<extra></extra>",
            showlegend=False,
            visible=False,
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
            showlegend=False,
            visible=False,
        ),
        secondary_y=False,
    )

    edge_colors = {
        "InStyleOf": "#6C8EBF",
        "InterpolatesFrom": "#C97A40",
        "CoverOf": "#5B9B74",
        "LyricalReferenceTo": "#9A68A6",
        "DirectlySamples": "#A65D5D",
    }
    for edge_type in EDGE_TYPE_ORDER:
        col = f"events_{edge_type}"
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[col],
                mode="lines",
                name=f"Events - {edge_type}",
                line=dict(color=edge_colors.get(edge_type, "#888888"), width=1.8),
                hovertemplate=f"Year: %{{x}}<br>{edge_type} events: %{{y}}<extra></extra>",
                showlegend=False,
                visible=False,
            ),
            secondary_y=False,
        )

    # Trace indices for preset buttons.
    core_idx = [0, 1, 2, 3]
    reach_idx = [4, 5]
    edge_idx = {edge_type: 6 + i for i, edge_type in enumerate(EDGE_TYPE_ORDER)}
    all_idx = core_idx + reach_idx + list(edge_idx.values())

    def visible_array(enabled_indices):
        arr = [False] * len(fig.data)
        for i in enabled_indices:
            arr[i] = True
        return arr

    # Time window presets.
    early_end = min(max_year, peak_year - 5)
    buildup_start = max(min_year, peak_year - 4)
    buildup_end = min(max_year, peak_year - 1)
    peak_window_start = max(min_year, peak_year - 1)
    peak_window_end = min(max_year, peak_year + 3)
    recent_start = min(max_year, peak_year + 4)

    time_buttons = [
        dict(label="Full", method="relayout", args=[{"xaxis.range": [min_year - 0.2, max_year + 0.2]}]),
    ]
    if early_end > min_year:
        time_buttons.append(
            dict(label="Early Phase", method="relayout", args=[{"xaxis.range": [min_year - 0.2, early_end + 0.2]}])
        )
    if buildup_end > buildup_start:
        time_buttons.append(
            dict(
                label="Build-up",
                method="relayout",
                args=[{"xaxis.range": [buildup_start - 0.2, buildup_end + 0.2]}],
            )
        )
    if peak_window_end > peak_window_start:
        time_buttons.append(
            dict(
                label="Peak Window",
                method="relayout",
                args=[{"xaxis.range": [peak_window_start - 0.2, peak_window_end + 0.2]}],
            )
        )
    if recent_start < max_year:
        time_buttons.append(
            dict(
                label="Recent Tail",
                method="relayout",
                args=[{"xaxis.range": [recent_start - 0.2, max_year + 0.2]}],
            )
        )

    # Edge detail presets.
    edge_buttons = [
        dict(label="Core", method="update", args=[{"visible": visible_array(core_idx)}]),
        dict(label="Core + Reach", method="update", args=[{"visible": visible_array(core_idx + reach_idx)}]),
        dict(
            label="Style/Sample",
            method="update",
            args=[
                {
                    "visible": visible_array(
                        core_idx
                        + reach_idx
                        + [
                            edge_idx["InStyleOf"],
                            edge_idx["InterpolatesFrom"],
                            edge_idx["DirectlySamples"],
                        ]
                    )
                }
            ],
        ),
        dict(
            label="Reference/Cover",
            method="update",
            args=[
                {
                    "visible": visible_array(
                        core_idx + reach_idx + [edge_idx["CoverOf"], edge_idx["LyricalReferenceTo"]]
                    )
                }
            ],
        ),
        dict(label="All Details", method="update", args=[{"visible": visible_array(all_idx)}]),
    ]

    fig.update_layout(
        title=dict(text="Q2a: Oceanus Folk Influence Over Time", x=0.02, y=0.98, xanchor="left"),
        template="plotly_white",
        barmode="overlay",
        hovermode="x unified",
        dragmode="zoom",
        height=780,
        width=1320,
        margin=dict(l=72, r=72, t=170, b=175),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.26,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=time_buttons,
                x=0.01,
                y=1.17,
                xanchor="left",
                yanchor="top",
                showactive=True,
                pad=dict(t=0, r=8),
            ),
            dict(
                type="buttons",
                direction="right",
                buttons=edge_buttons,
                x=0.01,
                y=1.10,
                xanchor="left",
                yanchor="top",
                showactive=True,
                pad=dict(t=0, r=8),
            ),
        ],
        annotations=[
            dict(
                text="Time Preset",
                x=0.01,
                y=1.205,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="#334155"),
            ),
            dict(
                text="Edge Detail Preset",
                x=0.01,
                y=1.135,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="#334155"),
            ),
        ],
        xaxis=dict(
            title="Year",
            range=[min_year - 0.2, max_year + 0.2],
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

    # Add after layout so it is not overwritten by panel labels.
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

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2a interactive line chart for Oceanus Folk influence over time.")
    parser.add_argument("--input", default="MC1_graph.json", help="Path to MC1 graph JSON.")
    parser.add_argument("--out-html", default="outputs/influence_trend_interactive.html", help="Output interactive HTML path.")
    parser.add_argument("--out-csv", default="outputs/influence_trend_yearly.csv", help="Output yearly CSV path.")
    parser.add_argument("--out-edge-csv", default="outputs/influence_trend_edge_type_yearly.csv", help="Output edge-type yearly CSV path.")
    args = parser.parse_args()

    graph = load_graph(args.input)
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = graph["links"]

    df, edge_df = build_yearly_metrics(nodes, links)
    if df.empty:
        raise RuntimeError("No yearly influence data found. Check filtering logic or input graph.")

    for path in (args.out_html, args.out_csv, args.out_edge_csv):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    df.to_csv(args.out_csv, index=False)
    edge_df.to_csv(args.out_edge_csv, index=False)

    fig = build_interactive_figure(df)
    fig.write_html(
        args.out_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "reset",
            "responsive": True,
            "modeBarButtonsToAdd": ["autoScale2d", "resetScale2d"],
        },
    )

    print(f"Saved interactive chart: {args.out_html}")
    print(f"Saved yearly metrics: {args.out_csv}")
    print(f"Saved edge-type metrics: {args.out_edge_csv}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
