#!/usr/bin/env python3
"""Q2b network export for Gephi/Cytoscape.

This script builds a richer Oceanus Folk influence-collaboration network and exports:
- GraphML (recommended for Cytoscape)
- GEXF (recommended for Gephi)
- node/edge CSV tables

Design goals:
1) richer structure than star-only graph
2) explicit node size metric representing influence signal
3) clean attributes for external styling/layout tools
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, Set, Tuple

import networkx as nx
import pandas as pd

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


def load_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def node_type(nodes: Dict[int, dict], node_id: int) -> str:
    return nodes.get(node_id, {}).get("Node Type", "")


def node_name(nodes: Dict[int, dict], node_id: int) -> str:
    n = nodes.get(node_id, {})
    return n.get("name") or n.get("stage_name") or f"id_{node_id}"


def is_artist_node(nodes: Dict[int, dict], node_id: int) -> bool:
    return node_type(nodes, node_id) in {"Person", "MusicalGroup"}


def is_work_node(nodes: Dict[int, dict], node_id: int) -> bool:
    return node_type(nodes, node_id) in {"Song", "Album"}


def build_oceanus_sets(nodes: Dict[int, dict], links: list) -> Tuple[Set[int], Set[int], Set[int]]:
    oceanus_work_ids: Set[int] = {
        node_id
        for node_id, node in nodes.items()
        if node.get("Node Type") in {"Song", "Album"} and node.get("genre") == "Oceanus Folk"
    }

    oceanus_artist_ids: Set[int] = set()
    oceanus_group_ids: Set[int] = set()

    for edge in links:
        edge_type = edge.get("Edge Type")
        src = edge.get("source")
        dst = edge.get("target")

        if edge_type in CREATIVE_EDGE_TYPES and dst in oceanus_work_ids and is_artist_node(nodes, src):
            oceanus_artist_ids.add(src)
            if node_type(nodes, src) == "MusicalGroup":
                oceanus_group_ids.add(src)

    # Expand by membership in Oceanus-associated groups.
    for edge in links:
        if edge.get("Edge Type") == "MemberOf" and edge.get("target") in oceanus_group_ids:
            src = edge.get("source")
            if node_type(nodes, src) == "Person":
                oceanus_artist_ids.add(src)

    return oceanus_work_ids, oceanus_artist_ids, oceanus_group_ids


def build_work_contributors(nodes: Dict[int, dict], links: list) -> Dict[int, Set[int]]:
    contributors: Dict[int, Set[int]] = defaultdict(set)
    for edge in links:
        if edge.get("Edge Type") not in CREATIVE_EDGE_TYPES:
            continue
        src = edge.get("source")
        dst = edge.get("target")
        if is_artist_node(nodes, src) and is_work_node(nodes, dst):
            contributors[dst].add(src)
    return contributors


def collect_signals(nodes: Dict[int, dict], links: list):
    oceanus_work_ids, oceanus_artist_ids, _ = build_oceanus_sets(nodes, links)
    contributors_by_work = build_work_contributors(nodes, links)

    influenced_work_ids: Set[int] = set()

    # 1) Identify non-Oceanus works influenced by Oceanus works/artists.
    for edge in links:
        if edge.get("Edge Type") not in INFLUENCE_EDGE_TYPES:
            continue

        src = edge.get("source")
        dst = edge.get("target")

        if not is_work_node(nodes, src):
            continue

        src_genre = nodes.get(src, {}).get("genre")
        if src_genre == "Oceanus Folk":
            continue

        if dst in oceanus_work_ids or dst in oceanus_artist_ids:
            influenced_work_ids.add(src)

    artist_influence = Counter()
    artist_collaboration = Counter()
    genre_influence = Counter()
    genre_collaboration = Counter()

    artist_genre_influence = Counter()
    artist_genre_collaboration = Counter()
    artist_pair_collaboration = Counter()

    # Work-level influence signals.
    for work_id in influenced_work_ids:
        work_node = nodes.get(work_id, {})
        genre = work_node.get("genre")
        if genre:
            genre_influence[genre] += 1

        non_o_artists = {
            aid
            for aid in contributors_by_work.get(work_id, set())
            if aid not in oceanus_artist_ids and is_artist_node(nodes, aid)
        }

        for artist_id in non_o_artists:
            artist_influence[artist_id] += 1
            if genre:
                artist_genre_influence[(artist_id, genre)] += 1

    # Work-level collaboration signals: any work co-created by Oceanus + non-Oceanus artists.
    for work_id, contributors in contributors_by_work.items():
        if not contributors:
            continue

        has_oceanus = any(aid in oceanus_artist_ids for aid in contributors)
        if not has_oceanus:
            continue

        non_o_artists = sorted({aid for aid in contributors if aid not in oceanus_artist_ids and is_artist_node(nodes, aid)})
        if not non_o_artists:
            continue

        genre = nodes.get(work_id, {}).get("genre")
        if genre:
            genre_collaboration[genre] += 1

        for artist_id in non_o_artists:
            artist_collaboration[artist_id] += 1
            if genre:
                artist_genre_collaboration[(artist_id, genre)] += 1

        for a, b in combinations(non_o_artists, 2):
            artist_pair_collaboration[(a, b)] += 1

    return {
        "oceanus_work_ids": oceanus_work_ids,
        "oceanus_artist_ids": oceanus_artist_ids,
        "influenced_work_ids": influenced_work_ids,
        "artist_influence": artist_influence,
        "artist_collaboration": artist_collaboration,
        "genre_influence": genre_influence,
        "genre_collaboration": genre_collaboration,
        "artist_genre_influence": artist_genre_influence,
        "artist_genre_collaboration": artist_genre_collaboration,
        "artist_pair_collaboration": artist_pair_collaboration,
    }


def scale_size(value: float, vmax: float, min_size: float, max_size: float) -> float:
    if vmax <= 0:
        return min_size
    ratio = math.sqrt(max(0.0, value) / vmax)
    return min_size + (max_size - min_size) * ratio


def pick_entities(
    artist_influence: Counter,
    artist_collaboration: Counter,
    genre_influence: Counter,
    genre_collaboration: Counter,
    max_artists: int,
    max_genres: int,
    min_artist_signal: int,
    min_genre_signal: int,
    rank_by: str,
) -> Tuple[Set[int], Set[str], Dict[int, int], Dict[str, int]]:
    artist_total = {
        aid: int(artist_influence.get(aid, 0) + artist_collaboration.get(aid, 0))
        for aid in set(artist_influence) | set(artist_collaboration)
    }
    genre_total = {
        g: int(genre_influence.get(g, 0) + genre_collaboration.get(g, 0))
        for g in set(genre_influence) | set(genre_collaboration)
    }

    if rank_by == "influence":
        artist_score = {aid: int(artist_influence.get(aid, 0)) for aid in artist_total}
        genre_score = {g: int(genre_influence.get(g, 0)) for g in genre_total}
    else:
        artist_score = artist_total
        genre_score = genre_total

    sorted_artists = sorted(
        artist_score.items(),
        key=lambda x: (
            x[1],
            artist_influence.get(x[0], 0),
            artist_collaboration.get(x[0], 0),
        ),
        reverse=True,
    )
    sorted_genres = sorted(
        genre_score.items(),
        key=lambda x: (
            x[1],
            genre_influence.get(x[0], 0),
            genre_collaboration.get(x[0], 0),
        ),
        reverse=True,
    )

    selected_artists = {aid for aid, score in sorted_artists if score >= min_artist_signal}
    selected_genres = {g for g, score in sorted_genres if score >= min_genre_signal}

    if len(selected_artists) > max_artists:
        selected_artists = {aid for aid, _ in sorted_artists[:max_artists]}
    if len(selected_genres) > max_genres:
        selected_genres = {g for g, _ in sorted_genres[:max_genres]}

    return selected_artists, selected_genres, artist_total, genre_total


def build_export_graph(
    nodes: Dict[int, dict],
    signals: dict,
    selected_artists: Set[int],
    selected_genres: Set[str],
    artist_total: Dict[int, int],
    genre_total: Dict[str, int],
    min_pair_collab: int,
    max_artist_pairs: int,
) -> nx.Graph:
    G = nx.Graph()

    center_id = "center::Oceanus Folk"
    center_infl = float(len(signals["influenced_work_ids"]))
    center_collab = float(sum(signals["artist_collaboration"].values()))
    center_total = max(1.0, center_infl + center_collab)

    max_artist_signal = max([artist_total.get(a, 0) for a in selected_artists] + [1])
    max_genre_signal = max([genre_total.get(g, 0) for g in selected_genres] + [1])

    G.add_node(
        center_id,
        label="Oceanus Folk",
        category="center",
        original_type="Concept",
        influence_weight=center_infl,
        collaboration_weight=center_collab,
        total_signal=float(center_total),
        size=70.0,
    )

    # Genre nodes.
    for genre in sorted(selected_genres):
        infl = int(signals["genre_influence"].get(genre, 0))
        coll = int(signals["genre_collaboration"].get(genre, 0))
        total = infl + coll
        if total <= 0:
            continue

        gid = f"genre::{genre}"
        G.add_node(
            gid,
            label=genre,
            category="genre",
            original_type="Genre",
            influence_weight=float(infl),
            collaboration_weight=float(coll),
            total_signal=float(total),
            size=scale_size(float(total), float(max_genre_signal), 22.0, 46.0),
        )

        relation = "influence+collaboration" if infl > 0 and coll > 0 else ("influence" if infl > 0 else "collaboration")
        G.add_edge(
            center_id,
            gid,
            edge_kind="center_to_genre",
            relation=relation,
            influence_weight=float(infl),
            collaboration_weight=float(coll),
            weight=float(total),
        )

    # Artist nodes.
    for artist_id in sorted(selected_artists):
        infl = int(signals["artist_influence"].get(artist_id, 0))
        coll = int(signals["artist_collaboration"].get(artist_id, 0))
        total = int(artist_total.get(artist_id, 0))
        if total <= 0:
            continue

        aid = f"artist::{artist_id}"
        G.add_node(
            aid,
            label=node_name(nodes, artist_id),
            category="artist",
            original_type=node_type(nodes, artist_id),
            influence_weight=float(infl),
            collaboration_weight=float(coll),
            total_signal=float(total),
            size=scale_size(float(total), float(max_artist_signal), 18.0, 52.0),
        )

        relation = "influence+collaboration" if infl > 0 and coll > 0 else ("influence" if infl > 0 else "collaboration")
        G.add_edge(
            center_id,
            aid,
            edge_kind="center_to_artist",
            relation=relation,
            influence_weight=float(infl),
            collaboration_weight=float(coll),
            weight=float(total),
        )

    # Artist-Genre edges (adds structure for layout).
    keys = set(signals["artist_genre_influence"]) | set(signals["artist_genre_collaboration"])
    for artist_id, genre in keys:
        if artist_id not in selected_artists or genre not in selected_genres:
            continue

        infl = int(signals["artist_genre_influence"].get((artist_id, genre), 0))
        coll = int(signals["artist_genre_collaboration"].get((artist_id, genre), 0))
        total = infl + coll
        if total <= 0:
            continue

        aid = f"artist::{artist_id}"
        gid = f"genre::{genre}"
        if aid not in G or gid not in G:
            continue

        relation = "influence+collaboration" if infl > 0 and coll > 0 else ("influence" if infl > 0 else "collaboration")
        G.add_edge(
            aid,
            gid,
            edge_kind="artist_to_genre",
            relation=relation,
            influence_weight=float(infl),
            collaboration_weight=float(coll),
            weight=float(total),
        )

    # Artist-Artist collaboration edges (limited to strongest pairs).
    pair_items = [
        (pair, w)
        for pair, w in signals["artist_pair_collaboration"].items()
        if w >= min_pair_collab and pair[0] in selected_artists and pair[1] in selected_artists
    ]
    pair_items.sort(key=lambda x: x[1], reverse=True)
    if max_artist_pairs > 0:
        pair_items = pair_items[:max_artist_pairs]

    for (a, b), w in pair_items:
        aid = f"artist::{a}"
        bid = f"artist::{b}"
        if aid not in G or bid not in G:
            continue
        G.add_edge(
            aid,
            bid,
            edge_kind="artist_to_artist",
            relation="collaboration",
            influence_weight=0.0,
            collaboration_weight=float(w),
            weight=float(w),
        )

    return G


def export_tables(G: nx.Graph, out_nodes_csv: str, out_edges_csv: str) -> None:
    node_rows = []
    for node_id, attrs in G.nodes(data=True):
        row = {"id": node_id}
        for k, v in attrs.items():
            row[k] = v
        node_rows.append(row)

    edge_rows = []
    for src, dst, attrs in G.edges(data=True):
        row = {"source": src, "target": dst}
        for k, v in attrs.items():
            row[k] = v
        edge_rows.append(row)

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)

    nodes_df.to_csv(out_nodes_csv, index=False)
    edges_df.to_csv(out_edges_csv, index=False)


def ensure_output_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2b export network to Gephi/Cytoscape formats.")
    parser.add_argument("--input", default="MC1_graph.json", help="Path to MC1 graph JSON")

    parser.add_argument("--out-graphml", default="outputs/influence_network.graphml", help="GraphML output (Cytoscape-friendly)")
    parser.add_argument("--out-gexf", default="outputs/influence_network.gexf", help="GEXF output (Gephi-friendly)")
    parser.add_argument("--out-nodes-csv", default="outputs/influence_network_nodes.csv", help="Node table CSV")
    parser.add_argument("--out-edges-csv", default="outputs/influence_network_edges.csv", help="Edge table CSV")

    parser.add_argument("--max-artists", type=int, default=180, help="Maximum number of artist nodes")
    parser.add_argument("--max-genres", type=int, default=45, help="Maximum number of genre nodes")
    parser.add_argument("--min-artist-signal", type=int, default=2, help="Minimum artist signal (influence+collaboration)")
    parser.add_argument("--min-genre-signal", type=int, default=2, help="Minimum genre signal (influence+collaboration)")

    parser.add_argument("--min-pair-collab", type=int, default=2, help="Minimum artist-artist collaboration count for pair edges")
    parser.add_argument("--max-artist-pairs", type=int, default=800, help="Maximum number of artist-artist pair edges")
    parser.add_argument(
        "--rank-by",
        choices=["influence", "total"],
        default="total",
        help="Entity ranking metric for selection thresholds/top lists. "
        "'total' uses influence+collaboration; 'influence' aligns with the strict Q2b prompt.",
    )

    args = parser.parse_args()

    graph = load_graph(args.input)
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = graph["links"]

    signals = collect_signals(nodes, links)

    selected_artists, selected_genres, artist_total, genre_total = pick_entities(
        artist_influence=signals["artist_influence"],
        artist_collaboration=signals["artist_collaboration"],
        genre_influence=signals["genre_influence"],
        genre_collaboration=signals["genre_collaboration"],
        max_artists=args.max_artists,
        max_genres=args.max_genres,
        min_artist_signal=args.min_artist_signal,
        min_genre_signal=args.min_genre_signal,
        rank_by=args.rank_by,
    )

    G = build_export_graph(
        nodes=nodes,
        signals=signals,
        selected_artists=selected_artists,
        selected_genres=selected_genres,
        artist_total=artist_total,
        genre_total=genre_total,
        min_pair_collab=args.min_pair_collab,
        max_artist_pairs=args.max_artist_pairs,
    )

    ensure_output_dirs([args.out_graphml, args.out_gexf, args.out_nodes_csv, args.out_edges_csv])

    nx.write_graphml(G, args.out_graphml)
    nx.write_gexf(G, args.out_gexf)
    export_tables(G, args.out_nodes_csv, args.out_edges_csv)

    print(f"Saved GraphML: {args.out_graphml}")
    print(f"Saved GEXF: {args.out_gexf}")
    print(f"Saved node table: {args.out_nodes_csv}")
    print(f"Saved edge table: {args.out_edges_csv}")
    print(f"Graph stats -> nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"Rank-by mode: {args.rank_by}")

    # Quick summary for sanity checks.
    node_counts = Counter(nx.get_node_attributes(G, "category").values())
    edge_counts = Counter(nx.get_edge_attributes(G, "edge_kind").values())
    print("Node categories:", dict(node_counts))
    print("Edge kinds:", dict(edge_counts))


if __name__ == "__main__":
    main()
