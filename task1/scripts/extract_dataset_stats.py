#!/usr/bin/env python3
"""Extract high-level statistics for MC1 graph data.

Examples:
    python3 scripts/extract_dataset_stats.py
    python3 scripts/extract_dataset_stats.py --output scripts/output/dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract important summary statistics from MC1_graph.json."
    )
    parser.add_argument(
        "--graph-path",
        default="data/MC1_graph.json",
        help="Path to graph JSON (default: data/MC1_graph.json).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N values shown for long distributions (default: 10).",
    )
    parser.add_argument(
        "--plot-dir",
        default="",
        help=(
            "Optional directory to save plots. "
            "If omitted, no plots are generated."
        ),
    )
    return parser.parse_args()


def safe_edge_type(link: Dict[str, Any]) -> str:
    return (
        link.get("Edge Type")
        or link.get("edge_type")
        or link.get("type")
        or "Unknown"
    )


def to_int_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    if text.isdigit():
        return int(text)
    return None


def top_counter_items(counter: Counter, top_n: int) -> Dict[str, int]:
    return dict(counter.most_common(max(1, top_n)))


def percentage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100.0, 2)


def print_distribution(title: str, distribution: Dict[str, int], total: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, count in distribution.items():
        print(f"{key:22s} {count:8d} ({percentage(count, total):6.2f}%)")


def build_stats(data: Dict[str, Any], top_n: int) -> Dict[str, Any]:
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    node_type_counter: Counter = Counter()
    edge_type_counter: Counter = Counter()
    node_genre_counter: Counter = Counter()

    release_years: list[int] = []
    written_years: list[int] = []
    notoriety_years: list[int] = []
    notable_counter: Counter = Counter()

    for node in nodes:
        node_type = node.get("Node Type") or "Unknown"
        node_type_counter[node_type] += 1

        if node_type in {"Song", "Album"}:
            genre = node.get("genre")
            if genre:
                node_genre_counter[str(genre)] += 1

            notable_counter["work_total"] += 1
            if node.get("notable") is True:
                notable_counter["work_notable"] += 1

            release_year = to_int_year(node.get("release_date"))
            if release_year is not None:
                release_years.append(release_year)

            written_year = to_int_year(node.get("written_date"))
            if written_year is not None:
                written_years.append(written_year)

            notoriety_year = to_int_year(node.get("notoriety_date"))
            if notoriety_year is not None:
                notoriety_years.append(notoriety_year)

    for link in links:
        edge_type_counter[safe_edge_type(link)] += 1

    release_year_counter = Counter(release_years)
    written_year_counter = Counter(written_years)
    notoriety_year_counter = Counter(notoriety_years)

    stats: Dict[str, Any] = {
        "totals": {
            "nodes": len(nodes),
            "edges": len(links),
            "directed": bool(data.get("directed")),
            "multigraph": bool(data.get("multigraph")),
        },
        "distributions": {
            "node_types": dict(node_type_counter.most_common()),
            "edge_types": dict(edge_type_counter.most_common()),
            "genres_top_n": top_counter_items(node_genre_counter, top_n),
        },
        "works": {
            "count": notable_counter.get("work_total", 0),
            "notable_count": notable_counter.get("work_notable", 0),
            "notable_pct": percentage(
                notable_counter.get("work_notable", 0),
                notable_counter.get("work_total", 0),
            ),
        },
        "year_ranges": {
            "release_date": {
                "min": min(release_years) if release_years else None,
                "max": max(release_years) if release_years else None,
                "count_with_value": len(release_years),
            },
            "written_date": {
                "min": min(written_years) if written_years else None,
                "max": max(written_years) if written_years else None,
                "count_with_value": len(written_years),
            },
            "notoriety_date": {
                "min": min(notoriety_years) if notoriety_years else None,
                "max": max(notoriety_years) if notoriety_years else None,
                "count_with_value": len(notoriety_years),
            },
        },
        "year_distributions": {
            "release_date": dict(sorted(release_year_counter.items())),
            "written_date": dict(sorted(written_year_counter.items())),
            "notoriety_date": dict(sorted(notoriety_year_counter.items())),
        },
    }
    return stats


def print_summary(stats: Dict[str, Any], top_n: int) -> None:
    totals = stats["totals"]
    dists = stats["distributions"]
    works = stats["works"]
    year_ranges = stats["year_ranges"]

    print("MC1 Graph Dataset Statistics")
    print("============================")
    print(f"Total nodes: {totals['nodes']}")
    print(f"Total edges: {totals['edges']}")
    print(f"Directed: {totals['directed']}")
    print(f"Multigraph: {totals['multigraph']}")

    print_distribution("Node Type Distribution", dists["node_types"], totals["nodes"])
    print_distribution("Edge Type Distribution", dists["edge_types"], totals["edges"])

    print(f"\nTop {top_n} Work Genres (Song/Album)")
    print("-" * (30 + len(str(top_n))))
    genre_total = sum(dists["genres_top_n"].values())
    for genre, count in dists["genres_top_n"].items():
        print(f"{genre:22s} {count:8d} ({percentage(count, genre_total):6.2f}% of top list)")

    print("\nWork Notability")
    print("--------------")
    print(f"Song+Album nodes: {works['count']}")
    print(f"Notable works:    {works['notable_count']} ({works['notable_pct']:.2f}%)")

    print("\nYear Coverage")
    print("-------------")
    for key in ("release_date", "written_date", "notoriety_date"):
        item = year_ranges[key]
        print(
            f"{key:14s} min={item['min']}, max={item['max']}, count_with_value={item['count_with_value']}"
        )


def _plot_bar(
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    values: Dict[str, int],
    color: str = "#4f46e5",
) -> None:
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    labels = list(values.keys())
    counts = list(values.values())
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, counts, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_node_edge_combined(
    output_path: Path,
    node_values: Dict[str, int],
    edge_values: Dict[str, int],
) -> None:
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    if not node_values and not edge_values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    node_labels = list(node_values.keys())
    node_counts = list(node_values.values())
    axes[0].bar(node_labels, node_counts, color="#22c55e")
    axes[0].set_title("Node Type Distribution")
    axes[0].set_xlabel("Node Type")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    edge_labels = list(edge_values.keys())
    edge_counts = list(edge_values.values())
    axes[1].bar(edge_labels, edge_counts, color="#f59e0b")
    axes[1].set_title("Edge Type Distribution")
    axes[1].set_xlabel("Edge Type")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_year_hist(
    output_path: Path,
    title: str,
    year_counts: Dict[Any, int],
    color: str = "#0ea5e9",
) -> None:
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    if not year_counts:
        return
    sorted_items = sorted(
        ((int(year), count) for year, count in year_counts.items()),
        key=lambda x: x[0],
    )
    years = [y for y, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(years, counts, width=0.9, color=color)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_plots(stats: Dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dists = stats["distributions"]
    year_dist = stats["year_distributions"]

    paths: list[Path] = []

    combined_path = output_dir / "node_edge_type_distribution.png"
    _plot_node_edge_combined(
        output_path=combined_path,
        node_values=dists["node_types"],
        edge_values=dists["edge_types"],
    )
    paths.append(combined_path)

    genre_path = output_dir / "top_genres.png"
    _plot_bar(
        output_path=genre_path,
        title="Top Work Genres (Song/Album)",
        x_label="Genre",
        y_label="Count",
        values=dists["genres_top_n"],
        color="#a855f7",
    )
    paths.append(genre_path)

    release_path = output_dir / "release_year_distribution.png"
    _plot_year_hist(
        output_path=release_path,
        title="Release Year Distribution (Song/Album)",
        year_counts=year_dist["release_date"],
        color="#06b6d4",
    )
    paths.append(release_path)

    written_path = output_dir / "written_year_distribution.png"
    _plot_year_hist(
        output_path=written_path,
        title="Written Year Distribution (Song/Album)",
        year_counts=year_dist["written_date"],
        color="#0ea5e9",
    )
    paths.append(written_path)

    notoriety_path = output_dir / "notoriety_year_distribution.png"
    _plot_year_hist(
        output_path=notoriety_path,
        title="Notoriety Year Distribution (Song/Album)",
        year_counts=year_dist["notoriety_date"],
        color="#14b8a6",
    )
    paths.append(notoriety_path)

    return [p for p in paths if p.exists()]


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph_path)
    with graph_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stats = build_stats(data, top_n=args.top_n)
    print_summary(stats, top_n=args.top_n)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSaved JSON stats to: {output_path}")

    if args.plot_dir:
        try:
            plot_paths = save_plots(stats, Path(args.plot_dir))
        except ImportError as exc:
            raise SystemExit(
                "Plot generation requires matplotlib. Install with: pip install matplotlib"
            ) from exc

        if plot_paths:
            print("\nSaved plots:")
            for path in plot_paths:
                print(f"- {path}")


if __name__ == "__main__":
    main()
