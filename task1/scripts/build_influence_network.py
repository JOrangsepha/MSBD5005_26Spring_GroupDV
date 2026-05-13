#!/usr/bin/env python3
"""Build D3-ready influence network payload from influence utilities.

Output format:
{
  "meta": {...},
  "nodes": [
    {
      "id": 17255,
      "name": "Sailor Shift",
      "node_type": "Person",
      "role": "center|influencer|influenced",
      "influence_out_count": 48,
      "influence_in_count": 142,
      "size_score": 190
    }
  ],
  "links": [
    {
      "source": 17255,
      "target": 17256,
      "direction": "out",
      "weight": 5,
      "evidence": [...],
      "dominant_kind": "collaboration_work",
      "dominant_edge_type": null
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.influence_utils import InfluenceExtractor


def _dominant_values(evidence: List[Dict[str, Any]]) -> Tuple[str | None, str | None]:
    kind_counts = Counter(ev.get("kind") for ev in evidence if ev.get("kind"))
    edge_type_counts = Counter(ev.get("edge_type") for ev in evidence if ev.get("edge_type"))
    dominant_kind = kind_counts.most_common(1)[0][0] if kind_counts else None
    dominant_edge_type = edge_type_counts.most_common(1)[0][0] if edge_type_counts else None
    return dominant_kind, dominant_edge_type


def _parse_year(value: Any) -> Optional[int]:
    """Extract a 4-digit year from mixed date values."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 3000 else None
    text = str(value)
    match = re.search(r"(\d{4})", text)
    if not match:
        return None
    year = int(match.group(1))
    return year if 1000 <= year <= 3000 else None


def _node_best_year(node: Dict[str, Any]) -> Optional[int]:
    """Return earliest available year among written/release/notoriety fields."""
    years = [
        _parse_year(node.get("written_date")),
        _parse_year(node.get("release_date")),
        _parse_year(node.get("notoriety_date")),
    ]
    valid = [y for y in years if y is not None]
    return min(valid) if valid else None


def _evidence_years(extractor: InfluenceExtractor, evidence: List[Dict[str, Any]]) -> List[int]:
    """Derive years for each influence evidence record from referenced works."""
    years: List[int] = []
    for ev in evidence:
        source_work = extractor.nodes.get(ev.get("source_work_id"), {})
        target_work = extractor.nodes.get(ev.get("target_work_id"), {})
        source_year = _node_best_year(source_work) if source_work else None
        target_year = _node_best_year(target_work) if target_work else None
        year = source_year if source_year is not None else target_year
        if year is not None:
            years.append(year)
    return sorted(years)


def _enrich_evidence_with_work_types(
    extractor: InfluenceExtractor, evidence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach source/target work node types (e.g., Song/Album) to evidence."""
    enriched: List[Dict[str, Any]] = []
    for ev in evidence:
        ev_copy = dict(ev)
        source_work = extractor.nodes.get(ev.get("source_work_id"), {})
        target_work = extractor.nodes.get(ev.get("target_work_id"), {})
        if source_work:
            ev_copy["source_work_type"] = source_work.get("Node Type")
        if target_work:
            ev_copy["target_work_type"] = target_work.get("Node Type")
        enriched.append(ev_copy)
    return enriched


def build_network_payload(extracted: Dict[str, Any], extractor: InfluenceExtractor) -> Dict[str, Any]:
    center = extracted["center"]
    influences = extracted["influences"]
    influenced_by = extracted["influenced_by"]

    nodes_map: Dict[Any, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []

    center_id = center["node_id"]
    nodes_map[center_id] = {
        "id": center_id,
        "name": center["name"],
        "node_type": center["node_type"],
        "role": "center",
        "influence_out_count": len(influences),
        "influence_in_count": len(influenced_by),
        "size_score": len(influences) + len(influenced_by),
    }

    for row in influences:
        node_id = row["node_id"]
        nodes_map.setdefault(
            node_id,
            {
                "id": node_id,
                "name": row["name"],
                "node_type": row["node_type"],
                "role": "influenced",
                "influence_out_count": 0,
                "influence_in_count": 0,
                "size_score": 0,
            },
        )
        nodes_map[node_id]["influence_in_count"] += 1
        nodes_map[node_id]["size_score"] += row["evidence_count"]

        dominant_kind, dominant_edge_type = _dominant_values(row["evidence"])
        enriched_evidence = _enrich_evidence_with_work_types(extractor, row["evidence"])
        years = _evidence_years(extractor, enriched_evidence)
        links.append(
            {
                "source": center_id,
                "target": node_id,
                "direction": "out",
                "weight": row["evidence_count"],
                "evidence": enriched_evidence,
                "dominant_kind": dominant_kind,
                "dominant_edge_type": dominant_edge_type,
                "first_year": years[0] if years else None,
                "last_year": years[-1] if years else None,
                "year_count": len(years),
            }
        )

    for row in influenced_by:
        node_id = row["node_id"]
        if node_id not in nodes_map:
            nodes_map[node_id] = {
                "id": node_id,
                "name": row["name"],
                "node_type": row["node_type"],
                "role": "influencer",
                "influence_out_count": 0,
                "influence_in_count": 0,
                "size_score": 0,
            }
        elif nodes_map[node_id]["role"] == "influenced":
            nodes_map[node_id]["role"] = "both"

        nodes_map[node_id]["influence_out_count"] += 1
        nodes_map[node_id]["size_score"] += row["evidence_count"]

        dominant_kind, dominant_edge_type = _dominant_values(row["evidence"])
        enriched_evidence = _enrich_evidence_with_work_types(extractor, row["evidence"])
        years = _evidence_years(extractor, enriched_evidence)
        links.append(
            {
                "source": node_id,
                "target": center_id,
                "direction": "in",
                "weight": row["evidence_count"],
                "evidence": enriched_evidence,
                "dominant_kind": dominant_kind,
                "dominant_edge_type": dominant_edge_type,
                "first_year": years[0] if years else None,
                "last_year": years[-1] if years else None,
                "year_count": len(years),
            }
        )

    nodes = sorted(nodes_map.values(), key=lambda n: (n["role"] != "center", -n["size_score"], n["name"]))
    links.sort(key=lambda l: (-l["weight"], l["direction"]))
    all_years = [l["first_year"] for l in links if l.get("first_year") is not None]

    return {
        "meta": {
            "center_id": center_id,
            "center_name": center["name"],
            "node_count": len(nodes),
            "link_count": len(links),
            "notes": [
                "One directed link per relationship to center.",
                "direction=out means center influences target.",
                "direction=in means source influences center.",
                "first_year/last_year are inferred from evidence work dates.",
            ],
            "year_range": {
                "min": min(all_years) if all_years else None,
                "max": max(all_years) if all_years else None,
            },
        },
        "nodes": nodes,
        "links": links,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D3-ready network JSON around one entity.")
    parser.add_argument(
        "--graph-path",
        default="data/MC1_graph.json",
        help="Path to MC1 graph JSON (default: data/MC1_graph.json)",
    )
    parser.add_argument(
        "--name",
        default="Sailor Shift",
        help='Target entity name (default: "Sailor Shift")',
    )
    parser.add_argument(
        "--output",
        default="sailor_shift_network.json",
        help="Output JSON file path (default: sailor_shift_network.json at project root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = InfluenceExtractor(Path(args.graph_path))
    extracted = extractor.extract_for_name(args.name)
    payload = build_network_payload(extracted, extractor)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved network JSON: {output_path}")
    print(
        json.dumps(
            {
                "center": payload["meta"]["center_name"],
                "node_count": payload["meta"]["node_count"],
                "link_count": payload["meta"]["link_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
