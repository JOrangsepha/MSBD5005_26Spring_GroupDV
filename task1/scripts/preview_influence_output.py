#!/usr/bin/env python3
"""Preview expected output format from src/influence_utils.py.

Examples:
    python3 scripts/preview_influence_output.py
    python3 scripts/preview_influence_output.py --name "Sailor Shift" --full
    python3 scripts/preview_influence_output.py --output /tmp/sailor_output.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.influence_utils import InfluenceExtractor


def _truncate_entries(entries: List[Dict[str, Any]], top_n: int, evidence_n: int) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for item in entries[:top_n]:
        preview.append(
            {
                "node_id": item["node_id"],
                "name": item["name"],
                "node_type": item["node_type"],
                "evidence_count": item["evidence_count"],
                "evidence": item["evidence"][:evidence_n],
            }
        )
    return preview


def build_preview_payload(full_payload: Dict[str, Any], top_n: int, evidence_n: int) -> Dict[str, Any]:
    return {
        "format_notes": {
            "center": "Target entity node metadata.",
            "influences": "Nodes influenced by center; sorted by evidence_count descending.",
            "influenced_by": "Nodes that influenced center; sorted by evidence_count descending.",
            "evidence_kinds": [
                "work_reference",
                "collaboration_work",
                "collaboration_group",
            ],
        },
        "preview_counts": {
            "influences_total": len(full_payload["influences"]),
            "influenced_by_total": len(full_payload["influenced_by"]),
        },
        "sample": {
            "center": full_payload["center"],
            "influences": _truncate_entries(full_payload["influences"], top_n=top_n, evidence_n=evidence_n),
            "influenced_by": _truncate_entries(full_payload["influenced_by"], top_n=top_n, evidence_n=evidence_n),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview/output influence extraction JSON format."
    )
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
        "--full",
        action="store_true",
        help="Print full extraction output instead of schema preview + sample.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top influences/influenced_by rows to show in sample preview (default: 5).",
    )
    parser.add_argument(
        "--evidence-n",
        type=int,
        default=2,
        help="Number of evidence entries shown per sample row (default: 2).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional file path to save JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph_path)
    extractor = InfluenceExtractor(graph_path)
    full_payload = extractor.extract_for_name(args.name)

    payload = (
        full_payload
        if args.full
        else build_preview_payload(full_payload, top_n=args.top_n, evidence_n=args.evidence_n)
    )

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
