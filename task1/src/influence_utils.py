"""Utilities to extract influence/influenced relationships from MC1 graph data.

This module follows the project's influence rules:
1) A influences B if B created a work that references A's work.
2) "Influenced by" is the reverse relation.
3) Collaboration also counts as influence:
   - co-creators of the same work influence each other;
   - members of the same musical group influence each other.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

# Edge types used to identify creators of songs/albums.
CREATOR_EDGE_TYPES = {"PerformerOf", "ComposerOf", "ProducerOf", "LyricistOf"}

# Edge types that encode one work drawing from another work.
WORK_INFLUENCE_EDGE_TYPES = {
    "InStyleOf",
    "InterpolatesFrom",
    "CoverOf",
    "LyricalReferenceTo",
    "DirectlySamples",
}


@dataclass(frozen=True)
class InfluenceEvidence:
    """Why one entity influences another."""

    kind: str
    source_work_id: Optional[Any] = None
    source_work_name: Optional[str] = None
    target_work_id: Optional[Any] = None
    target_work_name: Optional[str] = None
    edge_type: Optional[str] = None
    group_id: Optional[Any] = None
    group_name: Optional[str] = None


class InfluenceExtractor:
    """Extracts influence/influenced links around an entity from MC1 graph JSON."""

    def __init__(self, graph_path: str | Path) -> None:
        """Initialize extractor and pre-index graph data.

        Args:
            graph_path: Path to node-link JSON graph (e.g., data/MC1_graph.json).

        Side effects:
            - Loads full graph JSON into memory.
            - Builds `nodes`, `links`, and lightweight in/out edge indexes for reuse.
        """
        self.graph_path = Path(graph_path)
        self.data = self._load_graph(self.graph_path)
        self.nodes: Dict[Any, Dict[str, Any]] = {
            node["id"]: node for node in self.data.get("nodes", [])
        }
        self.links: List[Dict[str, Any]] = self.data.get("links", [])
        self.out_edges: DefaultDict[Any, List[Dict[str, Any]]] = defaultdict(list)
        self.in_edges: DefaultDict[Any, List[Dict[str, Any]]] = defaultdict(list)
        self._build_edge_indexes()

    @staticmethod
    def _load_graph(graph_path: Path) -> Dict[str, Any]:
        """Read graph JSON file and return parsed dictionary."""
        with graph_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _edge_type(link: Dict[str, Any]) -> str:
        """Return edge type in a schema-tolerant way.

        MC1 data uses `Edge Type`, but this helper also accepts `edge_type` or
        `type` so methods remain robust if input format is slightly different.
        """
        return (
            link.get("Edge Type")
            or link.get("edge_type")
            or link.get("type")
            or ""
        )

    def _build_edge_indexes(self) -> None:
        """Build source->edges and target->edges indexes from raw links."""
        for link in self.links:
            src = link.get("source")
            dst = link.get("target")
            if src is None or dst is None:
                continue
            self.out_edges[src].append(link)
            self.in_edges[dst].append(link)

    def find_node_ids(self, name: str, node_types: Optional[Iterable[str]] = None) -> List[Any]:
        """Find candidate node IDs by exact, case-insensitive alias match.

        Args:
            name: Value compared against `name` and `stage_name`.
            node_types: Optional whitelist of node types (e.g., {"Person"}).

        Returns:
            A list of matching node IDs. More than one ID can be returned when
            names are duplicated in the graph.
        """
        needle = name.strip().lower()
        allowed = set(node_types) if node_types else None
        ids: List[Any] = []

        for node_id, node in self.nodes.items():
            ntype = node.get("Node Type")
            if allowed and ntype not in allowed:
                continue
            aliases = [
                str(node.get("name", "")).strip().lower(),
                str(node.get("stage_name", "")).strip().lower(),
            ]
            if needle and needle in aliases:
                ids.append(node_id)
        return ids

    def get_node_label(self, node_id: Any) -> str:
        """Return display label for node.

        Preference order is `stage_name`, then `name`, then raw node ID.
        """
        node = self.nodes.get(node_id, {})
        return node.get("stage_name") or node.get("name") or str(node_id)

    def _build_creator_maps(self) -> Tuple[DefaultDict[Any, Set[Any]], DefaultDict[Any, Set[Any]]]:
        """Build creator/work lookup maps from authoring edges.

        Uses `CREATOR_EDGE_TYPES` (PerformerOf, ComposerOf, ProducerOf, LyricistOf)
        and only keeps valid creator->(Song|Album) relationships.

        Returns:
            Tuple containing:
            - work_to_creators: work_id -> set of creator IDs
            - creator_to_works: creator_id -> set of work IDs
        """
        work_to_creators: DefaultDict[Any, Set[Any]] = defaultdict(set)
        creator_to_works: DefaultDict[Any, Set[Any]] = defaultdict(set)

        for link in self.links:
            edge_type = self._edge_type(link)
            if edge_type not in CREATOR_EDGE_TYPES:
                continue

            creator_id = link.get("source")
            work_id = link.get("target")
            if creator_id not in self.nodes or work_id not in self.nodes:
                continue

            creator_type = self.nodes[creator_id].get("Node Type")
            work_type = self.nodes[work_id].get("Node Type")
            if creator_type not in {"Person", "MusicalGroup", "RecordLabel"}:
                continue
            if work_type not in {"Song", "Album"}:
                continue

            work_to_creators[work_id].add(creator_id)
            creator_to_works[creator_id].add(work_id)

        return work_to_creators, creator_to_works

    def _build_group_membership_map(self) -> DefaultDict[Any, Set[Any]]:
        """Build group membership map from `MemberOf` edges.

        Returns:
            Mapping of musical_group_id -> set(person_id), filtered so source is
            `Person` and target is `MusicalGroup`.
        """
        group_members: DefaultDict[Any, Set[Any]] = defaultdict(set)
        for link in self.links:
            if self._edge_type(link) != "MemberOf":
                continue
            person_id = link.get("source")
            group_id = link.get("target")
            if person_id not in self.nodes or group_id not in self.nodes:
                continue
            if self.nodes[person_id].get("Node Type") != "Person":
                continue
            if self.nodes[group_id].get("Node Type") != "MusicalGroup":
                continue
            group_members[group_id].add(person_id)
        return group_members

    def build_influence_graph(self) -> Dict[Any, Dict[str, List[InfluenceEvidence]]]:
        """Build influence adjacency with evidence.

        This method is the core rule engine and applies AGENTS.md theory:
        - Rule 1/2: work references induce directional influence.
        - Rule 3a: co-creators on the same work influence each other.
        - Rule 3b: members of the same musical group influence each other.

        Output shape:
            {
              influencer_id: {
                influenced_id: [InfluenceEvidence, ...]
              }
            }
        """
        work_to_creators, _ = self._build_creator_maps()
        group_members = self._build_group_membership_map()

        influence_adj: Dict[Any, Dict[str, List[InfluenceEvidence]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Rule 1: A influences B when B's work references A's work.
        for link in self.links:
            edge_type = self._edge_type(link)
            if edge_type not in WORK_INFLUENCE_EDGE_TYPES:
                continue

            source_work = link.get("source")  # newer/referencing work
            target_work = link.get("target")  # earlier/referenced work
            if source_work not in self.nodes or target_work not in self.nodes:
                continue

            source_creators = work_to_creators.get(source_work, set())
            target_creators = work_to_creators.get(target_work, set())

            for influencer in target_creators:
                for influenced in source_creators:
                    if influencer == influenced:
                        continue
                    influence_adj[influencer][influenced].append(
                        InfluenceEvidence(
                            kind="work_reference",
                            source_work_id=source_work,
                            source_work_name=self.get_node_label(source_work),
                            target_work_id=target_work,
                            target_work_name=self.get_node_label(target_work),
                            edge_type=edge_type,
                        )
                    )

        # Rule 3a: Co-creators on same work influence each other (bidirectional).
        for work_id, creators in work_to_creators.items():
            creators_list = list(creators)
            for i, a in enumerate(creators_list):
                for b in creators_list[i + 1 :]:
                    if a == b:
                        continue
                    ev = InfluenceEvidence(
                        kind="collaboration_work",
                        source_work_id=work_id,
                        source_work_name=self.get_node_label(work_id),
                    )
                    influence_adj[a][b].append(ev)
                    influence_adj[b][a].append(ev)

        # Rule 3b: Same-group members influence each other (bidirectional).
        for group_id, members in group_members.items():
            members_list = list(members)
            for i, a in enumerate(members_list):
                for b in members_list[i + 1 :]:
                    ev = InfluenceEvidence(
                        kind="collaboration_group",
                        group_id=group_id,
                        group_name=self.get_node_label(group_id),
                    )
                    influence_adj[a][b].append(ev)
                    influence_adj[b][a].append(ev)

        return influence_adj

    def extract_around_entity(self, entity_id: Any) -> Dict[str, Any]:
        """Extract directional neighborhood around one center entity.

        Args:
            entity_id: Node ID for the center person/group.

        Returns:
            JSON-serializable dict with:
            - center: metadata for center node
            - influences: entities influenced by center (outgoing)
            - influenced_by: entities that influence center (incoming)

            Each row includes `evidence_count` and full evidence records.
        """
        influence_adj = self.build_influence_graph()

        influences: List[Dict[str, Any]] = []
        for influenced_id, evidence in influence_adj.get(entity_id, {}).items():
            influences.append(
                {
                    "node_id": influenced_id,
                    "name": self.get_node_label(influenced_id),
                    "node_type": self.nodes.get(influenced_id, {}).get("Node Type"),
                    "evidence_count": len(evidence),
                    "evidence": [e.__dict__ for e in evidence],
                }
            )

        influenced_by_map: DefaultDict[Any, List[InfluenceEvidence]] = defaultdict(list)
        for influencer_id, influenced_map in influence_adj.items():
            if entity_id in influenced_map:
                influenced_by_map[influencer_id].extend(influenced_map[entity_id])

        influenced_by: List[Dict[str, Any]] = []
        for influencer_id, evidence in influenced_by_map.items():
            influenced_by.append(
                {
                    "node_id": influencer_id,
                    "name": self.get_node_label(influencer_id),
                    "node_type": self.nodes.get(influencer_id, {}).get("Node Type"),
                    "evidence_count": len(evidence),
                    "evidence": [e.__dict__ for e in evidence],
                }
            )

        influences.sort(key=lambda x: x["evidence_count"], reverse=True)
        influenced_by.sort(key=lambda x: x["evidence_count"], reverse=True)

        return {
            "center": {
                "node_id": entity_id,
                "name": self.get_node_label(entity_id),
                "node_type": self.nodes.get(entity_id, {}).get("Node Type"),
            },
            "influences": influences,
            "influenced_by": influenced_by,
        }

    def extract_for_name(self, name: str) -> Dict[str, Any]:
        """Resolve a Person/MusicalGroup by name, then extract its neighborhood.

        Raises:
            ValueError: if no match or multiple matches are found.
        """
        ids = self.find_node_ids(name, node_types={"Person", "MusicalGroup"})
        if not ids:
            raise ValueError(f"No Person/MusicalGroup found with name: {name}")
        if len(ids) > 1:
            options = ", ".join(f"{nid}:{self.get_node_label(nid)}" for nid in ids)
            raise ValueError(f"Multiple matches for '{name}': {options}")
        return self.extract_around_entity(ids[0])


def extract_sailor_shift_influence(graph_path: str | Path) -> Dict[str, Any]:
    """Shortcut helper for the project's default center node: Sailor Shift."""
    extractor = InfluenceExtractor(graph_path)
    return extractor.extract_for_name("Sailor Shift")
