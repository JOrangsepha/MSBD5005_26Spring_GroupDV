"""
Rewrite v2: Oceanus Folk Rising Star Pipeline

Data-driven comparison selection, OF-focused scoring with forward trend,
rank-percentile normalization. All influence/momentum computed from OF songs only.

Outputs: viz/data/artists.json, viz/data/timeline.json, viz/data/network.json
"""

import json, math, os, random, statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

with open(os.path.join(BASE, "MC1_graph.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

id2node = {n["id"]: n for n in data["nodes"]}

INFLUENCE_TYPES = {"InStyleOf", "InterpolatesFrom", "CoverOf", "DirectlySamples", "LyricalReferenceTo"}
CREATIVE_ROLES = {"PerformerOf", "ComposerOf", "LyricistOf", "ProducerOf"}

of_song_ids = {n["id"] for n in data["nodes"] if n.get("Node Type") == "Song" and n.get("genre") == "Oceanus Folk"}

# ---------- graph traversal ----------
entity_songs = defaultdict(set)
entity_roles = defaultdict(lambda: defaultdict(set))
song_to_entities = defaultdict(set)
album_to_entities = defaultdict(set)
group_members = defaultdict(list)
member_of_group = defaultdict(set)

for e in data["links"]:
    et = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if et in CREATIVE_ROLES and src.get("Node Type") in ("Person", "MusicalGroup"):
        if tgt.get("Node Type") == "Song":
            entity_songs[e["source"]].add(e["target"])
            entity_roles[e["source"]][et].add(e["target"])
            song_to_entities[e["target"]].add(e["source"])
        elif tgt.get("Node Type") == "Album":
            album_to_entities[e["target"]].add(e["source"])
    if et == "MemberOf":
        group_members[e["target"]].append(e["source"])
        member_of_group[e["source"]].add(e["target"])

song_labels = defaultdict(set)
entity_labels = defaultdict(set)
for e in data["links"]:
    et = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if et in ("RecordedBy", "DistributedBy") and src.get("Node Type") == "Song" and tgt.get("Node Type") == "RecordLabel":
        song_labels[e["source"]].add(tgt.get("name", "?"))
        for eid in song_to_entities.get(e["source"], set()):
            entity_labels[eid].add(tgt.get("name", "?"))

def get_entities_for_node(nid):
    n = id2node.get(nid, {})
    nt = n.get("Node Type")
    if nt in ("Person", "MusicalGroup"):
        return {nid}
    elif nt == "Song":
        return song_to_entities.get(nid, set())
    elif nt == "Album":
        return album_to_entities.get(nid, set())
    return set()


# Primary artist for influence network: one entity per graph node (raw-edge attribution)
ROLE_ORDER = ("PerformerOf", "ComposerOf", "LyricistOf", "ProducerOf")
song_primary_eid = {}
for e in data["links"]:
    et = e.get("Edge Type", "")
    if et not in CREATIVE_ROLES:
        continue
    src, tgt = e["source"], e["target"]
    src_n = id2node.get(src, {})
    tgt_n = id2node.get(tgt, {})
    if src_n.get("Node Type") not in ("Person", "MusicalGroup") or tgt_n.get("Node Type") != "Song":
        continue
    if et not in ROLE_ORDER:
        continue
    pri = ROLE_ORDER.index(et)
    sid = tgt
    cur = song_primary_eid.get(sid)
    if cur is None or pri < cur[0] or (pri == cur[0] and src < cur[1]):
        song_primary_eid[sid] = (pri, src)


def primary_entity_for_graph_node(nid):
    n = id2node.get(nid)
    if not n:
        return None
    nt = n.get("Node Type")
    if nt in ("Person", "MusicalGroup"):
        return nid
    if nt == "Song":
        t = song_primary_eid.get(nid)
        return t[1] if t else None
    if nt == "Album":
        ents = album_to_entities.get(nid, set())
        return min(ents) if ents else None
    return None


song_inf_out = defaultdict(int)
song_inf_in = defaultdict(int)
influence_edges = []
for e in data["links"]:
    if e.get("Edge Type") in INFLUENCE_TYPES:
        sid, tid = e["source"], e["target"]
        song_inf_out[sid] += 1
        song_inf_in[tid] += 1
        src_ents = get_entities_for_node(sid)
        tgt_ents = get_entities_for_node(tid)
        if src_ents and tgt_ents:
            influence_edges.append({
                "source_song": sid, "target_song": tid,
                "source_entities": list(src_ents), "target_entities": list(tgt_ents),
                "source_genre": id2node.get(sid, {}).get("genre", ""),
                "target_genre": id2node.get(tid, {}).get("genre", ""),
                "type": e.get("Edge Type", ""),
            })

entity_collaborators = defaultdict(set)
for sid, entities in song_to_entities.items():
    for eid in entities:
        entity_collaborators[eid].update(entities - {eid})


def _qualify_of_out(ps, src_nid):
    if not ps:
        return False
    if src_nid == ps:
        return True
    n = id2node.get(src_nid, {})
    nt = n.get("Node Type")
    if nt == "Song":
        return src_nid in of_song_ids and src_nid in entity_songs.get(ps, set())
    if nt == "Album":
        return ps in album_to_entities.get(src_nid, set())
    return False


def _qualify_all_out(ps, src_nid):
    if not ps:
        return False
    if src_nid == ps:
        return True
    n = id2node.get(src_nid, {})
    nt = n.get("Node Type")
    if nt == "Song":
        return src_nid in entity_songs.get(ps, set())
    if nt == "Album":
        return ps in album_to_entities.get(src_nid, set())
    return False


def _qualify_of_in(pt, tgt_nid):
    if not pt:
        return False
    if tgt_nid == pt:
        return True
    n = id2node.get(tgt_nid, {})
    nt = n.get("Node Type")
    if nt == "Song":
        return tgt_nid in of_song_ids and tgt_nid in entity_songs.get(pt, set())
    if nt == "Album":
        return pt in album_to_entities.get(tgt_nid, set())
    return False


def _qualify_all_in(pt, tgt_nid):
    if not pt:
        return False
    if tgt_nid == pt:
        return True
    n = id2node.get(tgt_nid, {})
    nt = n.get("Node Type")
    if nt == "Song":
        return tgt_nid in entity_songs.get(pt, set())
    if nt == "Album":
        return pt in album_to_entities.get(tgt_nid, set())
    return False


def song_year(sid):
    n = id2node.get(sid, {})
    y = n.get("release_date") or n.get("written_date") or ""
    return int(y) if y and y.isdigit() else None


# Primary-attributed influence totals + yearly (single O(E) pass; matches network.json edges)
of_last_by_eid = {}
all_last_by_eid = {}
for eid, sids in entity_songs.items():
    of_years = sorted(filter(None, (song_year(s) for s in sids & of_song_ids)))
    if of_years:
        of_last_by_eid[eid] = of_years[-1]
    all_years = sorted(filter(None, (song_year(s) for s in sids)))
    if all_years:
        all_last_by_eid[eid] = all_years[-1]

edge_of_out = defaultdict(int)
edge_of_in = defaultdict(int)
edge_all_out = defaultdict(int)
edge_all_in = defaultdict(int)
yearly_of_inf_out = defaultdict(lambda: defaultdict(int))
yearly_of_inf_in = defaultdict(lambda: defaultdict(int))
yearly_all_inf_out = defaultdict(lambda: defaultdict(int))
yearly_all_inf_in = defaultdict(lambda: defaultdict(int))

for e in influence_edges:
    src, tgt = e["source_song"], e["target_song"]
    ps = primary_entity_for_graph_node(src)
    if ps and _qualify_of_out(ps, src):
        edge_of_out[ps] += 1
        n = id2node.get(src, {})
        y = song_year(src) if n.get("Node Type") == "Song" else None
        if y is None and (src == ps or n.get("Node Type") == "Album"):
            y = of_last_by_eid.get(ps)
        if y is not None:
            yearly_of_inf_out[ps][y] += 1
        # Roll member-attributed OF edges up to parent MusicalGroup (timeline + inf totals)
        if id2node.get(ps, {}).get("Node Type") == "Person":
            for gid in member_of_group.get(ps, ()):
                if _qualify_of_out(gid, src):
                    edge_of_out[gid] += 1
                    if y is not None:
                        yearly_of_inf_out[gid][y] += 1
    if ps and _qualify_all_out(ps, src):
        edge_all_out[ps] += 1
        n = id2node.get(src, {})
        y = song_year(src) if n.get("Node Type") == "Song" else None
        if y is None and (src == ps or n.get("Node Type") == "Album"):
            y = all_last_by_eid.get(ps)
        if y is not None:
            yearly_all_inf_out[ps][y] += 1
        if id2node.get(ps, {}).get("Node Type") == "Person":
            for gid in member_of_group.get(ps, ()):
                if _qualify_all_out(gid, src):
                    edge_all_out[gid] += 1
                    if y is not None:
                        yearly_all_inf_out[gid][y] += 1

    pt = primary_entity_for_graph_node(tgt)
    if pt and _qualify_of_in(pt, tgt):
        edge_of_in[pt] += 1
        n = id2node.get(tgt, {})
        y = song_year(tgt) if n.get("Node Type") == "Song" else None
        if y is None and (tgt == pt or n.get("Node Type") == "Album"):
            y = of_last_by_eid.get(pt)
        if y is not None:
            yearly_of_inf_in[pt][y] += 1
        if id2node.get(pt, {}).get("Node Type") == "Person":
            for gid in member_of_group.get(pt, ()):
                if _qualify_of_in(gid, tgt):
                    edge_of_in[gid] += 1
                    if y is not None:
                        yearly_of_inf_in[gid][y] += 1
    if pt and _qualify_all_in(pt, tgt):
        edge_all_in[pt] += 1
        n = id2node.get(tgt, {})
        y = song_year(tgt) if n.get("Node Type") == "Song" else None
        if y is None and (tgt == pt or n.get("Node Type") == "Album"):
            y = all_last_by_eid.get(pt)
        if y is not None:
            yearly_all_inf_in[pt][y] += 1
        if id2node.get(pt, {}).get("Node Type") == "Person":
            for gid in member_of_group.get(pt, ()):
                if _qualify_all_in(gid, tgt):
                    edge_all_in[gid] += 1
                    if y is not None:
                        yearly_all_inf_in[gid][y] += 1


# ---------- per-entity helpers ----------
def build_profile(eid):
    node = id2node[eid]
    sids = entity_songs.get(eid, set())
    if not sids:
        return None

    songs = [id2node[s] for s in sids]
    total = len(sids)
    of_sids = sids & of_song_ids
    of_count = len(of_sids)
    genres = Counter(s.get("genre", "Unknown") for s in songs)

    of_notable = sum(1 for s in of_sids if id2node[s].get("notable"))
    all_notable = sum(1 for s in songs if s.get("notable"))

    of_inf_out = edge_of_out.get(eid, 0)
    of_inf_in = edge_of_in.get(eid, 0)
    all_inf_out = edge_all_out.get(eid, 0)
    all_inf_in = edge_all_in.get(eid, 0)

    of_years_list = sorted(filter(None, (song_year(s) for s in of_sids)))
    all_years_list = sorted(filter(None, (song_year(s) for s in sids)))

    of_first = of_years_list[0] if of_years_list else None
    of_last = of_years_list[-1] if of_years_list else None
    of_span = (of_last - of_first + 1) if of_first and of_last else 0
    all_first = all_years_list[0] if all_years_list else None
    all_last = all_years_list[-1] if all_years_list else None

    of_yearly = defaultdict(lambda: {"songs": 0, "notable": 0, "inf_out": 0, "inf_in": 0})
    for s in of_sids:
        y = song_year(s)
        if y is None:
            continue
        of_yearly[y]["songs"] += 1
        of_yearly[y]["notable"] += 1 if id2node[s].get("notable") else 0
    for y, c in yearly_of_inf_out[eid].items():
        of_yearly[y]["inf_out"] = c
    for y, c in yearly_of_inf_in[eid].items():
        of_yearly[y]["inf_in"] = c
    of_yearly = dict(sorted(of_yearly.items()))

    all_yearly = defaultdict(lambda: {"songs": 0, "notable": 0, "inf_out": 0, "inf_in": 0})
    for s in sids:
        y = song_year(s)
        if y is None:
            continue
        all_yearly[y]["songs"] += 1
        all_yearly[y]["notable"] += 1 if id2node[s].get("notable") else 0
    for y, c in yearly_all_inf_out[eid].items():
        all_yearly[y]["inf_out"] = c
    for y, c in yearly_all_inf_in[eid].items():
        all_yearly[y]["inf_in"] = c
    all_yearly = dict(sorted(all_yearly.items()))

    of_collabs = set()
    for s in of_sids:
        of_collabs.update(song_to_entities.get(s, set()) - {eid})

    of_labels = set()
    for s in of_sids:
        of_labels.update(song_labels.get(s, set()))

    role_div = len(entity_roles.get(eid, {}))

    return {
        "eid": eid,
        "name": node.get("name", f"Entity {eid}"),
        "type": node.get("Node Type", "Person"),
        "members": [id2node[m].get("name", "?") for m in group_members.get(eid, [])]
                   if node.get("Node Type") == "MusicalGroup" else [],
        "total_songs": total,
        "of_songs": of_count,
        "of_notable": of_notable,
        "of_notable_ratio": round(of_notable / max(of_count, 1), 4),
        "all_notable": all_notable,
        "all_notable_ratio": round(all_notable / max(total, 1), 4),
        "of_commitment": round(of_count / max(total, 1), 4),
        "of_inf_out": of_inf_out,
        "of_inf_in": of_inf_in,
        "all_inf_out": all_inf_out,
        "all_inf_in": all_inf_in,
        "of_first_year": of_first,
        "of_last_year": of_last,
        "of_span": of_span,
        "all_first_year": all_first,
        "all_last_year": all_last,
        "genres": dict(genres.most_common(10)),
        "genre_breadth": len(genres),
        "of_collaborators": len(of_collabs),
        "all_collaborators": len(entity_collaborators.get(eid, set())),
        "of_labels": len(of_labels),
        "all_labels": len(entity_labels.get(eid, set())),
        "role_diversity": role_div,
        "of_yearly": {str(k): v for k, v in of_yearly.items()},
        "all_yearly": {str(k): v for k, v in all_yearly.items()},
    }


# ---------- build all profiles ----------
of_entities = set()
for e in data["links"]:
    if e.get("Edge Type") in CREATIVE_ROLES and e["target"] in of_song_ids:
        src = id2node.get(e["source"], {})
        if src.get("Node Type") in ("Person", "MusicalGroup"):
            of_entities.add(e["source"])

def is_redundant_member(eid):
    for gid in member_of_group.get(eid, set()):
        gs = entity_songs.get(gid, set())
        my = entity_songs.get(eid, set())
        if gs and my and len(my & gs) / max(len(my), 1) > 0.8:
            return True
    return False

print("Finding redundant members...")
skip_ids = set()
persons = [eid for eid in entity_songs
           if id2node[eid].get("Node Type") == "Person"
           and not member_of_group.get(eid)
           and len(entity_songs[eid]) >= 2]
persons.sort()
for i, a in enumerate(persons):
    if a in skip_ids:
        continue
    for b in persons[i + 1:]:
        if b in skip_ids:
            continue
        if entity_songs[a] == entity_songs[b]:
            skip_ids.add(b)

print("Building profiles...")
all_profiles = {}
for eid in of_entities:
    if is_redundant_member(eid) or eid in skip_ids:
        continue
    p = build_profile(eid)
    if p and p["total_songs"] > 0:
        all_profiles[eid] = p

print("Adding extra profiles...")
# Entities on the other end of a raw influence edge from/to an OF artist (primary attribution only)
extra_eids = set()
for e in influence_edges:
    sid, tid = e["source_song"], e["target_song"]
    se = primary_entity_for_graph_node(sid)
    te = primary_entity_for_graph_node(tid)
    if not se or not te:
        continue
    if se in of_entities or te in of_entities:
        extra_eids.add(se)
        extra_eids.add(te)

for eid in extra_eids - of_entities:
    if is_redundant_member(eid) or eid in skip_ids:
        continue
    p = build_profile(eid)
    if p and p["total_songs"] > 0:
        all_profiles[eid] = p

print("Counting names...")
name_counter = Counter(p["name"] for p in all_profiles.values())
for eid, p in all_profiles.items():
    p["display_name"] = p["name"] if name_counter[p["name"]] == 1 else f"{p['name']} (#{eid})"


# ========== 1a: COMPARISON ARTIST SELECTION ==========
def find_entity(name, prefer_type=None):
    cands = [n for n in data["nodes"] if n.get("name") == name]
    if prefer_type:
        typed = [n for n in cands if n.get("Node Type") == prefer_type]
        if typed:
            cands = typed
    best_id, best_count = None, -1
    for c in cands:
        cnt = len(entity_songs.get(c["id"], set()))
        if cnt > best_count:
            best_count = cnt
            best_id = c["id"]
    return best_id

sailor_id = find_entity("Sailor Shift")
ivy_id = find_entity("Ivy Echos", "MusicalGroup")
exclude_from_comparison = {sailor_id, ivy_id}
if ivy_id:
    for m in group_members.get(ivy_id, []):
        exclude_from_comparison.add(m)

comp_eligible = []
for eid, p in all_profiles.items():
    if eid in exclude_from_comparison:
        continue
    if p["of_songs"] < 3:
        continue
    if p["of_span"] < 2 and p["of_songs"] < 5:
        continue
    if p["of_inf_in"] <= 0 and p["of_inf_out"] <= 2:
        continue
    comp_eligible.append((eid, p))

influence_magnet = max(comp_eligible, key=lambda x: x[1]["of_inf_in"], default=None)

rapid_candidates = [(eid, p) for eid, p in comp_eligible
                    if p["of_notable_ratio"] > 0.5
                    and eid != (influence_magnet[0] if influence_magnet else None)]
for eid, p in rapid_candidates:
    p["_velocity"] = p["of_songs"] / max(p["of_span"], 1)
rapid_riser = max(rapid_candidates, key=lambda x: x[1]["_velocity"], default=None)

COMPARISON_IDS = {"Sailor Shift": sailor_id}
if influence_magnet:
    COMPARISON_IDS[all_profiles[influence_magnet[0]]["display_name"]] = influence_magnet[0]
if rapid_riser:
    COMPARISON_IDS[all_profiles[rapid_riser[0]]["display_name"]] = rapid_riser[0]

for _, eid in COMPARISON_IDS.items():
    if eid and eid not in all_profiles:
        p = build_profile(eid)
        if p and p["total_songs"] > 0:
            all_profiles[eid] = p

comparison_names = []
archetype_map = {}
for name, eid in COMPARISON_IDS.items():
    if eid and eid in all_profiles:
        dname = all_profiles[eid]["display_name"]
        comparison_names.append(dname)
        if eid == sailor_id:
            archetype_map[dname] = "Sustained Producer"
        elif influence_magnet and eid == influence_magnet[0]:
            archetype_map[dname] = "Influence Magnet"
        elif rapid_riser and eid == rapid_riser[0]:
            archetype_map[dname] = "Rapid Riser"

print(f"Comparison artists: {comparison_names}")
print(f"Archetypes: {archetype_map}")


# ========== 1b: CAREER SERIES (popularity + influence signals) ==========
def build_career_series(profile):
    yearly = profile["of_yearly"]
    years = sorted(int(y) for y in yearly if str(y).isdigit())
    if not years:
        return []
    cum_pop, cum_inf, cum_notable = 0.0, 0, 0
    series = []
    for y in years:
        d = yearly.get(str(y), {})
        s = d.get("songs", 0)
        n = d.get("notable", 0)
        io = d.get("inf_out", 0)
        ii = d.get("inf_in", 0)
        cum_pop += s + 0.5 * n
        cum_inf += io + ii
        cum_notable += n
        series.append({
            "year": y,
            "songs": s, "notable": n, "inf_out": io, "inf_in": ii,
            "cum_pop": round(cum_pop, 2),
            "cum_inf": cum_inf,
            "cum_notable": cum_notable,
        })
    return series

def compute_milestones(profile, series):
    ms = {}
    if profile["of_first_year"]:
        ms["first_of_song"] = profile["of_first_year"]
    for pt in series:
        if "first_notable" not in ms and pt["cum_notable"] > 0:
            ms["first_notable"] = pt["year"]
        if "first_inf_received" not in ms and pt["inf_in"] > 0:
            ms["first_inf_received"] = pt["year"]
    if series:
        peak = max(series, key=lambda p: p["songs"] + p["inf_out"] + p["inf_in"])
        ms["peak_year"] = peak["year"]
    return ms

comparison_series = {}
comparison_milestones = {}
for name in comparison_names:
    p = None
    for eid, prof in all_profiles.items():
        if prof["display_name"] == name:
            p = prof
            break
    if p:
        s = build_career_series(p)
        comparison_series[name] = s
        comparison_milestones[name] = compute_milestones(p, s)


# ========== 1c: PREDICTION MODEL ==========
comp_eids = set(COMPARISON_IDS.values())
exclude_prediction = set(comp_eids)
if ivy_id:
    exclude_prediction.add(ivy_id)
    for m in group_members.get(ivy_id, []):
        exclude_prediction.add(m)

DIMENSIONS = [
    {"key": "of_momentum", "label": "OF Momentum", "weight": 1.5},
    {"key": "quality_signal", "label": "Quality Signal", "weight": 1.3},
    {"key": "influence_reach", "label": "Influence Reach", "weight": 1.4},
    {"key": "genre_bridge", "label": "Genre Bridge", "weight": 0.8},
    {"key": "collab_network", "label": "Collaboration Network", "weight": 0.7},
    {"key": "industry_traction", "label": "Industry Traction", "weight": 0.6},
]

# Determine the latest year in the dataset for recency filtering
_all_of_last = [p["of_last_year"] for p in all_profiles.values() if p.get("of_last_year")]
DATASET_MAX_YEAR = max(_all_of_last) if _all_of_last else 2040
# Rising-star candidates must have been active within the last 15 years of the dataset
RECENCY_HORIZON = 15
RECENCY_CUTOFF_YEAR = DATASET_MAX_YEAR - RECENCY_HORIZON  # 2025 for a 2040 dataset

candidates = []
for eid, p in all_profiles.items():
    if eid in exclude_prediction:
        continue
    if p["of_songs"] < 2:
        continue
    if p["of_commitment"] < 0.30:
        continue
    if (p.get("of_last_year") or 0) < RECENCY_CUTOFF_YEAR:
        continue
    candidates.append((eid, p))

print(f"Prediction candidates (>=30% OF, >=2 songs, active after {RECENCY_CUTOFF_YEAR}): {len(candidates)}")

global_of_years = sorted(filter(None, (p["of_last_year"] for _, p in candidates)))
global_of_min = min(global_of_years) if global_of_years else 1993
global_of_max = max(global_of_years) if global_of_years else 2040
global_of_range = max(global_of_max - global_of_min, 1)

def compute_raw_dims(p):
    oy = p["of_yearly"]
    years = sorted(int(y) for y in oy if str(y).isdigit())

    # OF Momentum: recent 3 years vs prior
    if len(years) >= 2:
        cutoff = years[-1] - 2
        recent = sum(oy[str(y)].get("songs", 0) + 0.35 * (oy[str(y)].get("inf_out", 0) + oy[str(y)].get("inf_in", 0))
                     for y in years if y >= cutoff)
        prior = sum(oy[str(y)].get("songs", 0) + 0.35 * (oy[str(y)].get("inf_out", 0) + oy[str(y)].get("inf_in", 0))
                    for y in years if y < cutoff)
        momentum = max(0.0, recent - prior * 0.5)
    else:
        momentum = sum(oy[str(y)].get("songs", 0) for y in years) if years else 0.0

    quality = p["of_notable_ratio"]
    influence = p["of_inf_out"] + 0.8 * p["of_inf_in"]

    genre_counts = p["genres"]
    total_g = sum(genre_counts.values())
    probs = [c / total_g for c in genre_counts.values() if c > 0]
    if len(probs) > 1:
        entropy = -sum(pp * math.log(pp) for pp in probs)
        genre_bridge = entropy / math.log(len(probs))
    else:
        genre_bridge = 0.0

    collab = p["of_collaborators"] * (1.0 + 0.2 * p["role_diversity"])
    industry = p["of_labels"] + 0.3 * p["role_diversity"]

    # Acceleration: output in last half / first half
    if len(years) >= 2:
        mid = years[len(years) // 2]
        first_half = sum(oy[str(y)].get("songs", 0) for y in years if y <= mid) or 0.5
        last_half = sum(oy[str(y)].get("songs", 0) for y in years if y > mid) or 0
        acceleration = min(1.0, max(0.0, last_half / first_half))
    else:
        acceleration = 0.5

    recency_raw = 0.0
    if p["of_last_year"]:
        recency_raw = (p["of_last_year"] - global_of_min) / global_of_range

    # Career breadth: penalize one-year spikes (e.g. high trend from recency alone)
    years_with_activity = sum(
        1 for y, v in oy.items()
        if str(y).isdigit() and (v.get("songs", 0) > 0 or v.get("inf_out", 0) > 0 or v.get("inf_in", 0) > 0)
    )
    span_y = max(p["of_span"] or 0, 1)
    activity_breadth_raw = min(1.0, years_with_activity / 5.0) * min(1.0, span_y / 10.0)
    if years_with_activity <= 1:
        activity_breadth_raw *= 0.25

    return {
        "of_momentum": round(momentum, 4),
        "quality_signal": round(quality, 4),
        "influence_reach": round(influence, 4),
        "genre_bridge": round(genre_bridge, 4),
        "collab_network": round(collab, 4),
        "industry_traction": round(industry, 4),
        "acceleration": round(acceleration, 4),
        "recency_raw": round(recency_raw, 4),
        "activity_breadth_raw": round(activity_breadth_raw, 4),
    }

raw_rows = []
for eid, p in candidates:
    dims = compute_raw_dims(p)
    raw_rows.append({"eid": eid, "profile": p, "dims": dims, "is_candidate": True})

# Include comparison artists in the percentile pool so their dim_pct matches PCP / radar axes
comp_eids_for_pool = set(COMPARISON_IDS.values())
for eid in comp_eids_for_pool:
    if eid not in all_profiles:
        continue
    if any(r["eid"] == eid for r in raw_rows):
        continue
    p = all_profiles[eid]
    raw_rows.append({"eid": eid, "profile": p, "dims": compute_raw_dims(p), "is_candidate": False})


def rank_percentile(values):
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    result = [0.0] * n
    for rank, (idx, _) in enumerate(indexed):
        result[idx] = rank / max(n - 1, 1)
    return result


def percentile_value_in_pool(v, pool):
    """Map a scalar into ~[0,1] using the same spirit as rank_percentile, vs candidate pool only."""
    n = len(pool)
    if n == 0:
        return 0.5
    s = sorted(pool)
    lo = bisect_left(s, v)
    hi = bisect_right(s, v)
    return ((lo + hi) / 2) / max(n - 1, 1)


def trend_index_geom_of_signals(momentum_pct, recency_pct, accel, breadth_pct):
    """Scalar trend index before pool percentile: weighted geometric mean of four 0–1 legs.

    Interprets forward potential as multiplicative evidence (Cobb–Douglas style): weak
    recency, output acceleration, or career breadth pulls the index down together with
    momentum—not a separate “demo” additive layer."""
    m = max(float(momentum_pct), 1e-9)
    r = max(float(recency_pct), 1e-9)
    a = max(float(accel), 0.02)
    b = max(float(breadth_pct), 1e-9)
    return math.exp(
        0.36 * math.log(m)
        + 0.24 * math.log(r)
        + 0.12 * math.log(a)
        + 0.28 * math.log(b)
    )


def rising_star_composite(wds_pct, trend_pct, geo_pct, cons_pct, breadth_pct,
                          years_inactive=0):
    """Final score in [0,1]: nested multiplicative structure (not a flat weighted sum).

    1) Joint headroom √(w·t): dimension strength and forward trend must co-occur in the pool.
    2) One-sided damp (min/max)^α: a flash on one axis cannot fully substitute for the other.
    3) Profile coherence √(geo·cons): balanced OF fingerprint as a multiplier on that core.
    4) Career credibility √breadth: thin or single-year OF footprints are down-weighted vs sustained arcs.
    5) Recency decay: artists inactive for many years are penalized exponentially."""
    w = max(float(wds_pct), 1e-9)
    t = max(float(trend_pct), 1e-9)
    joint = math.sqrt(w * t)
    mx, mn = max(w, t), min(w, t)
    balance = mn / mx
    core = joint * (balance ** 0.40)
    shape = math.sqrt(max(float(geo_pct) * float(cons_pct), 1e-9))
    cadence = math.sqrt(max(float(breadth_pct), 0.02))
    mod_shape = 0.48 + 0.52 * shape
    mod_cadence = 0.52 + 0.48 * cadence
    # Exponential recency decay: ~7% penalty per year of inactivity
    recency_decay = math.exp(-0.07 * max(years_inactive, 0))
    return min(1.0, core * mod_shape * mod_cadence * recency_decay)


dim_keys = [d["key"] for d in DIMENSIONS] + ["recency_raw", "activity_breadth_raw"]
percentiles = {}
for k in dim_keys:
    vals = [r["dims"][k] for r in raw_rows]
    percentiles[k] = rank_percentile(vals)

ranked = []
for i, row in enumerate(raw_rows):
    if not row["is_candidate"]:
        continue
    dim_pct = {}
    dim_contrib = {}
    weighted_sum = 0.0
    for d in DIMENSIONS:
        k = d["key"]
        pct = percentiles[k][i]
        dim_pct[k] = round(pct, 4)
        contrib = pct * d["weight"]
        dim_contrib[k] = round(contrib, 4)
        weighted_sum += contrib

    momentum_pct = dim_pct.get("of_momentum", 0)
    recency_pct = percentiles["recency_raw"][i]
    accel = row["dims"]["acceleration"]
    breadth_sig = percentiles["activity_breadth_raw"][i]
    trend_raw = trend_index_geom_of_signals(momentum_pct, recency_pct, accel, breadth_sig)

    # Geometric mean of dim percentiles — penalizes zero in any dimension
    pct_vals = [max(dim_pct[d["key"]], 0.01) for d in DIMENSIONS]
    geo_mean = math.exp(sum(math.log(v) for v in pct_vals) / len(pct_vals))

    # Consistency: 1 - normalized stdev (lower variance = more balanced)
    if len(pct_vals) > 1:
        consistency = 1.0 - min(1.0, statistics.stdev(pct_vals) / 0.4)
    else:
        consistency = 0.5

    ranked.append({
        "eid": row["eid"],
        "profile": row["profile"],
        "name": row["profile"]["display_name"],
        "weighted_dim_sum": round(weighted_sum, 4),
        "trend_raw": round(trend_raw, 4),
        "geo_mean": round(geo_mean, 4),
        "consistency": round(consistency, 4),
        "dim_pct": dim_pct,
        "dim_contrib": dim_contrib,
        "dims_raw": row["dims"],
        "selected": False,
        "rejection_reason": None,
    })

# Normalize trend_raw, geo_mean, consistency to rank percentile
trend_pcts = rank_percentile([r["trend_raw"] for r in ranked])
geo_pcts = rank_percentile([r["geo_mean"] for r in ranked])
cons_pcts = rank_percentile([r["consistency"] for r in ranked])
wds_pcts = rank_percentile([r["weighted_dim_sum"] for r in ranked])

total_weight = sum(d["weight"] for d in DIMENSIONS)
for i, r in enumerate(ranked):
    r["trend_pct"] = round(trend_pcts[i], 4)
    r["geo_pct"] = round(geo_pcts[i], 4)
    r["cons_pct"] = round(cons_pcts[i], 4)
    r["wds_pct"] = round(wds_pcts[i], 4)
    r["breadth_pct"] = round(percentiles["activity_breadth_raw"][i], 4)
    inactive = DATASET_MAX_YEAR - (r["profile"].get("of_last_year") or DATASET_MAX_YEAR)
    final_score = rising_star_composite(
        r["wds_pct"],
        r["trend_pct"],
        r["geo_pct"],
        r["cons_pct"],
        r["breadth_pct"],
        years_inactive=inactive,
    )
    r["score"] = round(final_score, 4)
    r["trend_score"] = round(r["trend_pct"], 4)

pool_tr = [r["trend_raw"] for r in ranked]
pool_gm = [r["geo_mean"] for r in ranked]
pool_cs = [r["consistency"] for r in ranked]
pool_wds = [r["weighted_dim_sum"] for r in ranked]

comparison_scores = {}
for i, row in enumerate(raw_rows):
    if row["is_candidate"]:
        continue
    p = row["profile"]
    dim_pct = {}
    dim_contrib = {}
    weighted_sum = 0.0
    for d in DIMENSIONS:
        k = d["key"]
        pct = percentiles[k][i]
        dim_pct[k] = round(pct, 4)
        contrib = pct * d["weight"]
        dim_contrib[k] = round(contrib, 4)
        weighted_sum += contrib
    momentum_pct = dim_pct.get("of_momentum", 0)
    recency_pct = percentiles["recency_raw"][i]
    accel = row["dims"]["acceleration"]
    breadth_sig = percentiles["activity_breadth_raw"][i]
    trend_raw = trend_index_geom_of_signals(momentum_pct, recency_pct, accel, breadth_sig)
    pct_vals = [max(dim_pct[d["key"]], 0.01) for d in DIMENSIONS]
    geo_mean = math.exp(sum(math.log(v) for v in pct_vals) / len(pct_vals))
    if len(pct_vals) > 1:
        consistency = 1.0 - min(1.0, statistics.stdev(pct_vals) / 0.4)
    else:
        consistency = 0.5
    trend_pct = round(percentile_value_in_pool(trend_raw, pool_tr), 4)
    geo_pct = round(percentile_value_in_pool(geo_mean, pool_gm), 4)
    cons_pct = round(percentile_value_in_pool(consistency, pool_cs), 4)
    wds_pct = round(percentile_value_in_pool(weighted_sum, pool_wds), 4)
    breadth_pct_c = round(percentiles["activity_breadth_raw"][i], 4)
    comp_inactive = DATASET_MAX_YEAR - (p.get("of_last_year") or DATASET_MAX_YEAR)
    final_score = rising_star_composite(
        wds_pct, trend_pct, geo_pct, cons_pct, breadth_pct_c,
        years_inactive=comp_inactive,
    )
    name = p["display_name"]
    comparison_scores[name] = {
        "name": name,
        "eid": row["eid"],
        "score": round(final_score, 4),
        "rank": None,
        "weighted_dim_sum": round(weighted_sum, 4),
        "trend_score": trend_pct,
        "trend_raw": round(trend_raw, 4),
        "trend_pct": trend_pct,
        "breadth_pct": breadth_pct_c,
        "geo_mean": round(geo_mean, 4),
        "geo_pct": geo_pct,
        "consistency": round(consistency, 4),
        "cons_pct": cons_pct,
        "wds_pct": wds_pct,
        "of_songs": p["of_songs"],
        "total_songs": p["total_songs"],
        "of_notable_ratio": p["of_notable_ratio"],
        "of_commitment": p["of_commitment"],
        "of_inf_out": p["of_inf_out"],
        "of_inf_in": p["of_inf_in"],
        "dim_pct": dim_pct,
        "dim_contrib": dim_contrib,
        "dims_raw": row["dims"],
    }

ranked.sort(key=lambda x: x["score"], reverse=True)
for i, r in enumerate(ranked):
    r["rank"] = i + 1

# Select top 3 with diversity filter
selected_eids = []
picked_songs = set()
picked_collabs = set()
for item in ranked:
    eid = item["eid"]
    my_songs = entity_songs.get(eid, set())
    if len(selected_eids) >= 3:
        item["rejection_reason"] = "Top 3 already selected"
        continue
    if picked_songs and (my_songs & picked_songs):
        item["rejection_reason"] = "Shares songs with selected"
        continue
    if eid in picked_collabs:
        item["rejection_reason"] = "Direct collaborator of selected"
        continue
    item["selected"] = True
    selected_eids.append(eid)
    picked_songs.update(my_songs)
    picked_collabs.update(entity_collaborators.get(eid, set()))

prediction_names = [all_profiles[eid]["display_name"] for eid in selected_eids if eid in all_profiles]
print(f"Predictions: {prediction_names}")


# ========== EXPORT: artists.json ==========
profiles_export = {}
for eid, p in all_profiles.items():
    pe = dict(p)
    pe.pop("_velocity", None)
    profiles_export[p["display_name"]] = pe

ranking_rows = []
for item in ranked:
    p = item["profile"]
    ranking_rows.append({
        "rank": item["rank"],
        "name": item["name"],
        "eid": item["eid"],
        "score": item["score"],
        "weighted_dim_sum": item["weighted_dim_sum"],
        "trend_score": item["trend_score"],
        "trend_raw": item["trend_raw"],
        "trend_pct": item["trend_pct"],
        "geo_mean": item["geo_mean"],
        "geo_pct": item["geo_pct"],
        "consistency": item["consistency"],
        "cons_pct": item["cons_pct"],
        "wds_pct": item["wds_pct"],
        "breadth_pct": item["breadth_pct"],
        "selected": item["selected"],
        "rejection_reason": item["rejection_reason"],
        "of_songs": p["of_songs"],
        "total_songs": p["total_songs"],
        "of_notable_ratio": p["of_notable_ratio"],
        "of_commitment": p["of_commitment"],
        "of_inf_out": p["of_inf_out"],
        "of_inf_in": p["of_inf_in"],
        "of_first_year": p["of_first_year"],
        "of_last_year": p["of_last_year"],
        "of_span": p["of_span"],
        "dim_pct": item["dim_pct"],
        "dim_contrib": item["dim_contrib"],
        "dims_raw": item["dims_raw"],
    })

# Career series for comparison artists
career_export = {}
for name in comparison_names:
    career_export[name] = {
        "series": comparison_series.get(name, []),
        "milestones": comparison_milestones.get(name, {}),
        "archetype": archetype_map.get(name, ""),
    }

print("Exporting artists.json...")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "artists.json"), "w", encoding="utf-8") as f:
    json.dump({
        "profiles": profiles_export,
        "comparison": comparison_names,
        "predictions": prediction_names,
        "archetypes": archetype_map,
        "career_series": career_export,
        "comparison_scores": comparison_scores,
        "dimension_model": {
            "name": "of-6d-nested-multiplicative-v3",
            "dimensions": DIMENSIONS,
            "trend_index": (
                "Weighted geometric mean of OF momentum, recency, output acceleration, and "
                "activity-breadth percentiles (exponents 0.36, 0.24, 0.12, 0.28); rank becomes trend_pct."
            ),
            "final_score": (
                "Nested product: sqrt(wds_pct*trend_pct) * (min/max)^0.40 * "
                "(0.48+0.52*sqrt(geo_pct*cons_pct)) * (0.52+0.48*sqrt(breadth_pct)), capped at 1."
            ),
        },
        "ranking": ranking_rows,
    }, f, ensure_ascii=False, indent=2)

print("Building network.json...")
# ========== EXPORT: network.json ==========
network_eids = set()
for eid, p in all_profiles.items():
    if p["of_songs"] > 0 or p["of_inf_out"] > 3 or p["of_inf_in"] > 3:
        network_eids.add(eid)

top_network = sorted(
    [(eid, all_profiles[eid]) for eid in network_eids if eid in all_profiles],
    key=lambda x: x[1]["of_inf_out"] + x[1]["of_inf_in"] + x[1]["of_songs"] * 2,
    reverse=True,
)[:80]
top_eids = {eid for eid, _ in top_network}
for eid in list(comp_eids) + list(selected_eids):
    if eid and eid in all_profiles:
        top_eids.add(eid)

# Neighbors of comparison artists: one hop per raw influence edge, primary attribution only
for e in influence_edges:
    se = primary_entity_for_graph_node(e["source_song"])
    te = primary_entity_for_graph_node(e["target_song"])
    if not se or not te or se == te:
        continue
    if se in comp_eids and te in all_profiles:
        top_eids.add(te)
    if te in comp_eids and se in all_profiles:
        top_eids.add(se)

# Find score for network nodes
score_by_eid = {}
for item in ranked:
    score_by_eid[item["eid"]] = item["score"]

network_nodes = []
for eid in top_eids:
    p = all_profiles[eid]
    network_nodes.append({
        "id": p["display_name"],
        "eid": eid,
        "of_songs": p["of_songs"],
        "of_notable_ratio": p["of_notable_ratio"],
        "of_inf_out": p["of_inf_out"],
        "of_inf_in": p["of_inf_in"],
        "type": p["type"],
        "genres": p["genres"],
        "of_first_year": p["of_first_year"],
        "of_last_year": p["of_last_year"],
        "score": score_by_eid.get(eid, 0),
        "is_comparison": eid in comp_eids,
        "is_prediction": eid in set(selected_eids),
    })

typed_edges = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for e in influence_edges:
    se = primary_entity_for_graph_node(e["source_song"])
    te = primary_entity_for_graph_node(e["target_song"])
    if not se or not te or se == te:
        continue
    if se in top_eids and te in top_eids and se in all_profiles and te in all_profiles:
        sname = all_profiles[se]["display_name"]
        tname = all_profiles[te]["display_name"]
        typed_edges[sname][tname][e.get("type", "Unknown")] += 1

network_links = []
for s, targets in typed_edges.items():
    for t, types in targets.items():
        w = sum(types.values())
        dom = max(types.items(), key=lambda x: x[1])[0] if types else "Unknown"
        network_links.append({"source": s, "target": t, "weight": w, "type": dom, "types": dict(types)})

print(f"Building network.json... nodes: {len(network_nodes)}, links: {len(network_links)}")

random.seed(42)
W_NET, H_NET = 900, 600
positions = {n["id"]: [random.uniform(60, W_NET - 60), random.uniform(60, H_NET - 60)] for n in network_nodes}
for _ in range(50):
    forces = {n["id"]: [0.0, 0.0] for n in network_nodes}
    ids = [n["id"] for n in network_nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dx = positions[ids[i]][0] - positions[ids[j]][0]
            dy = positions[ids[i]][1] - positions[ids[j]][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            rep = 3200 / (dist * dist)
            forces[ids[i]][0] += dx / dist * rep
            forces[ids[i]][1] += dy / dist * rep
            forces[ids[j]][0] -= dx / dist * rep
            forces[ids[j]][1] -= dy / dist * rep
    for lk in network_links:
        s, t = lk["source"], lk["target"]
        if s in positions and t in positions:
            dx = positions[t][0] - positions[s][0]
            dy = positions[t][1] - positions[s][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            spring = (dist - 90) * 0.012
            forces[s][0] += dx / dist * spring
            forces[s][1] += dy / dist * spring
            forces[t][0] -= dx / dist * spring
            forces[t][1] -= dy / dist * spring
    for nid in ids:
        forces[nid][0] += (W_NET / 2 - positions[nid][0]) * 0.007
        forces[nid][1] += (H_NET / 2 - positions[nid][1]) * 0.007
    for nid in ids:
        positions[nid][0] = max(30, min(W_NET - 30, positions[nid][0] + 0.28 * forces[nid][0]))
        positions[nid][1] = max(30, min(H_NET - 30, positions[nid][1] + 0.28 * forces[nid][1]))
for n in network_nodes:
    n["x"] = round(positions[n["id"]][0], 1)
    n["y"] = round(positions[n["id"]][1], 1)

with open(os.path.join(OUT, "network.json"), "w", encoding="utf-8") as f:
    json.dump({"nodes": network_nodes, "links": network_links}, f, ensure_ascii=False, indent=2)


# ========== EXPORT: timeline.json (kept minimal for network context) ==========
of_timeline = defaultdict(lambda: {"total": 0, "notable": 0, "artists": set()})
for n in data["nodes"]:
    if n.get("Node Type") == "Song" and n.get("genre") == "Oceanus Folk":
        y = n.get("release_date") or n.get("written_date") or ""
        if y and y.isdigit():
            of_timeline[y]["total"] += 1
            of_timeline[y]["notable"] += 1 if n.get("notable") else 0
            for eid in song_to_entities.get(n["id"], set()):
                of_timeline[y]["artists"].add(eid)

timeline_rows = []
seen = set()
for y in sorted(of_timeline.keys()):
    artists = of_timeline[y]["artists"]
    new = artists - seen
    seen.update(artists)
    timeline_rows.append({
        "year": y,
        "total": of_timeline[y]["total"],
        "notable": of_timeline[y]["notable"],
        "artist_count": len(artists),
        "new_artists": len(new),
    })

with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump({"oceanus_folk": timeline_rows}, f, ensure_ascii=False, indent=2)


# ========== SUMMARY ==========
print(f"\n=== V2 Pipeline Export Summary ===")
print(f"Profiles: {len(profiles_export)}")
print(f"Comparison: {comparison_names}")
print(f"Predictions: {prediction_names}")
print(f"Candidates: {len(ranking_rows)}")
print(f"Network: {len(network_nodes)} nodes, {len(network_links)} links")
print(f"Timeline: {len(timeline_rows)} years")
for name in prediction_names:
    r = next((rr for rr in ranking_rows if rr["name"] == name), None)
    if r:
        print(f"  #{r['rank']} {name}: score={r['score']}, trend={r['trend_score']}, "
              f"of_songs={r['of_songs']}, of_commit={r['of_commitment']}, "
              f"of_inf={r['of_inf_out']}+{r['of_inf_in']}")
