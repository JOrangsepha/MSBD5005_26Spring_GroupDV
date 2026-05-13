import os
import runpy

# Delegate to the clean refactor pipeline and terminate early so any
# stale trailing content in this file is never executed.
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess_refactor.py"), run_name="__main__")
raise SystemExit(0)

id2node = {n["id"]: n for n in data["nodes"]}

INFLUENCE_TYPES = {"InStyleOf", "InterpolatesFrom", "CoverOf", "DirectlySamples", "LyricalReferenceTo"}
CREATIVE_ROLES = {"PerformerOf", "ComposerOf", "LyricistOf", "ProducerOf"}

# -------------------------------------------------------------------
# Graph mappings
# -------------------------------------------------------------------
of_song_ids = {n["id"] for n in data["nodes"] if n.get("genre") == "Oceanus Folk"}

entity_songs = defaultdict(set)  # entity_id -> songs
entity_roles = defaultdict(lambda: defaultdict(set))  # entity_id -> role -> songs
song_to_entities = defaultdict(set)  # song_id -> entities
group_members = defaultdict(list)  # group_id -> members
member_of_group = defaultdict(set)  # person_id -> groups

for e in data["links"]:
    et = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if et in CREATIVE_ROLES and src.get("Node Type") in ("Person", "MusicalGroup") and tgt.get("Node Type") == "Song":
        entity_songs[e["source"]].add(e["target"])
        entity_roles[e["source"]][et].add(e["target"])
        song_to_entities[e["target"]].add(e["source"])
    if et == "MemberOf":
        group_members[e["target"]].append(e["source"])
        member_of_group[e["source"]].add(e["target"])

entity_labels = defaultdict(set)
for e in data["links"]:
    et = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if et in ("RecordedBy", "DistributedBy") and src.get("Node Type") == "Song" and tgt.get("Node Type") == "RecordLabel":
        for eid in song_to_entities.get(e["source"], set()):
            entity_labels[eid].add(tgt.get("name", "Unknown Label"))

song_inf_out = defaultdict(int)
song_inf_in = defaultdict(int)
influence_edges = []
for e in data["links"]:
    if e.get("Edge Type") in INFLUENCE_TYPES:
        sid = e["source"]
        tid = e["target"]
        song_inf_out[sid] += 1
        song_inf_in[tid] += 1
        src_song = id2node.get(sid, {})
        tgt_song = id2node.get(tid, {})
        src_entities = song_to_entities.get(sid, set())
        tgt_entities = song_to_entities.get(tid, set())
        if src_entities and tgt_entities:
            influence_edges.append(
                {
                    "source_song_id": sid,
                    "target_song_id": tid,
                    "source_entities": list(src_entities),
                    "target_entities": list(tgt_entities),
                    "source_genre": src_song.get("genre", ""),
                    "target_genre": tgt_song.get("genre", ""),
                    "type": e.get("Edge Type", ""),
                }
            )

entity_collaborators = defaultdict(set)
for sid, entities in song_to_entities.items():
    for eid in entities:
        entity_collaborators[eid].update(entities - {eid})


def entity_influence(eid):
    sids = entity_songs.get(eid, set())
    return (
        sum(song_inf_out.get(sid, 0) for sid in sids),
        sum(song_inf_in.get(sid, 0) for sid in sids),
    )


def is_redundant_member(eid):
    groups = member_of_group.get(eid, set())
    if not groups:
        return False
    songs = entity_songs.get(eid, set())
    if not songs:
        return False
    for gid in groups:
        gs = entity_songs.get(gid, set())
        if gs and len(songs & gs) / max(len(songs), 1) > 0.8:
            return True
    return False


def hidden_duplicates():
    skip = set()
    persons = [
        eid
        for eid in entity_songs
        if id2node[eid].get("Node Type") == "Person" and not member_of_group.get(eid) and len(entity_songs[eid]) >= 2
    ]
    persons.sort()
    for i, a in enumerate(persons):
        if a in skip:
            continue
        for b in persons[i + 1 :]:
            if b in skip:
                continue
            if entity_songs[a] == entity_songs[b]:
                skip.add(b)
    return skip


def yearly_metrics(eid):
    yearly = defaultdict(lambda: {"songs": 0, "notable": 0, "inf_out": 0, "inf_in": 0})
    for sid in entity_songs.get(eid, set()):
        node = id2node.get(sid, {})
        y = node.get("release_date", "?")
        if not y or y == "?":
            continue
        yearly[y]["songs"] += 1
        yearly[y]["notable"] += 1 if node.get("notable") else 0
        yearly[y]["inf_out"] += song_inf_out.get(sid, 0)
        yearly[y]["inf_in"] += song_inf_in.get(sid, 0)
    return dict(sorted(yearly.items()))


def build_profile(eid):
    node = id2node[eid]
    sids = entity_songs.get(eid, set())
    if not sids:
        return None

    songs = [id2node[sid] for sid in sids]
    yearly = yearly_metrics(eid)
    years = [int(y) for y in yearly if str(y).isdigit()]
    inf_out, inf_in = entity_influence(eid)

    total_songs = len(sids)
    notable = sum(1 for s in songs if s.get("notable"))
    of_count = sum(1 for sid in sids if sid in of_song_ids)
    genres = Counter(s.get("genre", "Unknown") for s in songs)
    role_div = len(entity_roles[eid])
    collaborators = len(entity_collaborators.get(eid, set()))
    labels = len(entity_labels.get(eid, set()))

    if years:
        first_year = str(min(years))
        last_year = str(max(years))
        span = max(years) - min(years) + 1
    else:
        first_year = "?"
        last_year = "?"
        span = 1
    velocity = total_songs / max(span, 1)

    # 6 dimensions
    if years:
        y_min, y_max = min(years), max(years)
        recent = list(range(max(y_min, y_max - 2), y_max + 1))
        prior = list(range(max(y_min, y_max - 5), max(y_min, y_max - 2)))

        def strength(window):
            pop = sum(yearly.get(str(y), {}).get("songs", 0) for y in window)
            inf = sum(
                yearly.get(str(y), {}).get("inf_out", 0) + yearly.get(str(y), {}).get("inf_in", 0)
                for y in window
            )
            return pop + 0.35 * inf

        momentum = max(0.0, strength(recent) - strength(prior))
    else:
        momentum = 0.0

    q = []
    for y, d in yearly.items():
        if d["songs"] > 0:
            q.append(d["notable"] / d["songs"])
    if not q:
        quality_consistency = 0.0
    elif len(q) == 1:
        quality_consistency = q[0]
    else:
        quality_consistency = max(0.0, statistics.mean(q) * (1.0 - min(statistics.pstdev(q), 1.0)))

    influence_centrality = inf_out + 0.8 * inf_in

    probs = [c / max(total_songs, 1) for c in genres.values() if c > 0]
    if len(probs) <= 1:
        cross_genre_transfer = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in probs)
        cross_genre_transfer = entropy / math.log(len(probs))

    collaboration_leverage = collaborators * (1.0 + 0.2 * role_div)
    industry_support = labels + 0.3 * role_div

    return {
        "id": eid,
        "name": node.get("name", f"Entity {eid}"),
        "type": node.get("Node Type", "Person"),
        "members": [id2node[mid].get("name", "?") for mid in group_members.get(eid, [])]
        if node.get("Node Type") == "MusicalGroup"
        else [],
        "total_songs": total_songs,
        "of_songs": of_count,
        "notable": notable,
        "notable_ratio": round(notable / max(total_songs, 1), 4),
        "first_year": first_year,
        "last_year": last_year,
        "velocity": round(velocity, 4),
        "genres": dict(genres.most_common(10)),
        "genre_breadth": len(genres),
        "influence_out": inf_out,
        "influence_in": inf_in,
        "collaborators": collaborators,
        "labels": labels,
        "role_diversity": role_div,
        "yearly": yearly,
        "dimensions_raw": {
            "momentum": round(momentum, 4),
            "quality_consistency": round(quality_consistency, 4),
            "influence_centrality": round(influence_centrality, 4),
            "cross_genre_transfer": round(cross_genre_transfer, 4),
            "collaboration_leverage": round(collaboration_leverage, 4),
            "industry_support": round(industry_support, 4),
        },
    }


def find_entity(name, prefer_type=None):
    cands = [n for n in data["nodes"] if n.get("name") == name]
    if prefer_type:
        typed = [n for n in cands if n.get("Node Type") == prefer_type]
        if typed:
            cands = typed
    best_id = None
    best_count = -1
    for c in cands:
        cnt = len(entity_songs.get(c["id"], set()))
        if cnt > best_count:
            best_count = cnt
            best_id = c["id"]
    return best_id


COMPARISON_IDS = {
    "Sailor Shift": find_entity("Sailor Shift"),
    "Drowned Harbor": find_entity("Drowned Harbor", "MusicalGroup"),
    "Orla Seabloom": find_entity("Orla Seabloom"),
}

of_entities = set()
for e in data["links"]:
    if e.get("Edge Type") in CREATIVE_ROLES and e["target"] in of_song_ids:
        src = id2node.get(e["source"], {})
        if src.get("Node Type") in ("Person", "MusicalGroup"):
            of_entities.add(e["source"])

all_profiles = {}
skip_ids = hidden_duplicates()
for eid in of_entities:
    if is_redundant_member(eid) or eid in skip_ids:
        continue
    p = build_profile(eid)
    if p and p["total_songs"] > 0:
        all_profiles[eid] = p
for _, eid in COMPARISON_IDS.items():
    if eid and eid not in all_profiles:
        p = build_profile(eid)
        if p and p["total_songs"] > 0:
            all_profiles[eid] = p

name_counter = Counter(p["name"] for p in all_profiles.values())
for eid, p in all_profiles.items():
    p["display_name"] = p["name"] if name_counter[p["name"]] == 1 else f"{p['name']} (#{eid})"

# -------------------------------------------------------------------
# Ranking model
# -------------------------------------------------------------------
DIMENSIONS = [
    {"key": "momentum", "label": "Momentum", "weight": 1.4},
    {"key": "quality_consistency", "label": "Quality Consistency", "weight": 1.2},
    {"key": "influence_centrality", "label": "Influence Centrality", "weight": 1.5},
    {"key": "cross_genre_transfer", "label": "Cross-Genre Transfer", "weight": 1.0},
    {"key": "collaboration_leverage", "label": "Collaboration Leverage", "weight": 1.0},
    {"key": "industry_support", "label": "Industry Support", "weight": 0.9},
]

exclude_ids = set(v for v in COMPARISON_IDS.values() if v is not None)
ivy = find_entity("Ivy Echos", "MusicalGroup")
if ivy:
    exclude_ids.add(ivy)
    for m in group_members.get(ivy, []):
        exclude_ids.add(m)

funnel_stages = [
    {"label": "OF-connected artists", "count": len(all_profiles), "desc": "Artists linked to Oceanus Folk songs"}
]

candidates = []
for eid, p in all_profiles.items():
    if eid in exclude_ids:
        continue
    if p["total_songs"] < 2 or p["of_songs"] < 1:
        continue
    candidates.append((eid, p))

funnel_stages.append(
    {"label": "Eligible pool", "count": len(candidates), "desc": ">=2 songs and >=1 Oceanus Folk song"}
)

years_last = [int(p["last_year"]) for _, p in candidates if str(p["last_year"]).isdigit()]
year_min = min(years_last) if years_last else 1990
year_max = max(years_last) if years_last else 2040


def robust_max(values):
    if not values:
        return 1.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int(0.95 * (len(vals) - 1)))
    cap = vals[idx]
    return cap if cap > 0 else max(vals[-1], 1.0)


maxima = {d["key"]: robust_max([p["dimensions_raw"][d["key"]] for _, p in candidates]) for d in DIMENSIONS}


def norm(v, key):
    return max(0.0, min(v / max(maxima.get(key, 1.0), 1e-9), 1.0))


ranked = []
for eid, p in candidates:
    of_commitment = p["of_songs"] / max(p["total_songs"], 1)
    recency = 0.0
    if str(p["last_year"]).isdigit():
        recency = (int(p["last_year"]) - year_min) / max(year_max - year_min, 1)
    recency = max(0.0, min(recency, 1.0))

    dim_norm = {}
    dim_contrib = {}
    core_score = 0.0
    for d in DIMENSIONS:
        k = d["key"]
        n = norm(p["dimensions_raw"][k], k)
        c = n * d["weight"]
        dim_norm[k] = round(n, 4)
        dim_contrib[k] = round(c, 4)
        core_score += c

    five_year_readiness = 0.55 * recency + 0.45 * (of_commitment ** 1.35)
    final_score = core_score * 0.78 + five_year_readiness * 1.22
    ranked.append(
        {
            "eid": eid,
            "profile": p,
            "name": p["display_name"],
            "score": round(final_score, 4),
            "core_score": round(core_score, 4),
            "five_year_readiness": round(five_year_readiness, 4),
            "of_commitment": round(of_commitment, 4),
            "recency": round(recency, 4),
            "dim_norm": dim_norm,
            "dim_contrib": dim_contrib,
            "selected": False,
            "rejection_reason": None,
        }
    )
ranked.sort(key=lambda x: x["score"], reverse=True)

funnel_stages.append({"label": "Ranked by model", "count": len(ranked), "desc": "6 dimensions + 5-year readiness"})

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
        item["rejection_reason"] = "Shares songs with selected artist"
        continue
    if eid in picked_collabs:
        item["rejection_reason"] = "Direct collaborator with selected artist"
        continue
    item["selected"] = True
    selected_eids.append(eid)
    picked_songs.update(my_songs)
    picked_collabs.update(entity_collaborators.get(eid, set()))

funnel_stages.append({"label": "Final predictions", "count": len(selected_eids), "desc": "Top 3 with diversity controls"})


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a)) or 1.0
    mb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (ma * mb)


comp_profiles = {name: all_profiles[eid] for name, eid in COMPARISON_IDS.items() if eid in all_profiles}
arch_map = {"Sailor Shift": "Next Superstar", "Drowned Harbor": "Next Pioneer", "Orla Seabloom": "Next Rapid Riser"}
archetype_labels = {}
archetype_mirrors = {}
taken = set()

for eid in selected_eids:
    p = all_profiles[eid]
    pv = [norm(p["dimensions_raw"][d["key"]], d["key"]) for d in DIMENSIONS]
    sims = []
    for cname, cp in comp_profiles.items():
        cv = [norm(cp["dimensions_raw"][d["key"]], d["key"]) for d in DIMENSIONS]
        sims.append((cname, cosine_sim(pv, cv)))
    sims.sort(key=lambda x: x[1], reverse=True)
    for cname, _ in sims:
        if cname not in taken:
            taken.add(cname)
            archetype_labels[p["display_name"]] = arch_map.get(cname, "Next Rising Star")
            archetype_mirrors[p["display_name"]] = comp_profiles[cname]["display_name"]
            break

prediction_explanations = {}
for rank, item in enumerate(ranked, start=1):
    if not item["selected"]:
        continue
    p = item["profile"]
    top_strengths = sorted(item["dim_contrib"].items(), key=lambda x: x[1], reverse=True)[:3]
    risks = []
    if item["of_commitment"] < 0.45:
        risks.append("Low Oceanus Folk commitment may limit genre-specific breakout potential.")
    if item["five_year_readiness"] < 0.5:
        risks.append("Recent momentum is moderate; breakout may take longer than five years.")
    if item["dim_norm"]["quality_consistency"] < 0.45:
        risks.append("Quality signal is volatile across active years.")
    if not risks:
        risks.append("No single weak dimension; monitor sustainability under higher output pressure.")

    outlook = "Moderate 5-year breakout probability"
    if item["five_year_readiness"] >= 0.72:
        outlook = "High 5-year breakout probability"
    elif item["five_year_readiness"] >= 0.55:
        outlook = "Moderate-to-high 5-year breakout probability"

    prediction_explanations[p["display_name"]] = {
        "rank": rank,
        "score": item["score"],
        "core_score": item["core_score"],
        "five_year_readiness": item["five_year_readiness"],
        "strengths": [
            {"key": k, "label": next(d["label"] for d in DIMENSIONS if d["key"] == k), "contribution": v}
            for k, v in top_strengths
        ],
        "risks": risks[:2],
        "five_year_outlook": outlook,
    }

profiles_export = {p["display_name"]: p for p in all_profiles.values()}
comparison = [all_profiles[eid]["display_name"] for eid in COMPARISON_IDS.values() if eid in all_profiles]
predictions = [all_profiles[eid]["display_name"] for eid in selected_eids if eid in all_profiles]

ranking_rows = []
for i, item in enumerate(ranked[:30], start=1):
    p = item["profile"]
    ranking_rows.append(
        {
            "rank": i,
            "name": item["name"],
            "eid": item["eid"],
            "score": item["score"],
            "core_score": item["core_score"],
            "five_year_readiness": item["five_year_readiness"],
            "selected": item["selected"],
            "rejection_reason": item["rejection_reason"],
            "of_songs": p["of_songs"],
            "total_songs": p["total_songs"],
            "notable_ratio": p["notable_ratio"],
            "influence_out": p["influence_out"],
            "influence_in": p["influence_in"],
            "first_year": p["first_year"],
            "last_year": p["last_year"],
            "dim_norm": item["dim_norm"],
            "dim_contrib": item["dim_contrib"],
            "of_commitment": item["of_commitment"],
            "recency": item["recency"],
        }
    )

# -------------------------------------------------------------------
# Timeline + chord
# -------------------------------------------------------------------
of_timeline = defaultdict(lambda: {"total": 0, "notable": 0, "artists": set()})
for n in data["nodes"]:
    if n.get("Node Type") == "Song" and n.get("genre") == "Oceanus Folk" and n.get("release_date"):
        y = n["release_date"]
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
    timeline_rows.append(
        {
            "year": y,
            "total": of_timeline[y]["total"],
            "notable": of_timeline[y]["notable"],
            "artist_count": len(artists),
            "new_artists": len(new),
            "returning_artists": len(artists) - len(new),
        }
    )

flow_count = defaultdict(int)
for e in influence_edges:
    sg = e["source_genre"]
    tg = e["target_genre"]
    if sg and tg:
        flow_count[(sg, tg)] += 1
genre_flow = [{"source": s, "target": t, "value": v} for (s, t), v in flow_count.items() if v >= 3]
genre_flow.sort(key=lambda x: -x["value"])

genre_count = defaultdict(int)
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg and tg:
        genre_count[sg] += 1
        genre_count[tg] += 1
chord_genres = [g for g, _ in sorted(genre_count.items(), key=lambda x: -x[1])[:12]]
genre_idx = {g: i for i, g in enumerate(chord_genres)}
chord_matrix = [[0] * len(chord_genres) for _ in range(len(chord_genres))]
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg in genre_idx and tg in genre_idx:
        chord_matrix[genre_idx[sg]][genre_idx[tg]] += 1

# -------------------------------------------------------------------
# Influence network
# -------------------------------------------------------------------
network_eids = set()
for eid, p in all_profiles.items():
    if p["of_songs"] > 0 or p["influence_out"] > 3 or p["influence_in"] > 3:
        network_eids.add(eid)

top_network = sorted(
    [(eid, all_profiles[eid]) for eid in network_eids if eid in all_profiles],
    key=lambda x: x[1]["influence_out"] + x[1]["influence_in"] + x[1]["of_songs"] * 2,
    reverse=True,
)[:80]
top_eids = {eid for eid, _ in top_network}
for eid in list(COMPARISON_IDS.values()) + list(selected_eids):
    if eid and eid in all_profiles:
        top_eids.add(eid)

comp_set = set(COMPARISON_IDS.values())
pred_set = set(selected_eids)

network_nodes = []
for eid in top_eids:
    p = all_profiles[eid]
    network_nodes.append(
        {
            "id": p["display_name"],
            "eid": eid,
            "songs": p["total_songs"],
            "of_songs": p["of_songs"],
            "influence_out": p["influence_out"],
            "influence_in": p["influence_in"],
            "notable": p["notable"],
            "type": p["type"],
            "genres": p["genres"],
            "first_year": p["first_year"],
            "last_year": p["last_year"],
            "is_comparison": eid in comp_set,
            "is_prediction": eid in pred_set,
        }
    )

typed_edges = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for e in influence_edges:
    for se in e["source_entities"]:
        for te in e["target_entities"]:
            if se != te and se in top_eids and te in top_eids and se in all_profiles and te in all_profiles:
                sname = all_profiles[se]["display_name"]
                tname = all_profiles[te]["display_name"]
                typed_edges[sname][tname][e.get("type", "Unknown")] += 1

network_links = []
for s, targets in typed_edges.items():
    for t, types in targets.items():
        w = sum(types.values())
        dom = max(types.items(), key=lambda x: x[1])[0] if types else "Unknown"
        network_links.append({"source": s, "target": t, "weight": w, "type": dom, "types": dict(types)})

# deterministic layout
random.seed(42)
W_NET, H_NET = 740, 560
positions = {n["id"]: [random.uniform(50, W_NET - 50), random.uniform(50, H_NET - 50)] for n in network_nodes}
for _ in range(500):
    forces = {n["id"]: [0.0, 0.0] for n in network_nodes}
    ids = [n["id"] for n in network_nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dx = positions[ids[i]][0] - positions[ids[j]][0]
            dy = positions[ids[i]][1] - positions[ids[j]][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            rep = 2600 / (dist * dist)
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
            spring = (dist - 80) * 0.01
            forces[s][0] += dx / dist * spring
            forces[s][1] += dy / dist * spring
            forces[t][0] -= dx / dist * spring
            forces[t][1] -= dy / dist * spring
    for nid in ids:
        forces[nid][0] += (W_NET / 2 - positions[nid][0]) * 0.008
        forces[nid][1] += (H_NET / 2 - positions[nid][1]) * 0.008
    for nid in ids:
        positions[nid][0] = max(25, min(W_NET - 25, positions[nid][0] + 0.3 * forces[nid][0]))
        positions[nid][1] = max(25, min(H_NET - 25, positions[nid][1] + 0.3 * forces[nid][1]))
for n in network_nodes:
    n["x"] = round(positions[n["id"]][0], 1)
    n["y"] = round(positions[n["id"]][1], 1)

# -------------------------------------------------------------------
# Write files
# -------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "artists.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "profiles": profiles_export,
            "comparison": comparison,
            "predictions": predictions,
            "dimension_model": {
                "name": "rising-star-6d-core-plus-readiness",
                "dimensions": DIMENSIONS,
                "readiness_formula": "0.55 * recency + 0.45 * (of_commitment^1.35)",
                "final_formula": "0.78 * core_score + 1.22 * five_year_readiness",
            },
            "selection_funnel": {"stages": funnel_stages, "ranking": ranking_rows},
            "archetype_labels": archetype_labels,
            "archetype_mirrors": archetype_mirrors,
            "prediction_explanations": prediction_explanations,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "oceanus_folk": timeline_rows,
            "genre_flow": genre_flow[:50],
            "chord_genres": chord_genres,
            "chord_matrix": chord_matrix,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(OUT, "network.json"), "w", encoding="utf-8") as f:
    json.dump({"nodes": network_nodes, "links": network_links}, f, ensure_ascii=False, indent=2)

print("=== Rising Star Refactor Export Summary ===")
print(f"Profiles: {len(profiles_export)}")
print(f"Comparison artists: {comparison}")
print(f"Predictions: {predictions}")
print(f"Ranking rows: {len(ranking_rows)}")
print(f"Timeline years: {len(timeline_rows)}")
print(f"Network: {len(network_nodes)} nodes, {len(network_links)} links")
"""
Refactored preprocess pipeline for Oceanus Folk Rising Stars (Task 3).
"""
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

with open(os.path.join(BASE, "MC1_graph.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

id2node = {n["id"]: n for n in data["nodes"]}
INFLUENCE_TYPES = {"InStyleOf", "InterpolatesFrom", "CoverOf", "DirectlySamples", "LyricalReferenceTo"}
CREATIVE_ROLES = {"PerformerOf", "ComposerOf", "LyricistOf", "ProducerOf"}

of_song_ids = {n["id"] for n in data["nodes"] if n.get("genre") == "Oceanus Folk"}

entity_songs = defaultdict(set)
entity_roles = defaultdict(lambda: defaultdict(set))
song_to_entities = defaultdict(set)
group_members = defaultdict(list)
member_of_group = defaultdict(set)

for e in data["links"]:
    etype = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if etype in CREATIVE_ROLES and src.get("Node Type") in ("Person", "MusicalGroup") and tgt.get("Node Type") == "Song":
        entity_songs[e["source"]].add(e["target"])
        entity_roles[e["source"]][etype].add(e["target"])
        song_to_entities[e["target"]].add(e["source"])
    if etype == "MemberOf":
        group_members[e["target"]].append(e["source"])
        member_of_group[e["source"]].add(e["target"])

entity_labels = defaultdict(set)
for e in data["links"]:
    etype = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if etype in ("RecordedBy", "DistributedBy") and src.get("Node Type") == "Song" and tgt.get("Node Type") == "RecordLabel":
        for eid in song_to_entities.get(e["source"], set()):
            entity_labels[eid].add(tgt.get("name", "Unknown Label"))

song_inf_out = defaultdict(int)
song_inf_in = defaultdict(int)
influence_edges = []
for e in data["links"]:
    if e.get("Edge Type") in INFLUENCE_TYPES:
        sid = e["source"]
        tid = e["target"]
        song_inf_out[sid] += 1
        song_inf_in[tid] += 1
        src_song = id2node.get(sid, {})
        tgt_song = id2node.get(tid, {})
        src_entities = song_to_entities.get(sid, set())
        tgt_entities = song_to_entities.get(tid, set())
        if src_entities and tgt_entities:
            influence_edges.append(
                {
                    "source_song_id": sid,
                    "target_song_id": tid,
                    "source_entities": list(src_entities),
                    "target_entities": list(tgt_entities),
                    "source_genre": src_song.get("genre", ""),
                    "target_genre": tgt_song.get("genre", ""),
                    "type": e.get("Edge Type", ""),
                }
            )

entity_collaborators = defaultdict(set)
for sid, ents in song_to_entities.items():
    for eid in ents:
        entity_collaborators[eid].update(ents - {eid})


def entity_influence(eid):
    songs = entity_songs.get(eid, set())
    return (
        sum(song_inf_out.get(sid, 0) for sid in songs),
        sum(song_inf_in.get(sid, 0) for sid in songs),
    )


def is_redundant_member(eid):
    groups = member_of_group.get(eid, set())
    if not groups:
        return False
    my_songs = entity_songs.get(eid, set())
    if not my_songs:
        return False
    for gid in groups:
        gs = entity_songs.get(gid, set())
        if gs and len(my_songs & gs) / max(len(my_songs), 1) > 0.8:
            return True
    return False


def hidden_band_duplicates():
    skip = set()
    persons = [
        eid
        for eid in entity_songs
        if id2node[eid].get("Node Type") == "Person" and not member_of_group.get(eid) and len(entity_songs[eid]) >= 2
    ]
    persons.sort()
    for i, a in enumerate(persons):
        if a in skip:
            continue
        for b in persons[i + 1 :]:
            if b in skip:
                continue
            if entity_songs[a] == entity_songs[b]:
                skip.add(b)
    return skip


def yearly_for_entity(eid):
    yearly = defaultdict(lambda: {"songs": 0, "notable": 0, "inf_out": 0, "inf_in": 0})
    for sid in entity_songs.get(eid, set()):
        n = id2node.get(sid, {})
        y = n.get("release_date", "?")
        if not y or y == "?":
            continue
        yearly[y]["songs"] += 1
        yearly[y]["notable"] += 1 if n.get("notable") else 0
        yearly[y]["inf_out"] += song_inf_out.get(sid, 0)
        yearly[y]["inf_in"] += song_inf_in.get(sid, 0)
    return dict(sorted(yearly.items()))


def build_profile(eid):
    node = id2node[eid]
    song_ids = entity_songs.get(eid, set())
    if not song_ids:
        return None

    songs = [id2node[sid] for sid in song_ids]
    yearly = yearly_for_entity(eid)
    years = [int(y) for y in yearly if str(y).isdigit()]
    inf_out, inf_in = entity_influence(eid)
    genres = Counter(s.get("genre", "Unknown") for s in songs)
    total = len(song_ids)
    notable = sum(1 for s in songs if s.get("notable"))
    of_count = sum(1 for sid in song_ids if sid in of_song_ids)
    role_div = len(entity_roles[eid])
    collaborators = len(entity_collaborators.get(eid, set()))
    labels = len(entity_labels.get(eid, set()))

    if years:
        first_year = str(min(years))
        last_year = str(max(years))
        span = max(years) - min(years) + 1
    else:
        first_year = "?"
        last_year = "?"
        span = 1
    velocity = total / max(span, 1)

    # 6 core dimensions
    if years:
        y_min, y_max = min(years), max(years)
        recent = list(range(max(y_min, y_max - 2), y_max + 1))
        prior = list(range(max(y_min, y_max - 5), max(y_min, y_max - 2)))

        def strength(ws):
            pop = sum(yearly.get(str(y), {}).get("songs", 0) for y in ws)
            inf = sum(yearly.get(str(y), {}).get("inf_out", 0) + yearly.get(str(y), {}).get("inf_in", 0) for y in ws)
            return pop + 0.35 * inf

        momentum = max(0.0, strength(recent) - strength(prior))
    else:
        momentum = 0.0

    yr_quality = []
    for y, v in yearly.items():
        if v["songs"] > 0:
            yr_quality.append(v["notable"] / v["songs"])
    if not yr_quality:
        quality_consistency = 0.0
    elif len(yr_quality) == 1:
        quality_consistency = yr_quality[0]
    else:
        quality_consistency = max(0.0, statistics.mean(yr_quality) * (1.0 - min(statistics.pstdev(yr_quality), 1.0)))

    influence_centrality = inf_out + 0.8 * inf_in

    probs = [c / total for c in genres.values() if c > 0]
    if len(probs) <= 1:
        cross_genre_transfer = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in probs)
        cross_genre_transfer = entropy / math.log(len(probs))

    collaboration_leverage = collaborators * (1.0 + 0.2 * role_div)
    industry_support = labels + 0.3 * role_div

    return {
        "id": eid,
        "name": node.get("name", f"Entity {eid}"),
        "type": node.get("Node Type", "Person"),
        "members": [id2node[mid].get("name", "?") for mid in group_members.get(eid, [])]
        if node.get("Node Type") == "MusicalGroup"
        else [],
        "total_songs": total,
        "of_songs": of_count,
        "notable": notable,
        "notable_ratio": round(notable / max(total, 1), 4),
        "first_year": first_year,
        "last_year": last_year,
        "velocity": round(velocity, 4),
        "genres": dict(genres.most_common(10)),
        "genre_breadth": len(genres),
        "influence_out": inf_out,
        "influence_in": inf_in,
        "collaborators": collaborators,
        "labels": labels,
        "role_diversity": role_div,
        "yearly": yearly,
        "dimensions_raw": {
            "momentum": round(momentum, 4),
            "quality_consistency": round(quality_consistency, 4),
            "influence_centrality": round(influence_centrality, 4),
            "cross_genre_transfer": round(cross_genre_transfer, 4),
            "collaboration_leverage": round(collaboration_leverage, 4),
            "industry_support": round(industry_support, 4),
        },
    }


of_entities = set()
for e in data["links"]:
    if e.get("Edge Type") in CREATIVE_ROLES and e["target"] in of_song_ids:
        src = id2node.get(e["source"], {})
        if src.get("Node Type") in ("Person", "MusicalGroup"):
            of_entities.add(e["source"])


def find_entity(name, prefer_type=None):
    cands = [n for n in data["nodes"] if n.get("name") == name]
    if prefer_type:
        typed = [n for n in cands if n.get("Node Type") == prefer_type]
        if typed:
            cands = typed
    best_id, best_cnt = None, -1
    for c in cands:
        cnt = len(entity_songs.get(c["id"], set()))
        if cnt > best_cnt:
            best_cnt = cnt
            best_id = c["id"]
    return best_id


COMPARISON_IDS = {
    "Sailor Shift": find_entity("Sailor Shift"),
    "Drowned Harbor": find_entity("Drowned Harbor", "MusicalGroup"),
    "Orla Seabloom": find_entity("Orla Seabloom"),
}

all_profiles = {}
dup_skip = hidden_band_duplicates()
for eid in of_entities:
    if is_redundant_member(eid) or eid in dup_skip:
        continue
    p = build_profile(eid)
    if p and p["total_songs"] > 0:
        all_profiles[eid] = p
for _, eid in COMPARISON_IDS.items():
    if eid and eid not in all_profiles:
        p = build_profile(eid)
        if p and p["total_songs"] > 0:
            all_profiles[eid] = p

name_count = Counter(p["name"] for p in all_profiles.values())
for eid, p in all_profiles.items():
    p["display_name"] = p["name"] if name_count[p["name"]] == 1 else f"{p['name']} (#{eid})"

DIMENSIONS = [
    {"key": "momentum", "label": "Momentum", "weight": 1.4},
    {"key": "quality_consistency", "label": "Quality Consistency", "weight": 1.2},
    {"key": "influence_centrality", "label": "Influence Centrality", "weight": 1.5},
    {"key": "cross_genre_transfer", "label": "Cross-Genre Transfer", "weight": 1.0},
    {"key": "collaboration_leverage", "label": "Collaboration Leverage", "weight": 1.0},
    {"key": "industry_support", "label": "Industry Support", "weight": 0.9},
]

comp_ids = set(v for v in COMPARISON_IDS.values() if v is not None)
exclude_ids = set(comp_ids)
ivy = find_entity("Ivy Echos", "MusicalGroup")
if ivy:
    exclude_ids.add(ivy)
    for m in group_members.get(ivy, []):
        exclude_ids.add(m)

funnel_stages = [
    {"label": "OF-connected artists", "count": len(all_profiles), "desc": "All artists linked to Oceanus Folk songs"}
]

candidates = []
for eid, p in all_profiles.items():
    if eid in exclude_ids:
        continue
    if p["total_songs"] < 2:
        continue
    if p["of_songs"] < 1:
        continue
    candidates.append((eid, p))
funnel_stages.append({"label": "Eligible pool", "count": len(candidates), "desc": ">=2 songs and >=1 Oceanus Folk song"})

years_last = [int(p["last_year"]) for _, p in candidates if str(p["last_year"]).isdigit()]
year_min = min(years_last) if years_last else 1990
year_max = max(years_last) if years_last else 2040


def robust_max(values):
    if not values:
        return 1.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int(0.95 * (len(vals) - 1)))
    cap = vals[idx]
    return cap if cap > 0 else max(vals[-1], 1.0)


maxima = {d["key"]: robust_max([p["dimensions_raw"][d["key"]] for _, p in candidates]) for d in DIMENSIONS}


def norm(v, key):
    return max(0.0, min(v / max(maxima.get(key, 1.0), 1e-9), 1.0))


ranked = []
for eid, p in candidates:
    of_commitment = p["of_songs"] / max(p["total_songs"], 1)
    recency = 0.0
    if str(p["last_year"]).isdigit():
        recency = (int(p["last_year"]) - year_min) / max(year_max - year_min, 1)
    recency = max(0.0, min(recency, 1.0))
    dim_norm = {}
    dim_contrib = {}
    core = 0.0
    for d in DIMENSIONS:
        k = d["key"]
        n = norm(p["dimensions_raw"][k], k)
        c = n * d["weight"]
        dim_norm[k] = round(n, 4)
        dim_contrib[k] = round(c, 4)
        core += c
    readiness = 0.55 * recency + 0.45 * (of_commitment ** 1.35)
    score = core * 0.78 + readiness * 1.22
    ranked.append(
        {
            "eid": eid,
            "profile": p,
            "name": p["display_name"],
            "score": round(score, 4),
            "core_score": round(core, 4),
            "five_year_readiness": round(readiness, 4),
            "of_commitment": round(of_commitment, 4),
            "recency": round(recency, 4),
            "dim_norm": dim_norm,
            "dim_contrib": dim_contrib,
            "selected": False,
            "rejection_reason": None,
        }
    )
ranked.sort(key=lambda x: x["score"], reverse=True)
funnel_stages.append({"label": "Ranked by model", "count": len(ranked), "desc": "6D core + 5-year readiness"})

selected = []
picked_songs = set()
picked_collabs = set()
for item in ranked:
    p = item["profile"]
    eid = item["eid"]
    songs = entity_songs.get(eid, set())
    if len(selected) >= 3:
        item["rejection_reason"] = "Top 3 already selected"
        continue
    if picked_songs and (songs & picked_songs):
        item["rejection_reason"] = "Shares songs with selected artist"
        continue
    if eid in picked_collabs:
        item["rejection_reason"] = "Direct collaborator with selected artist"
        continue
    item["selected"] = True
    selected.append(eid)
    picked_songs.update(songs)
    picked_collabs.update(entity_collaborators.get(eid, set()))
funnel_stages.append({"label": "Final predictions", "count": len(selected), "desc": "Top 3 after diversity controls"})


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a)) or 1.0
    mb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (ma * mb)


comp_profiles = {n: all_profiles[eid] for n, eid in COMPARISON_IDS.items() if eid in all_profiles}
archetype_map = {"Sailor Shift": "Next Superstar", "Drowned Harbor": "Next Pioneer", "Orla Seabloom": "Next Rapid Riser"}
archetype_labels = {}
archetype_mirrors = {}
taken = set()

for eid in selected:
    p = all_profiles[eid]
    pv = [norm(p["dimensions_raw"][d["key"]], d["key"]) for d in DIMENSIONS]
    sims = []
    for cname, cp in comp_profiles.items():
        cv = [norm(cp["dimensions_raw"][d["key"]], d["key"]) for d in DIMENSIONS]
        sims.append((cname, cosine_sim(pv, cv)))
    sims.sort(key=lambda x: x[1], reverse=True)
    for cname, _ in sims:
        if cname not in taken:
            taken.add(cname)
            archetype_labels[p["display_name"]] = archetype_map.get(cname, "Next Rising Star")
            archetype_mirrors[p["display_name"]] = comp_profiles[cname]["display_name"]
            break

prediction_explanations = {}
for i, item in enumerate(ranked, start=1):
    if not item["selected"]:
        continue
    p = item["profile"]
    top_strengths = sorted(item["dim_contrib"].items(), key=lambda x: x[1], reverse=True)[:3]
    risks = []
    if item["of_commitment"] < 0.45:
        risks.append("Low Oceanus Folk commitment may limit genre-specific breakout potential.")
    if item["five_year_readiness"] < 0.5:
        risks.append("Recent momentum is moderate; breakout may take longer than 5 years.")
    if item["dim_norm"]["quality_consistency"] < 0.45:
        risks.append("Quality signal is volatile across active years.")
    if not risks:
        risks.append("No single weak dimension; monitor whether output remains sustainable.")

    outlook = "Moderate 5-year breakout probability"
    if item["five_year_readiness"] >= 0.72:
        outlook = "High 5-year breakout probability"
    elif item["five_year_readiness"] >= 0.55:
        outlook = "Moderate-to-high 5-year breakout probability"

    prediction_explanations[p["display_name"]] = {
        "rank": i,
        "score": item["score"],
        "core_score": item["core_score"],
        "five_year_readiness": item["five_year_readiness"],
        "strengths": [
            {"key": k, "label": next(d["label"] for d in DIMENSIONS if d["key"] == k), "contribution": v}
            for k, v in top_strengths
        ],
        "risks": risks[:2],
        "five_year_outlook": outlook,
    }

profiles_export = {p["display_name"]: p for p in all_profiles.values()}
comparison = [all_profiles[eid]["display_name"] for eid in COMPARISON_IDS.values() if eid in all_profiles]
predictions = [all_profiles[eid]["display_name"] for eid in selected if eid in all_profiles]
ranking = []
for i, item in enumerate(ranked[:30], start=1):
    p = item["profile"]
    ranking.append(
        {
            "rank": i,
            "name": item["name"],
            "eid": item["eid"],
            "score": item["score"],
            "core_score": item["core_score"],
            "five_year_readiness": item["five_year_readiness"],
            "selected": item["selected"],
            "rejection_reason": item["rejection_reason"],
            "of_songs": p["of_songs"],
            "total_songs": p["total_songs"],
            "notable_ratio": p["notable_ratio"],
            "influence_out": p["influence_out"],
            "influence_in": p["influence_in"],
            "first_year": p["first_year"],
            "last_year": p["last_year"],
            "dim_norm": item["dim_norm"],
            "dim_contrib": item["dim_contrib"],
            "of_commitment": item["of_commitment"],
            "recency": item["recency"],
        }
    )

# timeline/chord
of_timeline = defaultdict(lambda: {"total": 0, "notable": 0, "artists": set()})
for n in data["nodes"]:
    if n.get("Node Type") == "Song" and n.get("genre") == "Oceanus Folk" and n.get("release_date"):
        y = n["release_date"]
        of_timeline[y]["total"] += 1
        of_timeline[y]["notable"] += 1 if n.get("notable") else 0
        for eid in song_to_entities.get(n["id"], set()):
            of_timeline[y]["artists"].add(eid)
timeline = []
seen = set()
for y in sorted(of_timeline.keys()):
    artists = of_timeline[y]["artists"]
    new = artists - seen
    seen.update(artists)
    timeline.append(
        {
            "year": y,
            "total": of_timeline[y]["total"],
            "notable": of_timeline[y]["notable"],
            "artist_count": len(artists),
            "new_artists": len(new),
            "returning_artists": len(artists) - len(new),
        }
    )

flow = defaultdict(int)
for e in influence_edges:
    sg = e["source_genre"]
    tg = e["target_genre"]
    if sg and tg:
        flow[(sg, tg)] += 1
genre_flow = [{"source": s, "target": t, "value": v} for (s, t), v in flow.items() if v >= 3]
genre_flow.sort(key=lambda x: -x["value"])

genre_counts = defaultdict(int)
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg and tg:
        genre_counts[sg] += 1
        genre_counts[tg] += 1
chord_genres = [g for g, _ in sorted(genre_counts.items(), key=lambda x: -x[1])[:12]]
idx = {g: i for i, g in enumerate(chord_genres)}
chord_matrix = [[0] * len(chord_genres) for _ in range(len(chord_genres))]
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg in idx and tg in idx:
        chord_matrix[idx[sg]][idx[tg]] += 1

# network
network_eids = set()
for eid, p in all_profiles.items():
    if p["of_songs"] > 0 or p["influence_out"] > 3 or p["influence_in"] > 3:
        network_eids.add(eid)
top = sorted(
    [(eid, all_profiles[eid]) for eid in network_eids if eid in all_profiles],
    key=lambda x: x[1]["influence_out"] + x[1]["influence_in"] + x[1]["of_songs"] * 2,
    reverse=True,
)[:80]
top_eids = {eid for eid, _ in top}
for eid in list(COMPARISON_IDS.values()) + list(selected):
    if eid and eid in all_profiles:
        top_eids.add(eid)

comp_set = set(COMPARISON_IDS.values())
pred_set = set(selected)

network_nodes = []
for eid in top_eids:
    p = all_profiles[eid]
    network_nodes.append(
        {
            "id": p["display_name"],
            "eid": eid,
            "songs": p["total_songs"],
            "of_songs": p["of_songs"],
            "influence_out": p["influence_out"],
            "influence_in": p["influence_in"],
            "notable": p["notable"],
            "type": p["type"],
            "genres": p["genres"],
            "first_year": p["first_year"],
            "last_year": p["last_year"],
            "is_comparison": eid in comp_set,
            "is_prediction": eid in pred_set,
        }
    )

typed_links = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for e in influence_edges:
    for se in e["source_entities"]:
        for te in e["target_entities"]:
            if se != te and se in top_eids and te in top_eids and se in all_profiles and te in all_profiles:
                sn = all_profiles[se]["display_name"]
                tn = all_profiles[te]["display_name"]
                et = e.get("type", "Unknown")
                typed_links[sn][tn][et] += 1
network_links = []
for s, tgts in typed_links.items():
    for t, types in tgts.items():
        w = sum(types.values())
        dom = max(types.items(), key=lambda x: x[1])[0] if types else "Unknown"
        network_links.append({"source": s, "target": t, "weight": w, "type": dom, "types": dict(types)})

random.seed(42)
W_NET, H_NET = 740, 560
positions = {n["id"]: [random.uniform(50, W_NET - 50), random.uniform(50, H_NET - 50)] for n in network_nodes}
for _ in range(500):
    forces = {n["id"]: [0.0, 0.0] for n in network_nodes}
    ids = [n["id"] for n in network_nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dx = positions[ids[i]][0] - positions[ids[j]][0]
            dy = positions[ids[i]][1] - positions[ids[j]][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            rep = 2600 / (dist * dist)
            fx = dx / dist * rep
            fy = dy / dist * rep
            forces[ids[i]][0] += fx
            forces[ids[i]][1] += fy
            forces[ids[j]][0] -= fx
            forces[ids[j]][1] -= fy
    for lk in network_links:
        s = lk["source"]
        t = lk["target"]
        if s in positions and t in positions:
            dx = positions[t][0] - positions[s][0]
            dy = positions[t][1] - positions[s][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            spring = (dist - 80) * 0.01
            forces[s][0] += dx / dist * spring
            forces[s][1] += dy / dist * spring
            forces[t][0] -= dx / dist * spring
            forces[t][1] -= dy / dist * spring
    for nid in ids:
        forces[nid][0] += (W_NET / 2 - positions[nid][0]) * 0.008
        forces[nid][1] += (H_NET / 2 - positions[nid][1]) * 0.008
    for nid in ids:
        positions[nid][0] = max(25, min(W_NET - 25, positions[nid][0] + forces[nid][0] * 0.3))
        positions[nid][1] = max(25, min(H_NET - 25, positions[nid][1] + forces[nid][1] * 0.3))
for n in network_nodes:
    n["x"] = round(positions[n["id"]][0], 1)
    n["y"] = round(positions[n["id"]][1], 1)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "artists.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "profiles": profiles_export,
            "comparison": comparison,
            "predictions": predictions,
            "dimension_model": {
                "name": "rising-star-6d-core-plus-readiness",
                "dimensions": DIMENSIONS,
                "readiness_formula": "0.55 * recency + 0.45 * (of_commitment^1.35)",
                "final_formula": "0.78 * core_score + 1.22 * five_year_readiness",
            },
            "selection_funnel": {"stages": funnel_stages, "ranking": ranking},
            "archetype_labels": archetype_labels,
            "archetype_mirrors": archetype_mirrors,
            "prediction_explanations": prediction_explanations,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "oceanus_folk": timeline,
            "genre_flow": genre_flow[:50],
            "chord_genres": chord_genres,
            "chord_matrix": chord_matrix,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(OUT, "network.json"), "w", encoding="utf-8") as f:
    json.dump({"nodes": network_nodes, "links": network_links}, f, ensure_ascii=False, indent=2)

print("=== Rising Star Refactor Export Summary ===")
print(f"Profiles: {len(profiles_export)}")
print(f"Comparison artists: {comparison}")
print(f"Predictions: {predictions}")
print(f"Ranking rows: {len(ranking)}")
print(f"Timeline years: {len(timeline)}")
print(f"Network: {len(network_nodes)} nodes, {len(network_links)} links")
"""
Refactored preprocess pipeline for Oceanus Folk Rising Stars (Task 3).

Goals supported:
1) Profile what defines a rising star (explicit weighted dimensions)
2) Compare three artist careers (popularity + influence trajectories)
3) Predict three next Oceanus Folk stars over the next five years

Outputs:
- viz/data/artists.json
- viz/data/timeline.json
- viz/data/network.json
"""
import json
import math
import os
import statistics
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

with open(os.path.join(BASE, "MC1_graph.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

id2node = {n["id"]: n for n in data["nodes"]}

INFLUENCE_TYPES = {"InStyleOf", "InterpolatesFrom", "CoverOf", "DirectlySamples", "LyricalReferenceTo"}
CREATIVE_ROLES = {"PerformerOf", "ComposerOf", "LyricistOf", "ProducerOf"}

# ------------------------------------------------------------------
# Build base mappings
# ------------------------------------------------------------------
of_song_ids = {n["id"] for n in data["nodes"] if n.get("genre") == "Oceanus Folk"}
song_year = {n["id"]: n.get("release_date", "?") for n in data["nodes"] if n.get("Node Type") == "Song"}

entity_songs = defaultdict(set)  # entity_id -> song ids
entity_roles = defaultdict(lambda: defaultdict(set))  # entity_id -> role -> song ids
song_to_entities = defaultdict(set)  # song_id -> entity ids
group_members = defaultdict(list)  # group_id -> person ids
member_of_group = defaultdict(set)  # person_id -> group ids

for e in data["links"]:
    etype = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})

    if etype in CREATIVE_ROLES:
        if src.get("Node Type") in ("Person", "MusicalGroup") and tgt.get("Node Type") == "Song":
            entity_songs[e["source"]].add(e["target"])
            entity_roles[e["source"]][etype].add(e["target"])
            song_to_entities[e["target"]].add(e["source"])

    if etype == "MemberOf":
        # source person -> target group
        group_members[e["target"]].append(e["source"])
        member_of_group[e["source"]].add(e["target"])

# Label support (industry support proxy)
entity_labels = defaultdict(set)
for e in data["links"]:
    etype = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if etype in ("RecordedBy", "DistributedBy"):
        if src.get("Node Type") == "Song" and tgt.get("Node Type") == "RecordLabel":
            for eid in song_to_entities.get(e["source"], set()):
                entity_labels[eid].add(tgt.get("name", "Unknown Label"))

# Song-level influence counts
song_inf_out = defaultdict(int)
song_inf_in = defaultdict(int)
influence_edges = []
for e in data["links"]:
    if e.get("Edge Type") in INFLUENCE_TYPES:
        src_song_id = e["source"]
        tgt_song_id = e["target"]
        song_inf_out[src_song_id] += 1
        song_inf_in[tgt_song_id] += 1

        src_song = id2node.get(src_song_id, {})
        tgt_song = id2node.get(tgt_song_id, {})
        src_entities = song_to_entities.get(src_song_id, set())
        tgt_entities = song_to_entities.get(tgt_song_id, set())
        if src_entities and tgt_entities:
            influence_edges.append(
                {
                    "source_song_id": src_song_id,
                    "target_song_id": tgt_song_id,
                    "source_song": src_song.get("name", ""),
                    "target_song": tgt_song.get("name", ""),
                    "source_entities": list(src_entities),
                    "target_entities": list(tgt_entities),
                    "source_genre": src_song.get("genre", ""),
                    "target_genre": tgt_song.get("genre", ""),
                    "type": e.get("Edge Type", ""),
                }
            )


def entity_influence(eid):
    songs = entity_songs.get(eid, set())
    out_val = sum(song_inf_out.get(sid, 0) for sid in songs)
    in_val = sum(song_inf_in.get(sid, 0) for sid in songs)
    return out_val, in_val


# Collaboration graph
entity_collaborators = defaultdict(set)
for sid, ents in song_to_entities.items():
    for eid in ents:
        entity_collaborators[eid].update(ents - {eid})


def is_redundant_member(eid):
    """
    If a person overlaps heavily with one of their groups, keep group representation.
    """
    groups = member_of_group.get(eid, set())
    if not groups:
        return False
    my_songs = entity_songs.get(eid, set())
    if not my_songs:
        return False
    for gid in groups:
        g_songs = entity_songs.get(gid, set())
        if g_songs and len(my_songs & g_songs) / max(len(my_songs), 1) > 0.8:
            return True
    return False


def find_hidden_band_representatives():
    """
    Detect exact same-song duplicates among solo people not officially grouped.
    """
    skip = set()
    person_ids = [
        eid
        for eid in entity_songs
        if id2node[eid].get("Node Type") == "Person" and not member_of_group.get(eid) and len(entity_songs[eid]) >= 2
    ]
    person_ids.sort()
    for i, a in enumerate(person_ids):
        if a in skip:
            continue
        sa = entity_songs[a]
        for b in person_ids[i + 1 :]:
            if b in skip:
                continue
            if sa == entity_songs[b]:
                skip.add(b)
    return skip


hidden_band_skip = find_hidden_band_representatives()


def yearly_influence_for_entity(eid):
    """
    Assign each song's influence to that song year for trend analysis.
    """
    yearly = defaultdict(lambda: {"songs": 0, "notable": 0, "inf_out": 0, "inf_in": 0})
    for sid in entity_songs.get(eid, set()):
        node = id2node.get(sid, {})
        y = node.get("release_date", "?")
        if not y or y == "?":
            continue
        yearly[y]["songs"] += 1
        yearly[y]["notable"] += 1 if node.get("notable") else 0
        yearly[y]["inf_out"] += song_inf_out.get(sid, 0)
        yearly[y]["inf_in"] += song_inf_in.get(sid, 0)
    return dict(sorted(yearly.items()))


def build_profile(eid):
    node = id2node[eid]
    song_ids = entity_songs.get(eid, set())
    if not song_ids:
        return None

    songs = [id2node[sid] for sid in song_ids]
    of_count = sum(1 for sid in song_ids if sid in of_song_ids)
    notable_count = sum(1 for s in songs if s.get("notable"))
    genres = Counter(s.get("genre", "Unknown") for s in songs)
    role_diversity = len(entity_roles[eid])
    labels = len(entity_labels.get(eid, set()))
    collaborators = len(entity_collaborators.get(eid, set()))
    inf_out, inf_in = entity_influence(eid)

    year_map = yearly_influence_for_entity(eid)
    years = [int(y) for y in year_map if str(y).isdigit()]
    if years:
        first_year = str(min(years))
        last_year = str(max(years))
    else:
        first_year = "?"
        last_year = "?"

    # Popularity velocity
    if len(years) >= 2:
        span = max(years) - min(years) + 1
        velocity = len(song_ids) / max(span, 1)
    else:
        velocity = float(len(song_ids))

    # Dimension 1: momentum (recent popularity+influence lift vs prior window)
    if years:
        y_min, y_max = min(years), max(years)
        recent_window = list(range(max(y_min, y_max - 2), y_max + 1))
        prior_window = list(range(max(y_min, y_max - 5), max(y_min, y_max - 2)))

        def window_strength(ws):
            pop = sum(year_map.get(str(y), {}).get("songs", 0) for y in ws)
            inf = sum(
                year_map.get(str(y), {}).get("inf_out", 0) + year_map.get(str(y), {}).get("inf_in", 0) for y in ws
            )
            return pop + 0.35 * inf

        recent_strength = window_strength(recent_window)
        prior_strength = window_strength(prior_window)
        momentum = max(0.0, recent_strength - prior_strength)
    else:
        momentum = 0.0

    # Dimension 2: quality consistency (mean notable ratio penalized by volatility)
    yearly_quality = []
    for y, v in year_map.items():
        if v["songs"] > 0:
            yearly_quality.append(v["notable"] / v["songs"])
    if not yearly_quality:
        quality_consistency = 0.0
    elif len(yearly_quality) == 1:
        quality_consistency = yearly_quality[0]
    else:
        quality_consistency = max(
            0.0, statistics.mean(yearly_quality) * (1.0 - min(statistics.pstdev(yearly_quality), 1.0))
        )

    # Dimension 3: influence centrality
    influence_centrality = inf_out + 0.8 * inf_in

    # Dimension 4: cross-genre transfer (entropy normalized)
    total_genre_songs = sum(genres.values()) or 1
    probs = [c / total_genre_songs for c in genres.values() if c > 0]
    if len(probs) <= 1:
        cross_genre_transfer = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in probs)
        cross_genre_transfer = entropy / math.log(len(probs))

    # Dimension 5: collaboration leverage
    collaboration_leverage = collaborators * (1.0 + 0.2 * role_diversity)

    # Dimension 6: industry support
    industry_support = labels + 0.3 * role_diversity

    return {
        "id": eid,
        "name": node.get("name", f"Entity {eid}"),
        "type": node.get("Node Type", "Person"),
        "members": [id2node[mid].get("name", "?") for mid in group_members.get(eid, [])]
        if node.get("Node Type") == "MusicalGroup"
        else [],
        "total_songs": len(song_ids),
        "of_songs": of_count,
        "notable": notable_count,
        "notable_ratio": round(notable_count / max(len(song_ids), 1), 4),
        "first_year": first_year,
        "last_year": last_year,
        "velocity": round(velocity, 4),
        "genres": dict(genres.most_common(10)),
        "genre_breadth": len(genres),
        "influence_out": inf_out,
        "influence_in": inf_in,
        "collaborators": collaborators,
        "labels": labels,
        "role_diversity": role_diversity,
        "yearly": year_map,
        "dimensions_raw": {
            "momentum": round(momentum, 4),
            "quality_consistency": round(quality_consistency, 4),
            "influence_centrality": round(influence_centrality, 4),
            "cross_genre_transfer": round(cross_genre_transfer, 4),
            "collaboration_leverage": round(collaboration_leverage, 4),
            "industry_support": round(industry_support, 4),
        },
    }


# ------------------------------------------------------------------
# Build profile universe
# ------------------------------------------------------------------
of_entities = set()
for e in data["links"]:
    if e.get("Edge Type") in CREATIVE_ROLES and e["target"] in of_song_ids:
        src = id2node.get(e["source"], {})
        if src.get("Node Type") in ("Person", "MusicalGroup"):
            of_entities.add(e["source"])


def find_entity(name, prefer_type=None):
    candidates = [n for n in data["nodes"] if n.get("name") == name]
    if prefer_type:
        typed = [n for n in candidates if n.get("Node Type") == prefer_type]
        if typed:
            candidates = typed
    best_id = None
    best_count = -1
    for c in candidates:
        cnt = len(entity_songs.get(c["id"], set()))
        if cnt > best_count:
            best_count = cnt
            best_id = c["id"]
    return best_id


COMPARISON_IDS = {
    "Sailor Shift": find_entity("Sailor Shift"),
    "Drowned Harbor": find_entity("Drowned Harbor", "MusicalGroup"),
    "Orla Seabloom": find_entity("Orla Seabloom"),
}

all_profiles = {}
for eid in of_entities:
    if is_redundant_member(eid):
        continue
    if eid in hidden_band_skip:
        continue
    prof = build_profile(eid)
    if prof and prof["total_songs"] > 0:
        all_profiles[eid] = prof

# Ensure comparisons are present
for _, eid in COMPARISON_IDS.items():
    if eid and eid not in all_profiles:
        prof = build_profile(eid)
        if prof and prof["total_songs"] > 0:
            all_profiles[eid] = prof

# Resolve duplicate names for stable display
name_counter = Counter(p["name"] for p in all_profiles.values())
for eid, p in all_profiles.items():
    p["display_name"] = p["name"] if name_counter[p["name"]] == 1 else f"{p['name']} (#{eid})"


# ------------------------------------------------------------------
# Scoring model
# ------------------------------------------------------------------
DIMENSIONS = [
    {"key": "momentum", "label": "Momentum", "weight": 1.4},
    {"key": "quality_consistency", "label": "Quality Consistency", "weight": 1.2},
    {"key": "influence_centrality", "label": "Influence Centrality", "weight": 1.5},
    {"key": "cross_genre_transfer", "label": "Cross-Genre Transfer", "weight": 1.0},
    {"key": "collaboration_leverage", "label": "Collaboration Leverage", "weight": 1.0},
    {"key": "industry_support", "label": "Industry Support", "weight": 0.9},
]

comp_ids = set(v for v in COMPARISON_IDS.values() if v is not None)
ivy_echos_id = find_entity("Ivy Echos", "MusicalGroup")
exclude_ids = set(comp_ids)
if ivy_echos_id:
    exclude_ids.add(ivy_echos_id)
    # also exclude members tied to that group
    for member in group_members.get(ivy_echos_id, []):
        exclude_ids.add(member)

funnel_stages = []
funnel_stages.append(
    {
        "label": "OF-connected artists",
        "count": len(all_profiles),
        "desc": "All artists with at least one Oceanus Folk collaboration",
    }
)

candidates = []
for eid, p in all_profiles.items():
    if eid in exclude_ids:
        continue
    if p["total_songs"] < 2:
        continue
    if p["of_songs"] < 1:
        continue
    candidates.append((eid, p))

funnel_stages.append(
    {
        "label": "Eligible pool",
        "count": len(candidates),
        "desc": "At least 2 songs and at least 1 Oceanus Folk song",
    }
)

years_last = [int(p["last_year"]) for _, p in candidates if str(p["last_year"]).isdigit()]
year_min = min(years_last) if years_last else 1990
year_max = max(years_last) if years_last else 2040


def robust_max(values):
    if not values:
        return 1.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1)))
    v = sorted_vals[idx]
    return v if v > 0 else max(sorted_vals[-1], 1.0)


maxima = {}
for d in DIMENSIONS:
    k = d["key"]
    maxima[k] = robust_max([p["dimensions_raw"][k] for _, p in candidates])


def norm_val(v, k):
    m = maxima.get(k, 1.0)
    return max(0.0, min(v / max(m, 1e-9), 1.0))


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (mag_a * mag_b)


ranked = []
for eid, p in candidates:
    of_commitment = p["of_songs"] / max(p["total_songs"], 1)
    recency = 0.0
    if str(p["last_year"]).isdigit():
        recency = (int(p["last_year"]) - year_min) / max(year_max - year_min, 1)
    recency = max(0.0, min(recency, 1.0))

    dim_norm = {}
    dim_contrib = {}
    core_score = 0.0
    for d in DIMENSIONS:
        k = d["key"]
        n = norm_val(p["dimensions_raw"][k], k)
        c = n * d["weight"]
        dim_norm[k] = round(n, 4)
        dim_contrib[k] = round(c, 4)
        core_score += c

    # 5-year readiness is kept explicit for forecast horizon
    five_year_readiness = 0.55 * recency + 0.45 * (of_commitment ** 1.35)
    final_score = core_score * 0.78 + five_year_readiness * 1.22

    ranked.append(
        {
            "eid": eid,
            "name": p["display_name"],
            "score": round(final_score, 4),
            "core_score": round(core_score, 4),
            "five_year_readiness": round(five_year_readiness, 4),
            "of_commitment": round(of_commitment, 4),
            "recency": round(recency, 4),
            "selected": False,
            "rejection_reason": None,
            "profile": p,
            "dim_norm": dim_norm,
            "dim_contrib": dim_contrib,
        }
    )

ranked.sort(key=lambda x: x["score"], reverse=True)
funnel_stages.append(
    {
        "label": "Ranked by model",
        "count": len(ranked),
        "desc": "Weighted 6-dimension score + five-year readiness",
    }
)

# Select top-3 with light diversity controls
selected_eids = []
picked_songs = set()
picked_collabs = set()
for i, item in enumerate(ranked):
    p = item["profile"]
    eid = item["eid"]
    my_songs = entity_songs.get(eid, set())
    if len(selected_eids) >= 3:
        item["rejection_reason"] = "Top 3 already selected"
        continue
    if picked_songs and (my_songs & picked_songs):
        item["rejection_reason"] = "Shares songs with higher-ranked selected artist"
        continue
    if eid in picked_collabs:
        item["rejection_reason"] = "Direct collaborator with selected artist"
        continue

    item["selected"] = True
    selected_eids.append(eid)
    picked_songs.update(my_songs)
    picked_collabs.update(entity_collaborators.get(eid, set()))

funnel_stages.append(
    {"label": "Final predictions", "count": len(selected_eids), "desc": "Top 3 after diversity controls (song/collab overlap)"}
)

# Archetype mapping by cosine similarity against comparison artists
comparison_profiles = {name: all_profiles[eid] for name, eid in COMPARISON_IDS.items() if eid in all_profiles}
archetype_map = {
    "Sailor Shift": "Next Superstar",
    "Drowned Harbor": "Next Pioneer",
    "Orla Seabloom": "Next Rapid Riser",
}
taken_arch = set()
archetype_labels = {}
archetype_mirrors = {}

for eid in selected_eids:
    p = all_profiles[eid]
    p_vec = [norm_val(p["dimensions_raw"][d["key"]], d["key"]) for d in DIMENSIONS]
    sims = []
    for cname, cp in comparison_profiles.items():
        c_vec = [norm_val(cp["dimensions_raw"][d["key"]], d["key"]) for d in DIMENSIONS]
        sims.append((cname, cosine_sim(p_vec, c_vec)))
    sims.sort(key=lambda x: x[1], reverse=True)

    for cname, _ in sims:
        if cname not in taken_arch:
            taken_arch.add(cname)
            archetype_labels[p["display_name"]] = archetype_map.get(cname, "Next Rising Star")
            archetype_mirrors[p["display_name"]] = comparison_profiles[cname]["display_name"]
            break


# Build prediction explanations
prediction_explanations = {}
name_to_ranked = {r["eid"]: r for r in ranked}
for rank_idx, r in enumerate(ranked, start=1):
    if not r["selected"]:
        continue
    p = r["profile"]
    top_strengths = sorted(r["dim_contrib"].items(), key=lambda x: x[1], reverse=True)[:3]
    risks = []
    if r["of_commitment"] < 0.45:
        risks.append("Low Oceanus Folk commitment may limit genre-specific breakout potential.")
    if r["five_year_readiness"] < 0.5:
        risks.append("Recent momentum is moderate; breakout may take longer than 5 years.")
    if r["dim_norm"]["quality_consistency"] < 0.45:
        risks.append("Quality signal is volatile across active years.")
    if not risks:
        risks.append("No single weak dimension; monitor whether output remains sustainable.")

    outlook = "Moderate 5-year breakout probability"
    if r["five_year_readiness"] >= 0.72:
        outlook = "High 5-year breakout probability"
    elif r["five_year_readiness"] >= 0.55:
        outlook = "Moderate-to-high 5-year breakout probability"

    prediction_explanations[p["display_name"]] = {
        "rank": rank_idx,
        "score": r["score"],
        "core_score": r["core_score"],
        "five_year_readiness": r["five_year_readiness"],
        "strengths": [{"key": k, "label": next(d["label"] for d in DIMENSIONS if d["key"] == k), "contribution": v} for k, v in top_strengths],
        "risks": risks[:2],
        "five_year_outlook": outlook,
    }


# ------------------------------------------------------------------
# Exports for views
# ------------------------------------------------------------------
profiles_export = {p["display_name"]: p for p in all_profiles.values()}
comparison_list = [all_profiles[eid]["display_name"] for eid in COMPARISON_IDS.values() if eid in all_profiles]
prediction_list = [all_profiles[eid]["display_name"] for eid in selected_eids if eid in all_profiles]

ranking_rows = []
for i, r in enumerate(ranked[:30], start=1):
    p = r["profile"]
    ranking_rows.append(
        {
            "rank": i,
            "name": p["display_name"],
            "eid": r["eid"],
            "score": r["score"],
            "core_score": r["core_score"],
            "five_year_readiness": r["five_year_readiness"],
            "selected": r["selected"],
            "rejection_reason": r["rejection_reason"],
            "of_songs": p["of_songs"],
            "total_songs": p["total_songs"],
            "notable_ratio": p["notable_ratio"],
            "influence_out": p["influence_out"],
            "influence_in": p["influence_in"],
            "first_year": p["first_year"],
            "last_year": p["last_year"],
            "dim_norm": r["dim_norm"],
            "dim_contrib": r["dim_contrib"],
            "of_commitment": r["of_commitment"],
            "recency": r["recency"],
        }
    )

# Timeline / chord context
of_timeline = defaultdict(lambda: {"total": 0, "notable": 0, "artists": set(), "new_artists": set()})
for n in data["nodes"]:
    if n.get("Node Type") == "Song" and n.get("genre") == "Oceanus Folk" and n.get("release_date"):
        y = n["release_date"]
        of_timeline[y]["total"] += 1
        of_timeline[y]["notable"] += 1 if n.get("notable") else 0
        for eid in song_to_entities.get(n["id"], set()):
            of_timeline[y]["artists"].add(eid)

seen = set()
timeline_rows = []
for y in sorted(of_timeline.keys()):
    artists = of_timeline[y]["artists"]
    new_artists = artists - seen
    seen.update(artists)
    timeline_rows.append(
        {
            "year": y,
            "total": of_timeline[y]["total"],
            "notable": of_timeline[y]["notable"],
            "artist_count": len(artists),
            "new_artists": len(new_artists),
            "returning_artists": len(artists) - len(new_artists),
        }
    )

genre_flow_count = defaultdict(int)
for e in influence_edges:
    sg = e["source_genre"]
    tg = e["target_genre"]
    if sg and tg:
        genre_flow_count[(sg, tg)] += 1
genre_flow = [{"source": s, "target": t, "value": v} for (s, t), v in genre_flow_count.items() if v >= 3]
genre_flow.sort(key=lambda x: -x["value"])

genre_influence_count = defaultdict(int)
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg and tg:
        genre_influence_count[sg] += 1
        genre_influence_count[tg] += 1
top_genres = [g for g, _ in sorted(genre_influence_count.items(), key=lambda x: -x[1])[:12]]
genre_idx = {g: i for i, g in enumerate(top_genres)}
chord_matrix = [[0] * len(top_genres) for _ in range(len(top_genres))]
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg in genre_idx and tg in genre_idx:
        chord_matrix[genre_idx[sg]][genre_idx[tg]] += 1

# Network context (keep influence network)
network_eids = set()
for eid, p in all_profiles.items():
    if p["of_songs"] > 0 or p["influence_out"] > 3 or p["influence_in"] > 3:
        network_eids.add(eid)

top_network = sorted(
    [(eid, all_profiles[eid]) for eid in network_eids if eid in all_profiles],
    key=lambda x: x[1]["influence_out"] + x[1]["influence_in"] + x[1]["of_songs"] * 2,
    reverse=True,
)[:80]
top_eids = {eid for eid, _ in top_network}
for eid in list(COMPARISON_IDS.values()) + list(selected_eids):
    if eid and eid in all_profiles:
        top_eids.add(eid)

comp_set = set(COMPARISON_IDS.values())
pred_set = set(selected_eids)

network_nodes = []
for eid in top_eids:
    p = all_profiles[eid]
    network_nodes.append(
        {
            "id": p["display_name"],
            "eid": eid,
            "songs": p["total_songs"],
            "of_songs": p["of_songs"],
            "influence_out": p["influence_out"],
            "influence_in": p["influence_in"],
            "notable": p["notable"],
            "type": p["type"],
            "genres": p["genres"],
            "first_year": p["first_year"],
            "last_year": p["last_year"],
            "is_comparison": eid in comp_set,
            "is_prediction": eid in pred_set,
        }
    )

artist_influence_typed = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for e in influence_edges:
    for se in e["source_entities"]:
        for te in e["target_entities"]:
            if se != te and se in top_eids and te in top_eids and se in all_profiles and te in all_profiles:
                sname = all_profiles[se]["display_name"]
                tname = all_profiles[te]["display_name"]
                etype = e.get("type", "Unknown")
                artist_influence_typed[sname][tname][etype] += 1

network_links = []
for src, targets in artist_influence_typed.items():
    for tgt, types in targets.items():
        weight = sum(types.values())
        dominant_type = max(types.items(), key=lambda x: x[1])[0] if types else "Unknown"
        network_links.append(
            {
                "source": src,
                "target": tgt,
                "weight": weight,
                "type": dominant_type,
                "types": dict(types),
            }
        )

# Precompute deterministic force-like layout
import random

random.seed(42)
W_NET, H_NET = 740, 560
positions = {}
for n in network_nodes:
    positions[n["id"]] = [random.uniform(50, W_NET - 50), random.uniform(50, H_NET - 50)]

for _ in range(500):
    forces = {n["id"]: [0.0, 0.0] for n in network_nodes}
    ids = [n["id"] for n in network_nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dx = positions[ids[i]][0] - positions[ids[j]][0]
            dy = positions[ids[i]][1] - positions[ids[j]][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            rep = 2600 / (dist * dist)
            fx = dx / dist * rep
            fy = dy / dist * rep
            forces[ids[i]][0] += fx
            forces[ids[i]][1] += fy
            forces[ids[j]][0] -= fx
            forces[ids[j]][1] -= fy
    for lk in network_links:
        s = lk["source"]
        t = lk["target"]
        if s in positions and t in positions:
            dx = positions[t][0] - positions[s][0]
            dy = positions[t][1] - positions[s][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            spring = (dist - 80) * 0.01
            forces[s][0] += dx / dist * spring
            forces[s][1] += dy / dist * spring
            forces[t][0] -= dx / dist * spring
            forces[t][1] -= dy / dist * spring
    for nid in ids:
        # center gravity
        forces[nid][0] += (W_NET / 2 - positions[nid][0]) * 0.008
        forces[nid][1] += (H_NET / 2 - positions[nid][1]) * 0.008
    for nid in ids:
        positions[nid][0] = max(25, min(W_NET - 25, positions[nid][0] + forces[nid][0] * 0.3))
        positions[nid][1] = max(25, min(H_NET - 25, positions[nid][1] + forces[nid][1] * 0.3))

for n in network_nodes:
    n["x"] = round(positions[n["id"]][0], 1)
    n["y"] = round(positions[n["id"]][1], 1)

# ------------------------------------------------------------------
# Write output files
# ------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "artists.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "profiles": profiles_export,
            "comparison": comparison_list,
            "predictions": prediction_list,
            "dimension_model": {
                "name": "rising-star-6d-core-plus-readiness",
                "dimensions": DIMENSIONS,
                "readiness_formula": "0.55 * recency + 0.45 * (of_commitment^1.35)",
                "final_formula": "0.78 * core_score + 1.22 * five_year_readiness",
                "notes": "Core dimensions are p95-clipped normalized; readiness keeps 5-year horizon explicit.",
            },
            "selection_funnel": {
                "stages": funnel_stages,
                "ranking": ranking_rows,
            },
            "archetype_labels": archetype_labels,
            "archetype_mirrors": archetype_mirrors,
            "prediction_explanations": prediction_explanations,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "oceanus_folk": timeline_rows,
            "genre_flow": genre_flow[:50],
            "chord_genres": top_genres,
            "chord_matrix": chord_matrix,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(OUT, "network.json"), "w", encoding="utf-8") as f:
    json.dump({"nodes": network_nodes, "links": network_links}, f, ensure_ascii=False, indent=2)

print("=== Rising Star Refactor Export Summary ===")
print(f"Profiles: {len(profiles_export)}")
print(f"Comparison artists: {comparison_list}")
print(f"Predictions: {prediction_list}")
print(f"Ranking rows: {len(ranking_rows)}")
print(f"Timeline years: {len(timeline_rows)}")
print(f"Network: {len(network_nodes)} nodes, {len(network_links)} links")
"""
Preprocess MC1_graph.json for the Rising Stars dashboard.
Fixes: group/member deduplication, song-level influence counting,
normalized scoring aligned with 6-metric framework.
Outputs: artists.json, timeline.json, network.json into viz/data/
"""
import json
import os
import math
import statistics
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

with open(os.path.join(BASE, "MC1_graph.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

id2node = {n["id"]: n for n in data["nodes"]}

INFLUENCE_TYPES = {"InStyleOf", "InterpolatesFrom", "CoverOf", "DirectlySamples", "LyricalReferenceTo"}
CREATIVE_ROLES = {"PerformerOf", "ComposerOf", "LyricistOf", "ProducerOf"}

of_song_ids = {n["id"] for n in data["nodes"] if n.get("genre") == "Oceanus Folk"}

# ── Build mappings by NODE ID ───────────────────────────────────────
entity_songs = defaultdict(set)        # entity_id -> set of song_ids
entity_roles = defaultdict(lambda: defaultdict(set))  # entity_id -> role -> set of song_ids
song_to_entities = defaultdict(set)    # song_id -> set of entity_ids
group_members = defaultdict(list)      # group_id -> list of member_ids
member_of_group = defaultdict(set)     # person_id -> set of group_ids

for e in data["links"]:
    etype = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})

    if etype in CREATIVE_ROLES:
        if src.get("Node Type") in ("Person", "MusicalGroup") and tgt.get("Node Type") == "Song":
            entity_songs[e["source"]].add(e["target"])
            entity_roles[e["source"]][etype].add(e["target"])
            song_to_entities[e["target"]].add(e["source"])

    if etype == "MemberOf":
        group_members[e["target"]].append(e["source"])
        member_of_group[e["source"]].add(e["target"])

# ── Label associations ──────────────────────────────────────────────
entity_labels = defaultdict(set)
for e in data["links"]:
    etype = e.get("Edge Type", "")
    src = id2node.get(e["source"], {})
    tgt = id2node.get(e["target"], {})
    if etype in ("RecordedBy", "DistributedBy"):
        if src.get("Node Type") == "Song" and tgt.get("Node Type") == "RecordLabel":
            for eid in song_to_entities.get(e["source"], set()):
                entity_labels[eid].add(tgt["name"])

# ── Song-level influence (deduplicated) ─────────────────────────────
# Count influence per SONG, not per entity-song pair
song_inf_out = defaultdict(int)  # song_id -> count of songs it influenced
song_inf_in = defaultdict(int)   # song_id -> count of songs that influenced it
influence_edges = []

for e in data["links"]:
    if e.get("Edge Type") in INFLUENCE_TYPES:
        song_inf_out[e["source"]] += 1
        song_inf_in[e["target"]] += 1

        src_song = id2node.get(e["source"], {})
        tgt_song = id2node.get(e["target"], {})
        src_entities = song_to_entities.get(e["source"], set())
        tgt_entities = song_to_entities.get(e["target"], set())

        if src_entities and tgt_entities:
            influence_edges.append({
                "source_song": src_song.get("name", ""),
                "target_song": tgt_song.get("name", ""),
                "source_entities": list(src_entities),
                "target_entities": list(tgt_entities),
                "type": e.get("Edge Type"),
                "source_genre": src_song.get("genre", ""),
                "target_genre": tgt_song.get("genre", ""),
            })

# Entity influence = sum of its songs' influence (no double counting)
def entity_influence(eid):
    songs = entity_songs.get(eid, set())
    out = sum(song_inf_out.get(sid, 0) for sid in songs)
    inp = sum(song_inf_in.get(sid, 0) for sid in songs)
    return out, inp

# ── Collaboration ───────────────────────────────────────────────────
entity_collaborators = defaultdict(set)
for song_id, entities in song_to_entities.items():
    for eid in entities:
        entity_collaborators[eid].update(entities - {eid})

# ── Group/member deduplication ──────────────────────────────────────
# If a person shares >80% songs with their group, mark as redundant
def is_redundant_member(eid):
    groups = member_of_group.get(eid, set())
    if not groups:
        return False
    my_songs = entity_songs.get(eid, set())
    if not my_songs:
        return False
    for gid in groups:
        g_songs = entity_songs.get(gid, set())
        if g_songs and len(my_songs & g_songs) / len(my_songs) > 0.8:
            return True
    return False

# Also detect "hidden bands" — people who share >80% songs but aren't in a formal group
def find_hidden_band_representatives():
    """Among people not in groups, find clusters sharing >80% songs and keep only one."""
    skip = set()
    person_ids = [eid for eid in entity_songs
                  if id2node[eid].get("Node Type") == "Person"
                  and not member_of_group.get(eid)
                  and len(entity_songs[eid]) >= 2]
    person_ids.sort()
    for i, a in enumerate(person_ids):
        if a in skip:
            continue
        sa = entity_songs[a]
        for b in person_ids[i+1:]:
            if b in skip:
                continue
            sb = entity_songs[b]
            if sa == sb:
                skip.add(b)  # keep a, skip b
    return skip

hidden_band_skip = find_hidden_band_representatives()

# ── Build profiles by entity ID ─────────────────────────────────────
def build_profile(eid):
    node = id2node[eid]
    song_ids = entity_songs.get(eid, set())
    if not song_ids:
        return None

    songs = [id2node[sid] for sid in song_ids]
    of_count = sum(1 for sid in song_ids if sid in of_song_ids)
    dates = sorted([s.get("release_date", "9999") for s in songs])
    notable_count = sum(1 for s in songs if s.get("notable"))
    genres = Counter(s.get("genre", "Unknown") for s in songs)

    by_year = defaultdict(list)
    for s in songs:
        by_year[s.get("release_date", "?")].append({
            "name": s.get("name", ""),
            "genre": s.get("genre", ""),
            "notable": s.get("notable", False),
        })

    first_year = dates[0] if dates else "?"
    last_year = dates[-1] if dates else "?"

    year_counts = Counter(s.get("release_date", "?") for s in songs)
    active_years = [y for y in year_counts if y != "?"]
    if len(active_years) >= 2:
        span = int(max(active_years)) - int(min(active_years)) + 1
        velocity = len(songs) / max(span, 1)
    else:
        velocity = len(songs)

    members = []
    if node.get("Node Type") == "MusicalGroup":
        members = [id2node[mid].get("name", "?") for mid in group_members.get(eid, [])]

    inf_out, inf_in = entity_influence(eid)
    collaborators = len(entity_collaborators.get(eid, set()))
    labels = len(entity_labels.get(eid, set()))
    role_diversity = len(entity_roles[eid])

    # Advanced dimensions for the evaluation model.
    active_year_ints = sorted(int(y) for y in active_years if str(y).isdigit())
    if active_year_ints:
        end_y = active_year_ints[-1]
        start_y = active_year_ints[0]
        recent_sum = sum(year_counts.get(str(y), 0) for y in range(end_y - 2, end_y + 1))
        early_sum = sum(year_counts.get(str(y), 0) for y in range(start_y, start_y + 3))
        recent_velocity = recent_sum / 3.0
        early_velocity = early_sum / 3.0
        trajectory_acceleration = max(0.0, recent_velocity - early_velocity)
    else:
        trajectory_acceleration = 0.0

    yearly_quality = []
    for y in active_years:
        songs_in_year = by_year.get(y, [])
        if songs_in_year:
            yearly_quality.append(
                sum(1 for s in songs_in_year if s.get("notable")) / len(songs_in_year)
            )
    if len(yearly_quality) <= 1:
        quality_consistency = 1.0 if yearly_quality else 0.0
    else:
        quality_consistency = max(
            0.0, 1.0 - min(statistics.pstdev(yearly_quality), 1.0)
        )

    total_genre_songs = sum(genres.values()) or 1
    probs = [c / total_genre_songs for c in genres.values() if c > 0]
    if len(probs) <= 1:
        genre_diversity = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in probs)
        genre_diversity = entropy / math.log(len(probs))

    influence_centrality = inf_out + 0.7 * inf_in
    influence_balance = 1.0 - abs(inf_out - inf_in) / (inf_out + inf_in + 1.0)
    collaboration_leverage = collaborators * (1.0 + 0.15 * role_diversity)
    industry_support = labels + 0.25 * role_diversity

    return {
        "id": eid,
        "name": node["name"],
        "type": node.get("Node Type", "Person"),
        "members": members,
        "total_songs": len(songs),
        "of_songs": of_count,
        "notable": notable_count,
        "notable_ratio": round(notable_count / len(songs), 3) if songs else 0,
        "first_year": first_year,
        "last_year": last_year,
        "velocity": round(velocity, 3),
        "genres": dict(genres.most_common(10)),
        "genre_breadth": len(genres),
        "influence_out": inf_out,
        "influence_in": inf_in,
        "collaborators": collaborators,
        "labels": labels,
        "by_year": {k: v for k, v in sorted(by_year.items())},
        "roles": {r: len(sids) for r, sids in entity_roles[eid].items()},
        "role_diversity": role_diversity,
        "trajectory_acceleration": round(trajectory_acceleration, 4),
        "quality_consistency": round(quality_consistency, 4),
        "influence_centrality": round(influence_centrality, 4),
        "genre_diversity": round(genre_diversity, 4),
        "influence_balance": round(influence_balance, 4),
        "collaboration_leverage": round(collaboration_leverage, 4),
        "industry_support": round(industry_support, 4),
    }

# ── Collect all entities with OF involvement ────────────────────────
of_entities = set()
for e in data["links"]:
    if e.get("Edge Type") in CREATIVE_ROLES and e["target"] in of_song_ids:
        src = id2node.get(e["source"], {})
        if src.get("Node Type") in ("Person", "MusicalGroup"):
            of_entities.add(e["source"])

# Hardcoded comparison artists
COMPARISON_NAMES = ["Sailor Shift", "Drowned Harbor", "Orla Seabloom"]

def find_entity(name, prefer_type=None):
    candidates = [n for n in data["nodes"] if n.get("name") == name]
    if prefer_type:
        typed = [c for c in candidates if c.get("Node Type") == prefer_type]
        if typed:
            candidates = typed
    best_id, best_songs = None, -1
    for c in candidates:
        ns = len(entity_songs.get(c["id"], set()))
        if ns > best_songs:
            best_songs = ns
            best_id = c["id"]
    return best_id

COMPARISON_IDS = {
    "Sailor Shift": find_entity("Sailor Shift"),
    "Drowned Harbor": find_entity("Drowned Harbor", "MusicalGroup"),
    "Orla Seabloom": find_entity("Orla Seabloom"),
}

# Build all profiles (excluding redundant members and hidden-band duplicates)
all_profiles = {}
for eid in of_entities:
    if is_redundant_member(eid):
        continue
    if eid in hidden_band_skip:
        continue
    p = build_profile(eid)
    if p and p["total_songs"] > 0:
        all_profiles[eid] = p

# Ensure comparison artists are always included
for name, eid in COMPARISON_IDS.items():
    if eid and eid not in all_profiles:
        p = build_profile(eid)
        if p and p["total_songs"] > 0:
            all_profiles[eid] = p

# ══════════════════════════════════════════════════════════════════
# PREDICTION SELECTION — Soft scoring, no hard cutoffs
# ══════════════════════════════════════════════════════════════════
#
# Logic (the bridge from Task 3a → 3b):
#   3a: Compare 3 artists → extract 6-metric framework
#   3b: Apply that SAME framework to ALL OF-connected artists,
#       with OF-commitment and recency as continuous score components
#       (not hard cutoffs). Only structural filter: ≥2 songs.
#       Rank by composite score, pick top 3 with diversity constraint.
# ══════════════════════════════════════════════════════════════════

comp_ids = set(COMPARISON_IDS.values())
# Exclude comparison artists, Ivy Echos, and their members
ivy_echos_id = find_entity("Ivy Echos", "MusicalGroup")
exclude_ids = set(comp_ids)
if ivy_echos_id:
    exclude_ids.add(ivy_echos_id)
for comp_eid in comp_ids:
    for e in data["links"]:
        if e.get("Edge Type") == "MemberOf" and e["source"] == comp_eid:
            exclude_ids.add(e["target"])
            for e2 in data["links"]:
                if e2.get("Edge Type") == "MemberOf" and e2["target"] == e["target"]:
                    exclude_ids.add(e2["source"])

# ── Funnel tracking ───────────────────────────────────────────────
funnel_stages = []
funnel_stages.append({"label": "OF-connected artists", "count": len(all_profiles),
                       "desc": "All artists with at least one Oceanus Folk song"})

# ── Step 1: Structural filter only — ≥2 songs, not comparison/redundant ──
rising_candidates = []
for eid, p in all_profiles.items():
    if eid in exclude_ids:
        continue
    if p["total_songs"] < 2:
        continue
    of_ratio = p["of_songs"] / p["total_songs"] if p["total_songs"] > 0 else 0
    rising_candidates.append((eid, p, of_ratio))

funnel_stages.append({"label": "≥2 songs, not comparison/redundant", "count": len(rising_candidates),
                       "desc": "Minimum statistical significance, exclude known artists"})

# ── Step 2: Score using an 8-dimension model ──
framework_keys = [
    "trajectory_acceleration",
    "quality_consistency",
    "influence_centrality",
    "genre_diversity",
    "collaboration_leverage",
    "industry_support",
    "recency",
    "of_commitment",
]
model_dimensions = [
    {"key": "trajectory_acceleration", "label": "Trajectory Acceleration", "weight": 1.35},
    {"key": "quality_consistency", "label": "Quality Consistency", "weight": 1.05},
    {"key": "influence_centrality", "label": "Influence Centrality", "weight": 1.55},
    {"key": "genre_diversity", "label": "Genre Diversity", "weight": 1.0},
    {"key": "collaboration_leverage", "label": "Collaboration Leverage", "weight": 1.1},
    {"key": "industry_support", "label": "Industry Support", "weight": 0.95},
    {"key": "recency", "label": "Recency", "weight": 1.1},
    {"key": "of_commitment", "label": "OF Commitment", "weight": 1.9},
]
dim_label_map = {d["key"]: d["label"] for d in model_dimensions}

robust_max = {}
for k in ["trajectory_acceleration", "influence_centrality", "collaboration_leverage", "industry_support"]:
    vals = sorted(p[k] for _, p, _ in rising_candidates)
    if not vals:
        robust_max[k] = 1
        continue
    idx = int(len(vals) * 0.95)
    robust_max[k] = vals[idx] if vals[idx] > 0 else max(vals[-1], 1)

# Determine the data's time range for recency normalization
all_last_years = []
for _, p, _ in rising_candidates:
    try:
        all_last_years.append(int(p["last_year"]))
    except ValueError:
        pass
year_min = min(all_last_years) if all_last_years else 1990
year_max = max(all_last_years) if all_last_years else 2040

def framework_score(p, of_ratio):
    """Composite score from 8 normalized dimensions with explicit contributions."""
    try:
        yr = int(p["last_year"])
        recency = (yr - year_min) / max(year_max - year_min, 1)
    except ValueError:
        recency = 0

    of_commitment_score = of_ratio ** 1.8
    metric_scores = {
        "trajectory_acceleration": min(
            p["trajectory_acceleration"] / max(robust_max["trajectory_acceleration"], 1e-9), 1.0
        ),
        "quality_consistency": max(0.0, min(p["quality_consistency"], 1.0)),
        "influence_centrality": min(
            p["influence_centrality"] / max(robust_max["influence_centrality"], 1e-9), 1.0
        ),
        "genre_diversity": max(0.0, min(p["genre_diversity"], 1.0)),
        "collaboration_leverage": min(
            p["collaboration_leverage"] / max(robust_max["collaboration_leverage"], 1e-9), 1.0
        ),
        "industry_support": min(
            p["industry_support"] / max(robust_max["industry_support"], 1e-9), 1.0
        ),
        "recency": max(0.0, min(recency, 1.0)),
        "of_commitment": max(0.0, min(of_commitment_score, 1.0)),
        "of_ratio_raw": max(0.0, min(of_ratio, 1.0)),
    }

    weighted_contrib = {}
    total = 0.0
    for dim in model_dimensions:
        k = dim["key"]
        c = metric_scores[k] * dim["weight"]
        weighted_contrib[k] = round(c, 4)
        total += c
    metric_scores = {k: round(v, 4) for k, v in metric_scores.items()}
    return round(total, 4), metric_scores, weighted_contrib

scored_candidates = []
for eid, p, of_ratio in rising_candidates:
    score, metric_scores, weighted_contrib = framework_score(p, of_ratio)
    scored_candidates.append((eid, p, of_ratio, score, metric_scores, weighted_contrib))

scored_candidates.sort(key=lambda x: x[3], reverse=True)

# Show top candidates with meaningful scores (score > 0 effectively)
top_count = sum(1 for _, _, _, s, _, _ in scored_candidates if s >= scored_candidates[0][3] * 0.3)
funnel_stages.append({"label": "Scored by 8-dimension model", "count": len(scored_candidates),
                       "desc": f"All candidates scored; top {min(top_count, 20)} shown in ranking"})

print("=== TOP CANDIDATES (8-dimension scoring) ===")
for eid, p, of_ratio, score, ms, wc in scored_candidates[:25]:
    safe_name = p["name"].encode("ascii", "replace").decode("ascii")
    print(f"  {safe_name:25s} {p['type']:14s} songs={p['total_songs']:2d} of={p['of_songs']:2d}({of_ratio:.0%}) "
          f"notable={p['notable']:2d}({p['notable_ratio']:.0%}) inf={p['influence_out']:2d}/{p['influence_in']:2d} "
          f"genres={p['genre_breadth']} score={score:.3f} {p['first_year']}-{p['last_year']}")

# ── Step 3: Pick top 3 with diversity (no song/collaborator overlap) ──
# Track rejection reasons for every candidate in the top pool
predictions = []
picked_songs = set()
picked_collabs = set()
candidate_status = []  # exported to JSON for the ranking chart

for rank, (eid, p, of_ratio, score, ms, wc) in enumerate(scored_candidates):
    my_songs = entity_songs.get(eid, set())
    status = {"name": p["name"], "eid": eid, "type": p["type"], "rank": rank + 1,
              "score": score, "of_ratio": round(of_ratio, 3),
              "metric_scores": ms,
              "weighted_contrib": wc,
              "raw_metrics": {
                  "trajectory_acceleration": p["trajectory_acceleration"],
                  "quality_consistency": p["quality_consistency"],
                  "influence_centrality": p["influence_centrality"],
                  "genre_diversity": p["genre_diversity"],
                  "collaboration_leverage": p["collaboration_leverage"],
                  "industry_support": p["industry_support"],
                  "recency": ms["recency"],
                  "of_commitment": ms["of_commitment"],
                  "of_ratio_raw": ms["of_ratio_raw"],
              },
              "of_songs": p["of_songs"], "total_songs": p["total_songs"],
              "influence_in": p["influence_in"], "influence_out": p["influence_out"],
              "notable_ratio": p["notable_ratio"],
              "first_year": p["first_year"], "last_year": p["last_year"],
              "selected": False, "rejection_reason": None}

    if len(predictions) >= 3:
        status["rejection_reason"] = f"Already selected 3"
    elif p["of_songs"] < 1:
        status["rejection_reason"] = "No Oceanus Folk songs (genre-alignment gate)"
    elif picked_songs and my_songs and (my_songs & picked_songs):
        status["rejection_reason"] = "Song overlap with higher-ranked selection"
    elif eid in picked_collabs:
        status["rejection_reason"] = "Direct collaborator of a selection"
    else:
        predictions.append(eid)
        picked_songs.update(my_songs)
        picked_collabs.update(entity_collaborators.get(eid, set()))
        status["selected"] = True

    candidate_status.append(status)
    if rank >= 24:
        break  # export top 25 for the ranking chart

funnel_stages.append({"label": "Top 3 selected (diversity filter)", "count": 3,
                       "desc": "Highest-scoring candidates with no song/collaborator overlap"})

# ── Step 4: Assign archetype labels via cosine similarity ──
def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1
    mag_b = math.sqrt(sum(x * x for x in b)) or 1
    return dot / (mag_a * mag_b)

PREDICTION_IDS = {}
ARCHETYPE_LABELS = {}
ARCHETYPE_MIRRORS = {}
taken_archetypes = set()

pred_sims = []
similarity_keys = [
    "trajectory_acceleration",
    "quality_consistency",
    "influence_centrality",
    "genre_diversity",
    "collaboration_leverage",
    "industry_support",
]
for eid in predictions:
    p = all_profiles[eid]
    comp_profiles = {name: all_profiles[cid] for name, cid in COMPARISON_IDS.items() if cid in all_profiles}
    p_vec = [p.get(k, 0) for k in similarity_keys]
    sims = {}
    for comp_name, cp in comp_profiles.items():
        c_vec = [cp.get(k, 0) for k in similarity_keys]
        sims[comp_name] = cosine_sim(p_vec, c_vec)
    pred_sims.append((eid, p, sorted(sims.items(), key=lambda x: -x[1])))

archetype_map = {
    "Sailor Shift": "Next Superstar",
    "Drowned Harbor": "Next Pioneer",
    "Orla Seabloom": "Next Rapid Riser",
}
for eid, p, ranked_sims in pred_sims:
    for comp_name, sim in ranked_sims:
        if comp_name not in taken_archetypes:
            taken_archetypes.add(comp_name)
            PREDICTION_IDS[p["name"]] = eid
            ARCHETYPE_LABELS[p["name"]] = archetype_map[comp_name]
            ARCHETYPE_MIRRORS[p["name"]] = comp_name
            print(f"  Archetype: {p['name']} -> {archetype_map[comp_name]} (mirrors {comp_name}, sim={sim:.3f})")
            break

print(f"\n=== SELECTED PREDICTIONS ===")
for name, eid in PREDICTION_IDS.items():
    p = all_profiles[eid]
    of_pct = p["of_songs"] * 100 // p["total_songs"]
    print(f"  [{ARCHETYPE_LABELS[name]}] {name} (id={eid}): "
          f"{p['total_songs']} songs, {p['of_songs']} OF ({of_pct}%), "
          f"notable={p['notable']}/{p['total_songs']}, inf={p['influence_out']}/{p['influence_in']}, "
          f"{p['first_year']}-{p['last_year']} -- mirrors {ARCHETYPE_MIRRORS[name]}")

# ── Build export profiles ──────────────────────────────────────────
profiles_export = {}
name_count = Counter()
for eid, p in all_profiles.items():
    name_count[p["name"]] += 1

for eid, p in all_profiles.items():
    key = p["name"]
    if name_count[key] > 1:
        key = f"{p['name']} (#{eid})"
        p["display_name"] = key
    else:
        p["display_name"] = p["name"]
    profiles_export[key] = p

comparison_names = {}
for display_name, eid in COMPARISON_IDS.items():
    p = all_profiles.get(eid)
    if p:
        comparison_names[display_name] = p["display_name"]

prediction_names = {}
for display_name, eid in PREDICTION_IDS.items():
    p = all_profiles.get(eid)
    if p:
        prediction_names[display_name] = p["display_name"]

# ── Prediction evidence package (for narrative cards / explainability) ──
selected_status_by_eid = {c["eid"]: c for c in candidate_status if c.get("selected")}
prediction_explanations = {}
for pred_name, eid in PREDICTION_IDS.items():
    p = all_profiles.get(eid)
    status = selected_status_by_eid.get(eid)
    if not p or not status:
        continue

    display_name = p["display_name"]
    contrib = status.get("weighted_contrib", {})
    metric_scores = status.get("metric_scores", {})
    top_strengths = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]

    strengths = [
        {"key": k, "label": dim_label_map.get(k, k), "contribution": round(v, 3)}
        for k, v in top_strengths
    ]

    risks = []
    if metric_scores.get("of_commitment", 0) < 0.5:
        risks.append("Low Oceanus Folk commitment could weaken genre-specific breakout odds.")
    if metric_scores.get("trajectory_acceleration", 0) < 0.25:
        risks.append("Limited recent acceleration may slow near-term growth.")
    if metric_scores.get("quality_consistency", 1) < 0.55:
        risks.append("Quality consistency is volatile across active years.")
    if metric_scores.get("influence_centrality", 0) < 0.25:
        risks.append("Influence centrality remains modest versus top-tier candidates.")
    if not risks:
        risks.append("No major single-dimension weakness; monitor sustainability under higher output pressure.")

    outlook_score = (
        metric_scores.get("trajectory_acceleration", 0) * 0.4
        + metric_scores.get("recency", 0) * 0.35
        + metric_scores.get("of_ratio_raw", 0) * 0.25
    )
    if outlook_score >= 0.72:
        outlook = "High 5-year breakout probability"
    elif outlook_score >= 0.5:
        outlook = "Moderate-to-high 5-year breakout probability"
    else:
        outlook = "Moderate 5-year breakout probability"

    prediction_explanations[display_name] = {
        "rank": status["rank"],
        "score": round(status["score"], 4),
        "strengths": strengths,
        "risks": risks[:2],
        "five_year_outlook": outlook,
    }

# ── Normalize radar metrics ─────────────────────────────────────────
radar_keys = [
    "trajectory_acceleration",
    "quality_consistency",
    "influence_centrality",
    "genre_diversity",
    "collaboration_leverage",
    "industry_support",
]
radar_labels = [
    "Trajectory Acceleration",
    "Quality Consistency",
    "Influence Centrality",
    "Genre Diversity",
    "Collab Leverage",
    "Industry Support",
]

maxvals = {}
for k in radar_keys:
    vals = [p[k] for p in profiles_export.values() if p[k] is not None]
    maxvals[k] = max(vals) if vals else 1

for key, p in profiles_export.items():
    p["radar"] = [round(p[k] / maxvals[k], 4) if maxvals[k] > 0 else 0 for k in radar_keys]

# ── Industry-wide scatter data (ALL OF-connected artists) ──────────
# For the scatter plot showing where predictions stand in the whole industry
comp_set = set(COMPARISON_IDS.values())
pred_set = set(PREDICTION_IDS.values())

industry_scatter = []
for key, p in profiles_export.items():
    role = "comparison" if p["id"] in comp_set else "prediction" if p["id"] in pred_set else "other"
    industry_scatter.append({
        "name": p["display_name"],
        "influence": p["influence_out"] + p["influence_in"],
        "notable_ratio": p["notable_ratio"],
        "of_songs": p["of_songs"],
        "total_songs": p["total_songs"],
        "genre_breadth": p["genre_breadth"],
        "velocity": p["velocity"],
        "collaborators": p["collaborators"],
        "role": role,
    })

# ── Timeline ────────────────────────────────────────────────────────
of_timeline = defaultdict(lambda: {"total": 0, "notable": 0, "artists": set(), "new_artists": set()})
seen_artists = set()
for n in data["nodes"]:
    if n.get("genre") == "Oceanus Folk" and n.get("release_date"):
        y = n["release_date"]
        of_timeline[y]["total"] += 1
        if n.get("notable"):
            of_timeline[y]["notable"] += 1
        for eid in song_to_entities.get(n["id"], set()):
            of_timeline[y]["artists"].add(eid)

sorted_years = sorted(of_timeline.keys())
seen_all = set()
for y in sorted_years:
    artists_this_year = of_timeline[y]["artists"]
    new = artists_this_year - seen_all
    of_timeline[y]["new_artists"] = new
    seen_all.update(artists_this_year)

timeline = []
for y in sorted_years:
    d = of_timeline[y]
    timeline.append({
        "year": y,
        "total": d["total"],
        "notable": d["notable"],
        "artist_count": len(d["artists"]),
        "new_artists": len(d["new_artists"]),
        "returning": len(d["artists"]) - len(d["new_artists"]),
    })

# ── Genre influence flow ────────────────────────────────────────────
genre_flow = defaultdict(int)
for e in influence_edges:
    sg = e["source_genre"]
    tg = e["target_genre"]
    if sg and tg:
        genre_flow[(sg, tg)] += 1

genre_flow_list = [{"source": k[0], "target": k[1], "value": v}
                   for k, v in genre_flow.items() if v >= 3]
genre_flow_list.sort(key=lambda x: -x["value"])

# ── Influence over time ────────────────────────────────────────────
inf_by_year = defaultdict(lambda: {"from_of": 0, "to_of": 0, "within_of": 0})
for e in influence_edges:
    src_genre = e["source_genre"]
    tgt_genre = e["target_genre"]
    src_song_nodes = [n for n in data["nodes"] if n.get("name") == e["source_song"] and n.get("Node Type") == "Song"]
    if src_song_nodes:
        yr = src_song_nodes[0].get("release_date", "?")
        if yr != "?":
            if src_genre == "Oceanus Folk" and tgt_genre == "Oceanus Folk":
                inf_by_year[yr]["within_of"] += 1
            elif src_genre == "Oceanus Folk":
                inf_by_year[yr]["from_of"] += 1
            elif tgt_genre == "Oceanus Folk":
                inf_by_year[yr]["to_of"] += 1

influence_timeline = [{"year": y, **inf_by_year[y]} for y in sorted(inf_by_year.keys())]

# ── Chord matrix ───────────────────────────────────────────────────
genre_influence_count = defaultdict(int)
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg and tg:
        genre_influence_count[sg] += 1
        genre_influence_count[tg] += 1

top_genres = [g for g, _ in sorted(genre_influence_count.items(), key=lambda x: -x[1])[:12]]
genre_idx = {g: i for i, g in enumerate(top_genres)}

chord_matrix = [[0]*len(top_genres) for _ in range(len(top_genres))]
for e in influence_edges:
    sg, tg = e["source_genre"], e["target_genre"]
    if sg in genre_idx and tg in genre_idx:
        chord_matrix[genre_idx[sg]][genre_idx[tg]] += 1

# ── Network (filtered) ─────────────────────────────────────────────
network_eids = set()
for eid, p in all_profiles.items():
    if p["of_songs"] > 0 or p["influence_out"] > 3 or p["influence_in"] > 3:
        network_eids.add(eid)

top_network = sorted(
    [(eid, all_profiles[eid]) for eid in network_eids if eid in all_profiles],
    key=lambda x: x[1]["influence_out"] + x[1]["influence_in"] + x[1]["of_songs"] * 2,
    reverse=True
)[:80]
top_eids = {t[0] for t in top_network}

for eid in list(COMPARISON_IDS.values()) + list(PREDICTION_IDS.values()):
    if eid and eid in all_profiles:
        top_eids.add(eid)

network_nodes = []
for eid in top_eids:
    p = all_profiles[eid]
    network_nodes.append({
        "id": p["display_name"],
        "eid": eid,
        "songs": p["total_songs"],
        "of_songs": p["of_songs"],
        "influence_out": p["influence_out"],
        "influence_in": p["influence_in"],
        "notable": p["notable"],
        "type": p["type"],
        "genres": p["genres"],
        "first_year": p["first_year"],
        "is_comparison": eid in comp_set,
        "is_prediction": eid in pred_set,
    })

# Network edges
artist_influence_typed = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for e in influence_edges:
    for se in e["source_entities"]:
        for te in e["target_entities"]:
            if se != te and se in top_eids and te in top_eids:
                sname = all_profiles[se]["display_name"] if se in all_profiles else str(se)
                tname = all_profiles[te]["display_name"] if te in all_profiles else str(te)
                etype = e.get("type", "Unknown")
                artist_influence_typed[sname][tname][etype] += 1

network_links = []
for src, targets in artist_influence_typed.items():
    for tgt, types in targets.items():
        weight = sum(types.values())
        dom_type = max(types.items(), key=lambda x: x[1])[0] if types else "Unknown"
        network_links.append({
            "source": src, "target": tgt, "weight": weight,
            "type": dom_type, "types": dict(types)
        })

# ── Pre-compute force layout ──────────────────────────────────────
import math, random
random.seed(42)

W_NET, H_NET = 700, 560
positions = {}
for n in network_nodes:
    positions[n["id"]] = [random.uniform(50, W_NET-50), random.uniform(50, H_NET-50)]

link_map = defaultdict(list)
for lk in network_links:
    link_map[lk["source"]].append(lk["target"])
    link_map[lk["target"]].append(lk["source"])

for iteration in range(500):
    alpha = 0.3 * (1 - iteration / 500)
    if alpha < 0.001:
        break
    forces = {n["id"]: [0.0, 0.0] for n in network_nodes}
    ids = [n["id"] for n in network_nodes]
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            dx = positions[ids[i]][0] - positions[ids[j]][0]
            dy = positions[ids[i]][1] - positions[ids[j]][1]
            dist = max(math.sqrt(dx*dx + dy*dy), 1)
            f = 3000 / (dist * dist)
            forces[ids[i]][0] += dx/dist * f * alpha
            forces[ids[i]][1] += dy/dist * f * alpha
            forces[ids[j]][0] -= dx/dist * f * alpha
            forces[ids[j]][1] -= dy/dist * f * alpha
    for lk in network_links:
        s, t = lk["source"], lk["target"]
        if s in positions and t in positions:
            dx = positions[t][0] - positions[s][0]
            dy = positions[t][1] - positions[s][1]
            dist = max(math.sqrt(dx*dx + dy*dy), 1)
            f = (dist - 80) * 0.01 * alpha
            forces[s][0] += dx/dist * f
            forces[s][1] += dy/dist * f
            forces[t][0] -= dx/dist * f
            forces[t][1] -= dy/dist * f
    for nid in ids:
        dx = W_NET/2 - positions[nid][0]
        dy = H_NET/2 - positions[nid][1]
        forces[nid][0] += dx * 0.01 * alpha
        forces[nid][1] += dy * 0.01 * alpha
    for nid in ids:
        positions[nid][0] = max(25, min(W_NET-25, positions[nid][0] + forces[nid][0]))
        positions[nid][1] = max(25, min(H_NET-25, positions[nid][1] + forces[nid][1]))

for n in network_nodes:
    n["x"] = round(positions[n["id"]][0], 1)
    n["y"] = round(positions[n["id"]][1], 1)

# ── Write outputs ──────────────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)

comp_list = [all_profiles[eid]["display_name"] for eid in COMPARISON_IDS.values() if eid in all_profiles]
pred_list = [all_profiles[eid]["display_name"] for eid in PREDICTION_IDS.values() if eid in all_profiles]

with open(os.path.join(OUT, "artists.json"), "w", encoding="utf-8") as f:
    json.dump({
        "profiles": profiles_export,
        "comparison": comp_list,
        "predictions": pred_list,
        "radar_keys": radar_keys,
        "radar_labels": radar_labels,
        "industry_scatter": industry_scatter,
        "archetype_labels": ARCHETYPE_LABELS,
        "archetype_mirrors": ARCHETYPE_MIRRORS,
        "prediction_explanations": prediction_explanations,
        "framework_keys": framework_keys,
        "selection_model": {
            "name": "8-dimension-rising-star-model",
            "dimensions": model_dimensions,
            "notes": "Outlier-prone dimensions normalized by p95 clipping; recency normalized to observed year range.",
        },
        "selection_funnel": {
            "stages": funnel_stages,
            "candidates": candidate_status,
        },
    }, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump({
        "oceanus_folk": timeline,
        "genre_flow": genre_flow_list[:50],
        "influence_timeline": influence_timeline,
        "chord_genres": top_genres,
        "chord_matrix": chord_matrix,
    }, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT, "network.json"), "w", encoding="utf-8") as f:
    json.dump({
        "nodes": network_nodes,
        "links": network_links,
    }, f, ensure_ascii=False, indent=2)

print(f"\n=== EXPORT SUMMARY ===")
print(f"Profiles: {len(profiles_export)}")
print(f"Comparison: {comp_list}")
print(f"Predictions: {pred_list}")
print(f"Timeline: {len(timeline)} years")
print(f"Network: {len(network_nodes)} nodes, {len(network_links)} links")
print(f"Industry scatter: {len(industry_scatter)} artists")
