"""Composite rising-star score evolving over career age.

Reimplements the scoring pipeline of viz/preprocess_refactor.py
(L582-880) so the score can be evaluated at every snapshot year,
not just at the dataset end. Plots one line per named artist with
X = career age, Y = composite score.

Outputs (D:\\download\\):
  - report_fig_t_trajectory.png   : static, paper-ready
  - trajectory_tails.html         : interactive D3.js companion
"""
import json
import math
import statistics
from bisect import bisect_left, bisect_right
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent  # repo-root/viz/
ARTISTS_JSON = ROOT / "data" / "artists.json"
OUT = ROOT  # viz/  -- final figure + interactive HTML are written alongside

# Three project-defined baselines (matches preprocess_refactor.py L463-516):
#   Sailor Shift            -- protagonist
#   Embers of Wrath         -- Influence Magnet (dynamic pick: max of_inf_in)
#   Orla Seabloom           -- Rapid Riser (dynamic pick: max velocity, notable_ratio>0.5)
BASELINES = ["Sailor Shift", "Embers of Wrath", "Orla Seabloom"]
# Sailor Shift's former band, called out in the paper introduction
# and excluded from prediction at L577-580. Shown for narrative context.
CONTEXT = ["Ivy Echos"]
# Top-3 predicted rising stars (case study §5.4)
CANDIDATES = ["Copper Canyon Ghosts", "Daniel O'Connell", "Beatrice Albright"]
ALL_ARTISTS = BASELINES + CONTEXT + CANDIDATES

PALETTE = {
    "Sailor Shift":         "#1f6f8b",
    "Embers of Wrath":      "#3b6e8f",
    "Orla Seabloom":        "#7a9b76",
    "Ivy Echos":            "#9aa6b2",  # muted grey-blue for the former-band context
    "Copper Canyon Ghosts": "#d1495b",
    "Daniel O'Connell":     "#edae49",
    "Beatrice Albright":    "#9b5de5",
}

CATEGORY = {
    **{n: "baseline" for n in BASELINES},
    **{n: "context"  for n in CONTEXT},
    **{n: "candidate" for n in CANDIDATES},
}

DIMENSIONS = [
    {"key": "of_momentum", "weight": 1.5},
    {"key": "quality_signal", "weight": 1.3},
    {"key": "influence_reach", "weight": 1.4},
    {"key": "genre_bridge", "weight": 0.8},
    {"key": "collab_network", "weight": 0.7},
    {"key": "industry_traction", "weight": 0.6},
]

RECENCY_HORIZON = 15


# ---------- LOAD ----------
def load_profiles():
    with open(ARTISTS_JSON, encoding="utf-8") as f:
        return json.load(f)["profiles"]


# ---------- VIRTUAL PROFILE ----------
def virtual_profile(p, cutoff_year):
    oy_full = p.get("of_yearly", {}) or {}
    truncated = {y: v for y, v in oy_full.items() if int(y) <= cutoff_year}
    if not truncated:
        return None
    years = sorted(int(y) for y in truncated)
    of_songs = sum(v.get("songs", 0) for v in truncated.values())
    of_notable = sum(v.get("notable", 0) for v in truncated.values())
    of_inf_out = sum(v.get("inf_out", 0) for v in truncated.values())
    of_inf_in = sum(v.get("inf_in", 0) for v in truncated.values())
    first_year = years[0]
    last_year = years[-1]
    return {
        "of_yearly": truncated,
        "of_songs": of_songs,
        "of_notable": of_notable,
        "of_inf_out": of_inf_out,
        "of_inf_in": of_inf_in,
        "of_first_year": first_year,
        "of_last_year": last_year,
        "of_span": last_year - first_year + 1,
        "of_notable_ratio": (of_notable / of_songs) if of_songs else 0.0,
        "of_commitment": p.get("of_commitment", 1.0),
        "of_collaborators": p.get("of_collaborators", 0),
        "of_labels": p.get("of_labels", 0),
        "role_diversity": p.get("role_diversity", 1),
        "genres": p.get("genres", {}),
        "display_name": p.get("display_name", p.get("name", "")),
        "eid": p.get("eid"),
    }


# ---------- RAW DIMS ----------
def compute_raw_dims(p, global_of_min, global_of_range):
    oy = p["of_yearly"]
    years = sorted(int(y) for y in oy if str(y).isdigit())

    def yact(y):
        v = oy[str(y)]
        return v.get("songs", 0) + 0.35 * (v.get("inf_out", 0) + v.get("inf_in", 0))

    if len(years) >= 2:
        cutoff = years[-1] - 2
        recent = sum(yact(y) for y in years if y >= cutoff)
        prior = sum(yact(y) for y in years if y < cutoff)
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

    years_with_activity = sum(
        1 for y, v in oy.items()
        if (v.get("songs", 0) > 0 or v.get("inf_out", 0) > 0 or v.get("inf_in", 0) > 0)
    )
    span_y = max(p["of_span"] or 0, 1)
    activity_breadth_raw = min(1.0, years_with_activity / 5.0) * min(1.0, span_y / 10.0)
    if years_with_activity <= 1:
        activity_breadth_raw *= 0.25

    return {
        "of_momentum": momentum,
        "quality_signal": quality,
        "influence_reach": influence,
        "genre_bridge": genre_bridge,
        "collab_network": collab,
        "industry_traction": industry,
        "acceleration": acceleration,
        "recency_raw": recency_raw,
        "activity_breadth_raw": activity_breadth_raw,
    }


# ---------- PERCENTILES ----------
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
    n = len(pool)
    if n == 0:
        return 0.5
    s = sorted(pool)
    lo = bisect_left(s, v)
    hi = bisect_right(s, v)
    return ((lo + hi) / 2) / max(n - 1, 1)


# ---------- COMPOSITE ----------
def trend_index_geom(m, r, a, b):
    m = max(float(m), 1e-9)
    r = max(float(r), 1e-9)
    a = max(float(a), 0.02)
    b = max(float(b), 1e-9)
    return math.exp(
        0.36 * math.log(m)
        + 0.24 * math.log(r)
        + 0.12 * math.log(a)
        + 0.28 * math.log(b)
    )


def composite(w, t, geo, cons, breadth, years_inactive=0):
    w = max(float(w), 1e-9)
    t = max(float(t), 1e-9)
    joint = math.sqrt(w * t)
    mx, mn = max(w, t), min(w, t)
    core = joint * ((mn / mx) ** 0.40)
    shape = math.sqrt(max(float(geo) * float(cons), 1e-9))
    cadence = math.sqrt(max(float(breadth), 0.02))
    mod_shape = 0.48 + 0.52 * shape
    mod_cadence = 0.52 + 0.48 * cadence
    recency_decay = math.exp(-0.07 * max(years_inactive, 0))
    return min(1.0, core * mod_shape * mod_cadence * recency_decay)


# ---------- POOL ----------
def find_pools(profiles):
    """Comparison pool = the three baselines (matches preprocess_refactor.py L463-516)
    plus Ivy Echos for narrative context. Build candidate pool with lifetime filter,
    excluding the comparison set and Ivy Echos members (same as L575-580).
    """
    by_name = {p["display_name"]: p for p in profiles.values() if "display_name" in p}
    # Comparison artists in the project's percentile pool: only the three baselines
    comps = [by_name[n] for n in BASELINES if n in by_name]
    # Ivy Echos itself is used only for narrative trace (NOT added to scoring pool;
    # it's excluded from prediction in the project's code path)
    if "Ivy Echos" in by_name:
        comps_extra_for_trace = [by_name["Ivy Echos"]]
    else:
        comps_extra_for_trace = []
    comp_eids = {c["eid"] for c in comps}

    ivy_excluded = set()
    for p in profiles.values():
        if p.get("type") == "MusicalGroup" and p.get("display_name", "").startswith("Ivy Echo"):
            ivy_excluded.add(p["eid"])
            for m in p.get("members", []):
                if m in by_name:
                    ivy_excluded.add(by_name[m]["eid"])
    exclude = comp_eids | ivy_excluded

    dataset_max_year = max((p["of_last_year"] for p in profiles.values()
                            if p.get("of_last_year")), default=2040)
    recency_cutoff = dataset_max_year - RECENCY_HORIZON

    candidates = []
    for p in profiles.values():
        if p.get("eid") in exclude:
            continue
        if p["of_songs"] < 2:
            continue
        if p["of_commitment"] < 0.30:
            continue
        if (p.get("of_last_year") or 0) < recency_cutoff:
            continue
        candidates.append(p)

    return candidates, comps, comps_extra_for_trace, dataset_max_year


# ---------- SNAPSHOT SCORER ----------
def score_year(target_year, candidate_pool, comparison_pool,
               global_of_min, global_of_range):
    """Return {display_name: composite} for everyone with virtual data at year T."""
    cand_virtual = []
    for p in candidate_pool:
        vp = virtual_profile(p, target_year)
        if vp is None:
            continue
        cand_virtual.append(vp)
    comp_virtual = []
    for p in comparison_pool:
        vp = virtual_profile(p, target_year)
        if vp is None:
            continue
        comp_virtual.append(vp)
    if not cand_virtual:
        return {}

    cand_dims = [compute_raw_dims(vp, global_of_min, global_of_range) for vp in cand_virtual]
    comp_dims = [compute_raw_dims(vp, global_of_min, global_of_range) for vp in comp_virtual]

    pct = {}
    dim_keys = [d["key"] for d in DIMENSIONS] + ["recency_raw", "activity_breadth_raw"]
    for k in dim_keys:
        pct[k] = rank_percentile([d[k] for d in cand_dims])

    cand_records = []
    for i, (vp, dims) in enumerate(zip(cand_virtual, cand_dims)):
        dim_pct_i = {d["key"]: pct[d["key"]][i] for d in DIMENSIONS}
        weighted_sum = sum(dim_pct_i[d["key"]] * d["weight"] for d in DIMENSIONS)
        trend_raw = trend_index_geom(
            dim_pct_i["of_momentum"], pct["recency_raw"][i],
            dims["acceleration"], pct["activity_breadth_raw"][i],
        )
        pct_vals = [max(dim_pct_i[d["key"]], 0.01) for d in DIMENSIONS]
        geo_mean = math.exp(sum(math.log(v) for v in pct_vals) / len(pct_vals))
        cons = (1.0 - min(1.0, statistics.stdev(pct_vals) / 0.4)
                if len(pct_vals) > 1 else 0.5)
        cand_records.append({
            "vp": vp, "dims": dims,
            "weighted_sum": weighted_sum,
            "trend_raw": trend_raw,
            "geo_mean": geo_mean,
            "consistency": cons,
            "breadth_pct": pct["activity_breadth_raw"][i],
        })

    pool_tr = [r["trend_raw"] for r in cand_records]
    pool_gm = [r["geo_mean"] for r in cand_records]
    pool_cs = [r["consistency"] for r in cand_records]
    pool_wds = [r["weighted_sum"] for r in cand_records]
    trend_pcts = rank_percentile(pool_tr)
    geo_pcts = rank_percentile(pool_gm)
    cons_pcts = rank_percentile(pool_cs)
    wds_pcts = rank_percentile(pool_wds)

    out = {}
    for i, r in enumerate(cand_records):
        inactive = max(0, target_year - r["vp"]["of_last_year"])
        s = composite(wds_pcts[i], trend_pcts[i], geo_pcts[i], cons_pcts[i],
                      r["breadth_pct"], years_inactive=inactive)
        out[r["vp"]["display_name"]] = s

    # comparison-side via percentile_value_in_pool
    for vp, dims in zip(comp_virtual, comp_dims):
        dim_pct = {d["key"]: percentile_value_in_pool(dims[d["key"]],
                                                      [cd[d["key"]] for cd in cand_dims])
                   for d in DIMENSIONS}
        weighted_sum = sum(dim_pct[d["key"]] * d["weight"] for d in DIMENSIONS)
        rec_pct = percentile_value_in_pool(dims["recency_raw"],
                                           [d["recency_raw"] for d in cand_dims])
        breadth_pct_c = percentile_value_in_pool(dims["activity_breadth_raw"],
                                                  [d["activity_breadth_raw"] for d in cand_dims])
        trend_raw = trend_index_geom(dim_pct["of_momentum"], rec_pct,
                                     dims["acceleration"], breadth_pct_c)
        pct_vals = [max(dim_pct[d["key"]], 0.01) for d in DIMENSIONS]
        geo_mean = math.exp(sum(math.log(v) for v in pct_vals) / len(pct_vals))
        cons = (1.0 - min(1.0, statistics.stdev(pct_vals) / 0.4)
                if len(pct_vals) > 1 else 0.5)
        trend_pct = percentile_value_in_pool(trend_raw, pool_tr)
        geo_pct = percentile_value_in_pool(geo_mean, pool_gm)
        cons_pct = percentile_value_in_pool(cons, pool_cs)
        wds_pct = percentile_value_in_pool(weighted_sum, pool_wds)
        inactive = max(0, target_year - vp["of_last_year"])
        s = composite(wds_pct, trend_pct, geo_pct, cons_pct, breadth_pct_c,
                      years_inactive=inactive)
        out[vp["display_name"]] = s
    return out


# ---------- SNAPSHOT POOL BUILDER (age-aligned) ----------
def score_age(A, candidate_pool, comparison_pool,
              global_of_min, global_of_range):
    """At career age A: virtualize every artist at their own year debut+A
    (no truncation past their lifetime; right-censoring grows years_inactive).
    Return {display_name: composite_score}.
    """
    def virtualize_at_age(p):
        debut = p.get("of_first_year")
        if not debut:
            return None
        snapshot_year = debut + A
        vp = virtual_profile(p, snapshot_year)
        if vp is None:
            return None
        # years_inactive measured at this snapshot
        vp["_years_inactive"] = max(0, snapshot_year - vp["of_last_year"])
        return vp

    cand_vp = [vp for vp in (virtualize_at_age(p) for p in candidate_pool)
               if vp is not None]
    comp_vp = [vp for vp in (virtualize_at_age(p) for p in comparison_pool)
               if vp is not None]
    if not cand_vp:
        return {}

    cand_dims = [compute_raw_dims(vp, global_of_min, global_of_range) for vp in cand_vp]
    comp_dims = [compute_raw_dims(vp, global_of_min, global_of_range) for vp in comp_vp]

    # Combined pool for dim percentiles (mirrors project's raw_rows at L682-695)
    combined_dims = cand_dims + comp_dims
    n_cand = len(cand_dims)
    pct = {}
    dim_keys = [d["key"] for d in DIMENSIONS] + ["recency_raw", "activity_breadth_raw"]
    for k in dim_keys:
        pct[k] = rank_percentile([d[k] for d in combined_dims])

    cand_records = []
    for i, (vp, dims) in enumerate(zip(cand_vp, cand_dims)):
        dim_pct_i = {d["key"]: pct[d["key"]][i] for d in DIMENSIONS}
        weighted_sum = sum(dim_pct_i[d["key"]] * d["weight"] for d in DIMENSIONS)
        trend_raw = trend_index_geom(
            dim_pct_i["of_momentum"], pct["recency_raw"][i],
            dims["acceleration"], pct["activity_breadth_raw"][i],
        )
        pct_vals = [max(dim_pct_i[d["key"]], 0.01) for d in DIMENSIONS]
        geo_mean = math.exp(sum(math.log(v) for v in pct_vals) / len(pct_vals))
        cons = (1.0 - min(1.0, statistics.stdev(pct_vals) / 0.4)
                if len(pct_vals) > 1 else 0.5)
        cand_records.append({
            "vp": vp, "dims": dims,
            "weighted_sum": weighted_sum, "trend_raw": trend_raw,
            "geo_mean": geo_mean, "consistency": cons,
            "breadth_pct": pct["activity_breadth_raw"][i],
        })

    pool_tr = [r["trend_raw"] for r in cand_records]
    pool_gm = [r["geo_mean"] for r in cand_records]
    pool_cs = [r["consistency"] for r in cand_records]
    pool_wds = [r["weighted_sum"] for r in cand_records]
    trend_pcts = rank_percentile(pool_tr)
    geo_pcts = rank_percentile(pool_gm)
    cons_pcts = rank_percentile(pool_cs)
    wds_pcts = rank_percentile(pool_wds)

    out = {}
    for i, r in enumerate(cand_records):
        s = composite(wds_pcts[i], trend_pcts[i], geo_pcts[i], cons_pcts[i],
                      r["breadth_pct"], years_inactive=r["vp"]["_years_inactive"])
        out[r["vp"]["display_name"]] = s

    for j, (vp, dims) in enumerate(zip(comp_vp, comp_dims)):
        # comparison dim_pct from combined pool, by index n_cand+j
        idx = n_cand + j
        dim_pct = {d["key"]: pct[d["key"]][idx] for d in DIMENSIONS}
        weighted_sum = sum(dim_pct[d["key"]] * d["weight"] for d in DIMENSIONS)
        rec_pct = pct["recency_raw"][idx]
        breadth_pct_c = pct["activity_breadth_raw"][idx]
        trend_raw = trend_index_geom(dim_pct["of_momentum"], rec_pct,
                                     dims["acceleration"], breadth_pct_c)
        pct_vals = [max(dim_pct[d["key"]], 0.01) for d in DIMENSIONS]
        geo_mean = math.exp(sum(math.log(v) for v in pct_vals) / len(pct_vals))
        cons = (1.0 - min(1.0, statistics.stdev(pct_vals) / 0.4)
                if len(pct_vals) > 1 else 0.5)
        trend_pct = percentile_value_in_pool(trend_raw, pool_tr)
        geo_pct = percentile_value_in_pool(geo_mean, pool_gm)
        cons_pct = percentile_value_in_pool(cons, pool_cs)
        wds_pct = percentile_value_in_pool(weighted_sum, pool_wds)
        s = composite(wds_pct, trend_pct, geo_pct, cons_pct, breadth_pct_c,
                      years_inactive=vp["_years_inactive"])
        out[vp["display_name"]] = s
    return out


# ---------- TRACE BUILDER ----------
def build_traces(profiles):
    candidate_pool, comparison_pool, extra_trace_pool, dataset_max_year = find_pools(profiles)
    # extra_trace_pool members (Ivy Echos) get their score computed via the same
    # comparison code path but are NOT in the project's scoring percentile pool.
    full_comp_pool = comparison_pool + extra_trace_pool
    print(f"Candidates: {len(candidate_pool)}, comparisons: "
          f"{[c['display_name'] for c in comparison_pool]}, "
          f"extra-for-trace: {[c['display_name'] for c in extra_trace_pool]}, "
          f"dataset_max_year={dataset_max_year}")

    cand_last = sorted(p["of_last_year"] for p in candidate_pool
                       if p.get("of_last_year"))
    global_of_min = cand_last[0] if cand_last else 1993
    global_of_max = cand_last[-1] if cand_last else dataset_max_year
    global_of_range = max(global_of_max - global_of_min, 1)

    by_name = {p["display_name"]: p for p in profiles.values() if "display_name" in p}

    # max career age we need = longest named-artist career
    max_age = 0
    for name in ALL_ARTISTS:
        p = by_name.get(name)
        if p and p.get("of_first_year") and p.get("of_last_year"):
            max_age = max(max_age, p["of_last_year"] - p["of_first_year"])
    # cache per-age score map for every age 0..max_age
    cache = {}
    for A in range(0, max_age + 1):
        cache[A] = score_age(A, candidate_pool, full_comp_pool,
                             global_of_min, global_of_range)

    # also compute each named artist's "published-style score" at dataset end
    # (project semantics: years_inactive = dataset_max_year - of_last_year)
    end_score = {}
    by_eid = {p["eid"]: p for p in profiles.values() if p.get("eid") is not None}
    for name in ALL_ARTISTS:
        p = by_name.get(name)
        if p is None:
            continue
        debut = p.get("of_first_year")
        end = p.get("of_last_year")
        if not debut or not end:
            continue
        # virtualize at lifetime end + use dataset_max_year inactivity
        age_at_end = end - debut
        snap_score_map = cache.get(age_at_end) or {}
        # NOTE: my snap uses years_inactive = max(0, snap_year - last) = 0 here,
        #       so to match the project's published score we'd add an inactivity term.
        end_score[name] = snap_score_map.get(name)

    traces = {n: [] for n in ALL_ARTISTS}
    for name in ALL_ARTISTS:
        p = by_name.get(name)
        if p is None:
            continue
        debut = p.get("of_first_year")
        end = p.get("of_last_year")
        if not debut or not end:
            continue
        career_len = end - debut
        for A in range(0, career_len + 1):
            s = cache.get(A, {}).get(name)
            if s is None:
                continue
            traces[name].append((A, s, debut + A))
    return traces, dataset_max_year, end_score


# ---------- FIGURE ----------
def make_figure(traces):
    fig, ax = plt.subplots(figsize=(11.0, 6.6))

    sailor_curve = traces.get("Sailor Shift", [])
    ss_final = sailor_curve[-1][1] if sailor_curve else None
    if ss_final is not None:
        ax.axhline(ss_final, color=PALETTE["Sailor Shift"], lw=1.0,
                   ls=(0, (4, 4)), alpha=0.45, zorder=1)
        ax.text(0.05, ss_final + 0.015,
                "Sailor Shift end-of-career level",
                fontsize=8.5, color=PALETTE["Sailor Shift"], alpha=0.7)

    label_rows = []
    max_age = 0
    age0_singletons = sorted(
        [(n, traces[n][0][1]) for n in ALL_ARTISTS
         if traces.get(n) and len(traces[n]) == 1 and traces[n][0][0] == 0],
        key=lambda r: -r[1],
    )
    jitter = {n: (k - (len(age0_singletons) - 1) / 2) * 0.45
              for k, (n, _) in enumerate(age0_singletons)}

    def style_for(name):
        cat = CATEGORY.get(name, "baseline")
        if cat == "candidate":
            return dict(lw=3.0, ms_line=42, ms_dot=190, alpha=0.95, zorder=5,
                        weight="bold", suffix="  *")
        if cat == "context":
            return dict(lw=1.6, ms_line=22, ms_dot=110, alpha=0.85, zorder=2,
                        weight="normal", suffix="  (Sailor's former band)")
        return dict(lw=2.2, ms_line=28, ms_dot=120, alpha=0.95, zorder=3,
                    weight="normal", suffix="")

    for name in ALL_ARTISTS:
        pts = traces.get(name) or []
        if not pts:
            continue
        st = style_for(name)
        ages = [a + jitter.get(name, 0.0) for a, _, _ in pts]
        scores = [s for _, s, _ in pts]
        max_age = max(max_age, int(pts[-1][0]))
        color = PALETTE[name]
        if len(ages) >= 2:
            # context line gets a dashed stroke to distinguish it visually
            ls = (0, (3, 2)) if CATEGORY[name] == "context" else "-"
            ax.plot(ages, scores, color=color, lw=st["lw"], ls=ls,
                    solid_capstyle="round",
                    alpha=st["alpha"], zorder=st["zorder"])
            ax.scatter(ages, scores, s=st["ms_line"],
                       color=color, edgecolor="white", lw=0.9,
                       zorder=st["zorder"] + 1)
        else:
            ax.scatter(ages, scores, s=st["ms_dot"],
                       color=color, edgecolor="white", lw=1.4,
                       zorder=st["zorder"] + 1)
        label_rows.append((name, ages[-1], scores[-1], color,
                           st["weight"], st["suffix"]))

    label_rows.sort(key=lambda r: -r[2])
    min_gap = 0.045
    placed = []
    for name, x, y, color, weight, suffix in label_rows:
        ytarget = y
        for py in placed:
            if abs(ytarget - py) < min_gap:
                ytarget = py - min_gap
        placed.append(ytarget)
        ax.annotate(
            name + suffix,
            xy=(x, y),
            xytext=(x + 0.45, ytarget),
            textcoords="data",
            fontsize=9.0, color=color,
            fontweight=weight,
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.4,
                            shrinkA=2, shrinkB=2) if abs(ytarget - y) > 0.01 else None,
        )

    ax.set_title("Composite rising-star score evolving over an artist's career",
                 fontsize=12.5, pad=10)
    ax.set_xlabel("Career age (years since first Oceanus Folk release)", fontsize=10.5)
    ax.set_ylabel("Composite rising-star score", fontsize=10.5)
    ax.grid(True, ls=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, max_age + 4.0)
    ymax = max(ax.get_ylim()[1], 0.55)
    ax.set_ylim(0, ymax)

    legend_handles = [
        Line2D([0], [0], color="#1f6f8b", lw=2.4, marker="o", markersize=7,
               label="Baseline superstar"),
        Line2D([0], [0], color="#9aa6b2", lw=1.6, ls=(0, (3, 2)),
               marker="o", markersize=6,
               label="Sailor Shift's former band (Ivy Echos)"),
        Line2D([0], [0], color="#d1495b", lw=3.0, marker="o", markersize=9,
               label="Predicted rising star  *"),
        Line2D([0], [0], color=PALETTE["Sailor Shift"], lw=1.0,
               ls=(0, (4, 4)), label="Sailor Shift end-of-career level"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.8,
              framealpha=0.92)

    fig.tight_layout()
    out_path = OUT / "report_fig_t_trajectory.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------- D3 ----------
D3_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Composite rising-star score over career age</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body { font: 13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; margin: 22px; color:#222; }
  h1 { font-size: 16px; margin: 0 0 6px; }
  .sub { color:#666; margin-bottom: 14px; max-width: 820px; }
  .axis path,.axis line { stroke:#bbb; }
  .axis text { fill:#555; font-size: 11px; }
  .tail { fill:none; stroke-linecap:round; }
  .tooltip {
    position:absolute; pointer-events:none; background:rgba(20,20,20,.88);
    color:#fff; padding:4px 8px; border-radius:4px; font-size:11px;
    transform:translate(-50%, -120%); opacity:0; transition:opacity .12s;
  }
  .ctrl label { margin-right:14px; font-size:12px; }
</style></head><body>
<h1>Composite rising-star score over career age</h1>
<div class="sub">
  One line per named artist; horizontal axis is years since their first
  Oceanus Folk release. Hover any marker for the exact score and year.
</div>
<div class="ctrl">
  <label><input type="checkbox" id="showBase" checked /> baselines</label>
  <label><input type="checkbox" id="showCand" checked /> predicted stars</label>
</div>
<svg id="chart" width="940" height="540"></svg>
<div class="tooltip" id="tt"></div>
<script>
const DATA=__DATA__, PALETTE=__PALETTE__, CANDIDATES=new Set(__CANDIDATES__);
const SS_FINAL=__SS_FINAL__;
const m={top:26,right:170,bottom:48,left:60};
const svg=d3.select("#chart"),
      W=+svg.attr("width")-m.left-m.right,
      H=+svg.attr("height")-m.top-m.bottom,
      g=svg.append("g").attr("transform",`translate(${m.left},${m.top})`),
      tt=d3.select("#tt");
const allPts=DATA.flatMap(d=>d.points);
const xS=d3.scaleLinear().domain([0,d3.max(allPts,p=>p.age)+1]).range([0,W]);
const yS=d3.scaleLinear().domain([0,Math.max(0.55,d3.max(allPts,p=>p.score)*1.1)]).range([H,0]);
g.append("g").attr("class","axis").attr("transform",`translate(0,${H})`).call(d3.axisBottom(xS));
g.append("g").attr("class","axis").call(d3.axisLeft(yS));
g.append("text").attr("x",W/2).attr("y",H+36).attr("text-anchor","middle").style("fill","#444")
  .text("Career age (years since first Oceanus Folk release)");
g.append("text").attr("transform","rotate(-90)").attr("x",-H/2).attr("y",-44)
  .attr("text-anchor","middle").style("fill","#444")
  .text("Composite rising-star score");

if(SS_FINAL!==null){
  g.append("line").attr("x1",0).attr("x2",W)
    .attr("y1",yS(SS_FINAL)).attr("y2",yS(SS_FINAL))
    .attr("stroke",PALETTE["Sailor Shift"]).attr("stroke-dasharray","4,4").attr("stroke-width",1);
  g.append("text").attr("x",6).attr("y",yS(SS_FINAL)-4)
    .attr("fill",PALETTE["Sailor Shift"]).style("font-size","10px").style("opacity",.7)
    .text("Sailor Shift end-of-career level");
}

const line=d3.line().x(p=>xS(p.age)).y(p=>yS(p.score));
const groups=g.selectAll(".artist").data(DATA).enter().append("g").attr("class","artist")
  .attr("data-name",d=>d.name);
groups.each(function(d){
  const sel=d3.select(this), isC=CANDIDATES.has(d.name), col=PALETTE[d.name];
  if(d.points.length>=2){
    sel.append("path").attr("class","tail").attr("d",line(d.points))
       .attr("stroke",col).attr("stroke-width",isC?3.0:2.2);
  }
  sel.selectAll("circle").data(d.points).enter().append("circle")
    .attr("cx",p=>xS(p.age)).attr("cy",p=>yS(p.score))
    .attr("r",isC?5:3.6).attr("fill",col)
    .attr("stroke","#fff").attr("stroke-width",1)
    .on("mousemove",(e,p)=>{tt.style("opacity",1).style("left",e.pageX+"px").style("top",e.pageY+"px")
        .html(`<b>${d.name}</b> · age ${p.age} · ${p.year}<br>composite ${p.score.toFixed(3)}`);})
    .on("mouseleave",()=>tt.style("opacity",0));
  const last=d.points[d.points.length-1];
  sel.append("text").attr("x",xS(last.age)+10).attr("y",yS(last.score)+4)
    .attr("fill",col).style("font-size","11px")
    .style("font-weight",isC?"700":"400")
    .text(d.name + (isC?"  *":""));
});

function refresh(){
  const sb=d3.select("#showBase").property("checked");
  const sc=d3.select("#showCand").property("checked");
  d3.selectAll("g.artist").style("display",function(){
    const n=this.getAttribute("data-name");
    const c=CANDIDATES.has(n);
    if(c && !sc) return "none";
    if(!c && !sb) return "none";
    return null;
  });
}
d3.selectAll("#showBase,#showCand").on("change",refresh);
</script></body></html>
"""


def write_d3(traces):
    payload = []
    for i, name in enumerate(ALL_ARTISTS):
        pts = traces.get(name) or []
        if not pts:
            continue
        payload.append({
            "id": i, "name": name,
            "points": [{"age": int(a), "score": float(s), "year": int(y)}
                       for a, s, y in pts]
        })
    ss = traces.get("Sailor Shift") or []
    ss_final = float(ss[-1][1]) if ss else None
    html = (D3_TEMPLATE
            .replace("__DATA__", json.dumps(payload))
            .replace("__PALETTE__", json.dumps(PALETTE))
            .replace("__CANDIDATES__", json.dumps(CANDIDATES))
            .replace("__SS_FINAL__", "null" if ss_final is None else f"{ss_final:.4f}"))
    (OUT / "trajectory_tails.html").write_text(html, encoding="utf-8")
    print(f"Wrote {OUT / 'trajectory_tails.html'}")


def main():
    profiles = load_profiles()
    traces, _, _ = build_traces(profiles)
    print("\n--- terminal composite scores ---")
    for n in ALL_ARTISTS:
        pts = traces.get(n) or []
        if pts:
            a, s, y = pts[-1]
            print(f"{n:<22}  age {a:>2}  year {y}  score {s:.4f}")
        else:
            print(f"{n:<22}  (no data)")
    make_figure(traces)
    write_d3(traces)


if __name__ == "__main__":
    main()
