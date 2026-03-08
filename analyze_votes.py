#!/usr/bin/env python3
"""
Congress 119 Voting Independence Index
Downloads Voteview data, scores each member, outputs JSON for GitHub Pages site.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
# Congress number is calculated automatically from the current date.
# A new Congress begins January 3rd of every odd-numbered year.
# 119th Congress began January 3, 2025. Formula: ((year - 1789) // 2) + 1
def current_congress():
    today = datetime.now(timezone.utc)
    year = today.year
    if year % 2 == 0 or (year % 2 == 1 and today < datetime(year, 1, 3, tzinfo=timezone.utc)):
        year -= 1
    return ((year - 1789) // 2) + 1

CONGRESS = current_congress()
VOTEVIEW_BASE = "https://voteview.com/static/data/out"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "H_votes":   f"{VOTEVIEW_BASE}/votes/H{CONGRESS}_votes.csv",
    "S_votes":   f"{VOTEVIEW_BASE}/votes/S{CONGRESS}_votes.csv",
    "H_members": f"{VOTEVIEW_BASE}/members/H{CONGRESS}_members.csv",
    "S_members": f"{VOTEVIEW_BASE}/members/S{CONGRESS}_members.csv",
}

# ── Download ───────────────────────────────────────────────────────────────────
def download(url, dest):
    print(f"Downloading {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)

for key, url in FILES.items():
    dest = DATA_DIR / Path(url).name
    download(url, dest)

# ── Load ───────────────────────────────────────────────────────────────────────
h_votes   = pd.read_csv(DATA_DIR / "H119_votes.csv")
s_votes   = pd.read_csv(DATA_DIR / "S119_votes.csv")
h_members = pd.read_csv(DATA_DIR / "H119_members.csv")
s_members = pd.read_csv(DATA_DIR / "S119_members.csv")

votes_all   = pd.concat([h_votes, s_votes],   ignore_index=True)
members_all = pd.concat([h_members, s_members], ignore_index=True)

# Keep only decisive votes (Yea=1, Nay=6)
votes_all = votes_all[votes_all["cast_code"].isin([1, 6])].copy()

# ── Party helpers ──────────────────────────────────────────────────────────────
# 100=D, 200=R, 328=Independent (Sanders/King caucus with D)
def caucus(code):
    if code == 100: return "D"
    if code == 200: return "R"
    if code == 328: return "D"
    return "O"

def display_party(code):
    if code == 100: return "D"
    if code == 200: return "R"
    if code == 328: return "I"
    return "O"

members_all["party"]         = members_all["party_code"].apply(caucus)
members_all["display_party"] = members_all["party_code"].apply(display_party)

# ── Party majority position per vote ──────────────────────────────────────────
party_map = members_all.set_index("icpsr")["party"].to_dict()
votes_all["party"] = votes_all["icpsr"].map(party_map)

vote_party = votes_all[votes_all["party"].isin(["D", "R"])].copy()

def majority_pos(s):
    c = s.value_counts()
    return c.idxmax() if len(c) else None

party_positions = (
    vote_party
    .groupby(["chamber", "rollnumber", "party"])["cast_code"]
    .agg(majority_pos)
    .unstack("party")
    .reset_index()
)
party_positions.columns = ["chamber", "rollnumber", "D_pos", "R_pos"]
party_positions = party_positions.dropna(subset=["D_pos", "R_pos"])
party_positions["vote_type"] = party_positions.apply(
    lambda r: "consensus" if r["D_pos"] == r["R_pos"] else "partisan", axis=1
)

votes_merged = votes_all.merge(party_positions, on=["chamber", "rollnumber"], how="inner")

# ── Label ──────────────────────────────────────────────────────────────────────
def independence_label(score_pct):
    s = round(score_pct, 2)
    if s < 1.0:  return "Mindless Drone"
    if s < 5.0:  return "Yes Man"
    if s < 10.0: return "Reluctant Rebel"
    if s < 20.0: return "Frequent Dissenter"
    if s < 30.0: return "Rebellious Streak"
    return "Lone Wolf"

# ── Score each member ──────────────────────────────────────────────────────────
records = []
for icpsr, grp in votes_merged.groupby("icpsr"):
    mrow = members_all[members_all["icpsr"] == icpsr]
    if mrow.empty: continue
    mrow = mrow.iloc[0]

    party = mrow["party"]
    disp  = mrow["display_party"]
    if party not in ("D", "R"): continue

    n_total = len(grp)
    if n_total < 30: continue  # exclude members with fewer than 30 recorded votes

    pp_col = f"{party}_pos"

    partisan  = grp[grp["vote_type"] == "partisan"]
    n_part    = len(partisan)
    party_unity = (partisan["cast_code"] == partisan[pp_col]).sum() / n_part if n_part else None

    consensus = grp[grp["vote_type"] == "consensus"]
    n_cons    = len(consensus)
    cons_loy  = (consensus["cast_code"] == consensus["D_pos"]).sum() / n_cons if n_cons else None
    cons_dev  = (1 - cons_loy) if cons_loy is not None else None

    p_dev = (1 - party_unity) if party_unity is not None else None
    c_dev = cons_dev
    if   p_dev is not None and c_dev is not None: ind = (p_dev + c_dev) / 2
    elif p_dev is not None:                        ind = p_dev
    elif c_dev is not None:                        ind = c_dev
    else:                                          ind = None

    ind_pct = round(ind * 100, 2) if ind is not None else None

    records.append({
        "name":                mrow["bioname"],
        "party":               disp,
        "caucus":              party,
        "state":               mrow["state_abbrev"],
        "chamber":             "House" if mrow["chamber"] == "House" else "Senate",
        "district":            int(mrow["district_code"]) if mrow["chamber"] == "House" else None,
        "independence_score":  ind_pct,
        "independence_label":  independence_label(ind_pct) if ind_pct is not None else None,
        "party_unity_pct":     round(party_unity * 100, 2) if party_unity is not None else None,
        "partisan_votes":      n_part,
        "consensus_loyalty_pct": round(cons_loy * 100, 2) if cons_loy is not None else None,
        "consensus_deviation_pct": round(cons_dev * 100, 2) if cons_dev is not None else None,
        "consensus_votes":     n_cons,
    })

members = sorted(records, key=lambda r: (r["chamber"], r["party"], r["name"]))

# ── Summary stats ──────────────────────────────────────────────────────────────
def group_stats(subset):
    scores = [r["independence_score"] for r in subset if r["independence_score"] is not None]
    if not scores: return {}
    return {
        "count":      len(subset),
        "avg_independence": round(sum(scores) / len(scores), 2),
        "min_independence": round(min(scores), 2),
        "max_independence": round(max(scores), 2),
        "label_dist": {
            label: sum(1 for r in subset if r["independence_label"] == label)
            for label in ["Mindless Drone","Yes Man","Reluctant Rebel",
                          "Frequent Dissenter","Rebellious Streak","Lone Wolf"]
        }
    }

summary = {
    "all":            group_stats(members),
    "house":          group_stats([r for r in members if r["chamber"] == "House"]),
    "senate":         group_stats([r for r in members if r["chamber"] == "Senate"]),
    "house_dem":      group_stats([r for r in members if r["chamber"] == "House"  and r["caucus"] == "D"]),
    "house_rep":      group_stats([r for r in members if r["chamber"] == "House"  and r["caucus"] == "R"]),
    "senate_dem":     group_stats([r for r in members if r["chamber"] == "Senate" and r["caucus"] == "D"]),
    "senate_rep":     group_stats([r for r in members if r["chamber"] == "Senate" and r["caucus"] == "R"]),
}

# ── Write JSON ─────────────────────────────────────────────────────────────────
output = {
    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "congress":   CONGRESS,
    "summary":    summary,
    "members":    members,
}

out_path = Path("docs/data.json")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(output, indent=2))

print(f"✓ Wrote {len(members)} members to {out_path}")
print(f"  Updated: {output['updated_at']}")
