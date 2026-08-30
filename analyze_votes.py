#!/usr/bin/env python3
"""
Congress 119 Voting Independence Index
Downloads Voteview data, scores each member, outputs JSON for GitHub Pages site.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
# ponytail: hardcoded for the 119th Congress (2025-2027); bump manually when it ends
CONGRESS = 119
VOTEVIEW_BASE = "https://voteview.com/static/data/out"
# Point these at a fixture/scratch dir to run offline (tests/test_scoring.py does).
DATA_DIR = Path(os.environ.get("VOTES_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OFFLINE = "VOTES_DATA_DIR" in os.environ

FILES = [
    f"{VOTEVIEW_BASE}/{kind}/{ch}{CONGRESS}_{kind}.csv"
    for kind in ("votes", "members", "rollcalls")
    for ch in ("H", "S")
]

MIN_VOTES = 30       # exclude members with fewer than this many classified votes
MAX_DISSENTS = 500   # per-member detail file cap

# ── Download ───────────────────────────────────────────────────────────────────
# ponytail: no local cache — CI checks out fresh and gitignores data/, so a
# conditional-GET cache never survives between runs. Just fetch every time.
if not OFFLINE:
    import requests

for url in [] if OFFLINE else FILES:
    dest = DATA_DIR / Path(url).name
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"Downloaded: {url}")

def read_csv(name):
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ── Members ────────────────────────────────────────────────────────────────────
# 100=D, 200=R, 328=Independent. Only Sanders and King caucus with the Democrats;
# any other 328 (e.g. a mid-congress party switcher) gets no caucus and is unscored.
CAUCUS        = {"100": "D", "200": "R"}
DISPLAY_PARTY = {"100": "D", "200": "R", "328": "I"}
DEM_CAUCUSING_INDEPENDENTS = {29147, 41300}  # Sanders (VT), King (ME)

members_all = {}
for ch in ("H", "S"):
    for m in read_csv(f"{ch}{CONGRESS}_members.csv"):
        icpsr = int(m["icpsr"])
        m["caucus"] = "D" if icpsr in DEM_CAUCUSING_INDEPENDENTS else CAUCUS.get(m["party_code"], "O")
        m["display_party"] = DISPLAY_PARTY.get(m["party_code"], "O")
        members_all[icpsr] = m

# ── Votes ──────────────────────────────────────────────────────────────────────
# cast_code: 0=not a member, 1-3=Yea, 4-6=Nay, 7-8=Present, 9=Not Voting.
# Normalize the Yea/Nay families to 1/6 so comparisons are one equality test.
def normalize(code):
    if 1 <= code <= 3: return 1
    if 4 <= code <= 6: return 6
    return code

# ponytail: whole file in memory — 119th is ~370k rows / ~25MB. Stream if a
# future congress makes that hurt.
votes = []           # (chamber, rollnumber, icpsr, cast) for decisive votes only
eligible = Counter()  # icpsr -> rows where they were a member
missed = Counter()    # icpsr -> present/not-voting
for ch in ("H", "S"):
    for v in read_csv(f"{ch}{CONGRESS}_votes.csv"):
        code = normalize(int(v["cast_code"]))
        if code == 0:
            continue
        icpsr = int(v["icpsr"])
        eligible[icpsr] += 1
        if code in (1, 6):
            votes.append((v["chamber"], int(v["rollnumber"]), icpsr, code))
        else:
            missed[icpsr] += 1

# ── Rollcall context ───────────────────────────────────────────────────────────
rollcalls = {}
for ch in ("H", "S"):
    for r in read_csv(f"{ch}{CONGRESS}_rollcalls.csv"):
        rollcalls[(r["chamber"], int(r["rollnumber"]))] = {
            "date": r["date"],
            "bill_number": r["bill_number"],
            "vote_question": r["vote_question"],
            "vote_desc": r["vote_desc"][:240],
            "vote_result": r["vote_result"],
        }

# ── Party majority position + cohesion weight per rollcall ─────────────────────
# tally[(chamber, rollnumber)][party] = Counter of normalized cast codes
tally = defaultdict(lambda: {"D": Counter(), "R": Counter()})
for chamber, roll, icpsr, code in votes:
    m = members_all.get(icpsr)
    if m and m["caucus"] in ("D", "R"):
        tally[(chamber, roll)][m["caucus"]][code] += 1

# rollcall -> {"D_pos","R_pos","D_weight","R_weight","vote_type"}
positions = {}
for key, parties in tally.items():
    info = {}
    for p, counts in parties.items():
        if not counts:
            break
        # tie-break to Yea so the result is deterministic, not dict-order luck
        pos = max((1, 6), key=lambda c: (counts[c], c == 1))
        # cohesion: 0.0 at a 50/50 party split, 1.0 at unanimous
        info[f"{p}_pos"] = pos
        info[f"{p}_weight"] = (counts[pos] / sum(counts.values()) - 0.5) * 2
    if len(info) != 4:
        continue  # a party cast no decisive votes on this rollcall — unclassifiable
    info["vote_type"] = "consensus" if info["D_pos"] == info["R_pos"] else "partisan"
    positions[key] = info

# ── Label ──────────────────────────────────────────────────────────────────────
# (threshold, label) pairs, ascending — mirrors LABELS/scoreColor in docs/index.html
LABEL_THRESHOLDS = [
    (1.0,  "Mindless Drone"),
    (5.0,  "Yes Man"),
    (10.0, "Reluctant Rebel"),
    (20.0, "Frequent Dissenter"),
    (30.0, "Rebellious Streak"),
    (float("inf"), "Lone Wolf"),
]
LABELS = [label for _, label in LABEL_THRESHOLDS]

def independence_label(score_pct):
    s = round(score_pct, 2)
    for threshold, label in LABEL_THRESHOLDS:
        if s < threshold:
            return label

def pct(x):
    return round(x * 100, 2) if x is not None else None

# ── Score each member ──────────────────────────────────────────────────────────
by_member = defaultdict(list)
for chamber, roll, icpsr, code in votes:
    if (chamber, roll) in positions:
        by_member[icpsr].append((chamber, roll, code))

records = []
dissents_by_member = {}
for icpsr, cast in by_member.items():
    m = members_all.get(icpsr)
    if not m or m["caucus"] not in ("D", "R"):
        continue
    if len(cast) < MIN_VOTES:
        continue
    party = m["caucus"]

    n_part = n_defect = 0          # unweighted partisan (party_unity_pct, back-compat)
    w_total = w_defect = 0.0       # cohesion-weighted partisan
    n_cons = n_cons_defect = 0
    dissents = []

    for chamber, roll, code in cast:
        info = positions[(chamber, roll)]
        partisan = info["vote_type"] == "partisan"
        pos = info[f"{party}_pos"] if partisan else info["D_pos"]
        weight = info[f"{party}_weight"] if partisan else 1.0
        defected = code != pos

        if partisan:
            n_part += 1
            w_total += weight
            if defected:
                n_defect += 1
                w_defect += weight
        else:
            n_cons += 1
            if defected:
                n_cons_defect += 1

        if defected:
            dissents.append({
                **rollcalls.get((chamber, roll), {}),
                "chamber": chamber,
                "rollnumber": roll,
                "kind": "partisan" if partisan else "consensus",
                "member_vote": "Yea" if code == 1 else "Nay",
                "party_position": "Yea" if pos == 1 else "Nay",
                "weight": round(weight, 4),
            })

    party_unity = 1 - n_defect / n_part if n_part else None
    p_dev = w_defect / w_total if w_total else None
    cons_loy = 1 - n_cons_defect / n_cons if n_cons else None
    c_dev = (1 - cons_loy) if cons_loy is not None else None

    if   p_dev is not None and c_dev is not None: ind = (p_dev + c_dev) / 2
    elif p_dev is not None:                       ind = p_dev
    else:                                         ind = c_dev

    dissents.sort(key=lambda d: d.get("date", ""), reverse=True)
    dissents_by_member[icpsr] = dissents

    records.append({
        "icpsr":               icpsr,
        "name":                m["bioname"],
        "party":               m["display_party"],
        "caucus":              party,
        "state":               m["state_abbrev"],
        "chamber":             m["chamber"],
        "district":            int(m["district_code"]) if m["chamber"] == "House" else None,
        "independence_score":  pct(ind),
        "independence_label":  independence_label(ind * 100) if ind is not None else None,
        "party_unity_pct":     pct(party_unity),
        "weighted_partisan_deviation_pct": pct(p_dev),
        "partisan_votes":      n_part,
        "consensus_loyalty_pct":   pct(cons_loy),
        "consensus_deviation_pct": pct(c_dev),
        "consensus_votes":     n_cons,
        "eligible_votes":      eligible[icpsr],
        "missed_votes":        missed[icpsr],
        "missed_pct":          pct(missed[icpsr] / eligible[icpsr]) if eligible[icpsr] else None,
        "dissent_count":       len(dissents),
    })

members = sorted(records, key=lambda r: (r["chamber"], r["party"], r["name"]))

# ── Summary stats ──────────────────────────────────────────────────────────────
def group_stats(subset):
    scores = [r["independence_score"] for r in subset if r["independence_score"] is not None]
    if not scores: return {}
    missed_pcts = [r["missed_pct"] for r in subset if r["missed_pct"] is not None]
    return {
        "count":      len(subset),
        "avg_independence": round(sum(scores) / len(scores), 2),
        "min_independence": round(min(scores), 2),
        "max_independence": round(max(scores), 2),
        "avg_missed_pct": round(sum(missed_pcts) / len(missed_pcts), 2) if missed_pcts else None,
        "label_dist": {
            label: sum(1 for r in subset if r["independence_label"] == label)
            for label in LABELS
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
now = datetime.now(timezone.utc)
output = {
    "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "congress":   CONGRESS,
    "summary":    summary,
    "members":    members,
}

docs = Path(os.environ.get("VOTES_OUT_DIR", "docs"))
docs.mkdir(parents=True, exist_ok=True)
(docs / "data.json").write_text(json.dumps(output, indent=2))

# Per-member detail
member_dir = docs / "members"
member_dir.mkdir(parents=True, exist_ok=True)
for stale in member_dir.glob("*.json"):
    stale.unlink()  # a member can drop out (party switch, resignation)
for r in members:
    (member_dir / f"{r['icpsr']}.json").write_text(json.dumps({
        "icpsr": r["icpsr"], "name": r["name"], "party": r["party"],
        "state": r["state"], "chamber": r["chamber"],
        "dissents": dissents_by_member[r["icpsr"]][:MAX_DISSENTS],
        "missed": {"eligible": r["eligible_votes"], "missed": r["missed_votes"],
                   "pct": r["missed_pct"]},
    }))

# History snapshot — one per UTC day, overwritten on re-run
hist_dir = docs / "history"
hist_dir.mkdir(parents=True, exist_ok=True)
today = now.strftime("%Y-%m-%d")
(hist_dir / f"{today}.json").write_text(json.dumps({
    "date": today,
    "updated_at": output["updated_at"],
    "congress": CONGRESS,
    "summary": summary,
    "members": [{k: r[k] for k in
                 ("icpsr", "name", "party", "chamber", "independence_score", "missed_pct")}
                for r in members],
}, indent=2))
(hist_dir / "index.json").write_text(json.dumps(
    sorted(p.stem for p in hist_dir.glob("*.json") if p.name != "index.json")))

print(f"✓ Wrote {len(members)} members to {docs / 'data.json'}")
print(f"  Per-member files: {len(members)} · History snapshot: {today}")
print(f"  Updated: {output['updated_at']}")
