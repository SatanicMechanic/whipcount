#!/usr/bin/env python3
"""
Congressional Voting Independence Index
Downloads Voteview data, scores each member, outputs JSON for GitHub Pages site.

The Congress is derived from today's date, so the weekly job rolls over to the
120th on its own. Set VOTES_CONGRESS to pin a specific one.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
VOTEVIEW_BASE = "https://voteview.com/static/data/out"
# VOTES_DATA_DIR relocates the CSV directory. VOTES_OFFLINE=1 skips the download
# and reuses whatever is already there. They are separate on purpose: relocating
# the cache must never silently stop fetching and republish stale data.
DATA_DIR = Path(os.environ.get("VOTES_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OFFLINE = os.environ.get("VOTES_OFFLINE") == "1"


def congress_on(day):
    """Which Congress is sitting on `day`.

    The 1st convened in 1789 and each runs two years, so the arithmetic is
    exact — no table to maintain. A Congress convenes on January 3 of the odd
    year, so shift back two days to keep Jan 1-2 with the outgoing one.
    """
    return ((day - timedelta(days=2)).year - 1789) // 2 + 1


# VOTES_CONGRESS pins a specific Congress — needed to rebuild an old one, and
# what the self-check uses so its fixture doesn't expire on New Year's Day.
CONGRESS = int(os.environ.get("VOTES_CONGRESS") or congress_on(datetime.now(timezone.utc).date()))
PINNED = bool(os.environ.get("VOTES_CONGRESS"))

MIN_VOTES = 30       # exclude members with fewer than this many classified votes
MAX_DISSENTS = 500   # per-member detail file cap

# ── Download ───────────────────────────────────────────────────────────────────
# ponytail: no local cache — CI checks out fresh and gitignores data/, so a
# conditional-GET cache never survives between runs. Just fetch every time.
if OFFLINE:
    print(f"! OFFLINE — reusing the CSVs already in {DATA_DIR}, downloading nothing")
else:
    import requests

    def download(congress):
        """Fetch all six CSVs. False if Voteview has none for this Congress yet."""
        for kind in ("votes", "members", "rollcalls"):
            for ch in ("H", "S"):
                url = f"{VOTEVIEW_BASE}/{kind}/{ch}{congress}_{kind}.csv"
                r = requests.get(url, timeout=60)
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                (DATA_DIR / Path(url).name).write_bytes(r.content)
                print(f"Downloaded: {url}")
        return True

    # A Congress convenes January 3, but Voteview publishes days or weeks later,
    # and the two chambers don't necessarily land together. Fall back one Congress
    # rather than failing every weekly run during the gap — the site keeps serving
    # the outgoing Congress until the new one's data lands, then switches itself.
    if not download(CONGRESS):
        if PINNED:
            raise SystemExit(f"Voteview has no data for Congress {CONGRESS}")
        print(f"! Voteview has no Congress {CONGRESS} data yet — using {CONGRESS - 1}")
        CONGRESS -= 1
        if not download(CONGRESS):
            raise SystemExit(f"Voteview has no data for Congress {CONGRESS}")

# ── Upstream schema ────────────────────────────────────────────────────────────
# Voteview is a third party that can change its files at any time. A renamed
# column would otherwise surface as a bare KeyError hundreds of lines down; a new
# cast_code or chamber value would not surface at all — it would silently stop
# being counted, and the index would just be quietly wrong. Check both, write
# anything unexpected to the drift report, and let CI open an issue from it.
REQUIRED_COLUMNS = {
    "members":   {"icpsr", "chamber", "state_abbrev", "party_code",
                  "district_code", "bioname"},
    "votes":     {"icpsr", "chamber", "rollnumber", "cast_code"},
    "rollcalls": {"chamber", "rollnumber", "date", "bill_number",
                  "vote_question", "vote_desc", "vote_result"},
}
KNOWN_CHAMBERS = {"House", "Senate", "President"}
KNOWN_CAST_CODES = set(range(10))   # 0 not a member · 1-6 Yea/Nay · 7-8 Present · 9 Not Voting
KNOWN_PARTY_CODES = {"100", "200", "328"}
SCHEMA_REPORT = Path(os.environ.get("VOTES_SCHEMA_REPORT", "schema-drift.txt"))
drift = []


def note_drift(msg):
    if msg not in drift:
        drift.append(msg)


def fail_schema(msg):
    """Unrecoverable drift: record it for CI, then stop."""
    SCHEMA_REPORT.write_text(msg + "\n")
    raise SystemExit(f"! {msg}")


def read_csv(name, kind):
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = set(reader.fieldnames or [])   # not set(rows[0]) — files can be empty
    missing = REQUIRED_COLUMNS[kind] - header
    if missing:
        fail_schema(f"{name}: Voteview dropped or renamed column(s): {', '.join(sorted(missing))}")
    return rows

# ── Members ────────────────────────────────────────────────────────────────────
# 100=D, 200=R, 328=Independent. Only Sanders and King caucus with the Democrats;
# any other 328 (e.g. a mid-congress party switcher) gets no caucus and is unscored.
CAUCUS        = {"100": "D", "200": "R"}
DISPLAY_PARTY = {"100": "D", "200": "R", "328": "I"}
DEM_CAUCUSING_INDEPENDENTS = {29147, 41300}  # Sanders (VT), King (ME)

# Unmapped members we've already looked at and accepted as unscorable. Voteview
# gives a mid-term party switcher a second row under a 9xxxx ICPSR, so 92336 is
# Kiley's post-switch stint alongside his ordinary 22336 row. Listed here only to
# keep the weekly drift report quiet; anyone not listed gets reported once.
ACCEPTED_UNSCORED = {92336}

# Non-voting delegates. They are barred from final-passage votes, so their
# record is a Committee-of-the-Whole subset — tens of votes against a chamber
# norm of hundreds — and their "missed" rate is the franchise, not attendance.
# Ranking them against voting members compares two different things, so they
# are dropped outright (unlike leaders, who are only flagged).
DELEGATE_STATES = {"DC", "PR", "VI", "GU", "AS", "MP"}

# Floor leaders schedule the votes they then vote on, so their loyalty is partly
# loyalty to their own agenda. Flagged rather than excluded — they are the first
# names a reader looks up.
#
# Keyed by Congress on purpose. ICPSR numbers follow the person, not the post, so
# carrying this table forward would label the 119th's Speaker "Speaker" for the
# rest of his career. An unlisted Congress flags nobody and says so at the end of
# the build — a missing badge is recoverable, a wrong one is a false claim.
LEADERSHIP_BY_CONGRESS = {
    119: {
        21727: "Speaker",          # Johnson (R-LA)
        20759: "Majority Leader",  # Scalise (R-LA)
        21531: "Majority Whip",    # Emmer (R-MN)
        21343: "Minority Leader",  # Jeffries (D-NY)
        21375: "Minority Whip",    # Clark (D-MA)
        29754: "Majority Leader",  # Thune (R-SD)
        40707: "Majority Whip",    # Barrasso (R-WY)
        14858: "Minority Leader",  # Schumer (D-NY)
        15021: "Minority Whip",    # Durbin (D-IL)
    },
}
LEADERSHIP = LEADERSHIP_BY_CONGRESS.get(CONGRESS, {})

members_all = {}
delegates = []
for ch in ("H", "S"):
    for m in read_csv(f"{ch}{CONGRESS}_members.csv", "members"):
        # Voteview carries the president's announced positions as ordinary
        # cast_code rows under chamber "President". They are not votes: drop the
        # row so they reach neither the party cohesion tally nor the index.
        if m["chamber"] not in KNOWN_CHAMBERS:
            note_drift(f"members: unrecognised chamber {m['chamber']!r} "
                       f"(e.g. {m['bioname']}) — skipped")
        if m["party_code"] not in KNOWN_PARTY_CODES:
            note_drift(f"members: unrecognised party_code {m['party_code']!r} "
                       f"(e.g. {m['bioname']}) — member left unscored")
        if m["chamber"] not in ("House", "Senate"):
            continue
        if m["state_abbrev"] in DELEGATE_STATES:
            delegates.append(m["bioname"])
            continue
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
votes = []                     # (chamber, rollnumber, icpsr, cast) — decisive only
rolls_seen = defaultdict(set)  # chamber -> every rollnumber it held
span = {}                      # icpsr -> [chamber, first roll, last roll] on record
cast_count = Counter()         # icpsr -> decisive votes actually cast

for ch in ("H", "S"):
    for v in read_csv(f"{ch}{CONGRESS}_votes.csv", "votes"):
        chamber, roll = v["chamber"], int(v["rollnumber"])
        rolls_seen[chamber].add(roll)
        raw = int(v["cast_code"])
        if raw not in KNOWN_CAST_CODES:
            # normalize() passes an unknown code straight through, where it lands
            # in the "missed" bucket. Silent miscounting is the failure to catch.
            note_drift(f"votes: unrecognised cast_code {raw} "
                       f"({chamber} rollcall {roll}) — counted as a missed vote")
        code = normalize(raw)
        if code == 0:
            continue
        icpsr = int(v["icpsr"])
        if icpsr in span:
            s = span[icpsr]
            s[1], s[2] = min(s[1], roll), max(s[2], roll)
        else:
            span[icpsr] = [chamber, roll, roll]
        if code in (1, 6):
            cast_count[icpsr] += 1
            votes.append((chamber, roll, icpsr, code))

# Attendance denominator: every rollcall the chamber held between a member's
# first and last appearance. Voteview drops the row entirely when a member
# doesn't vote rather than coding it 9 — the Speaker is absent from ~23% of
# House rollcalls this way — so counting rows would hand him perfect attendance.
# Bounding by first/last keeps mid-congress arrivals and departures from being
# charged for votes held outside their service.
# ponytail: a member absent for their own first or last rollcalls has that
# stretch fall outside the span, undercounting their misses by those few votes.
eligible = Counter()  # icpsr -> rollcalls held while they served
missed = Counter()    # icpsr -> those with no Yea/Nay recorded
for icpsr, (chamber, first, last) in span.items():
    eligible[icpsr] = sum(1 for r in rolls_seen[chamber] if first <= r <= last)
    missed[icpsr] = eligible[icpsr] - cast_count[icpsr]

# ── Rollcall context ───────────────────────────────────────────────────────────
rollcalls = {}
for ch in ("H", "S"):
    for r in read_csv(f"{ch}{CONGRESS}_rollcalls.csv", "rollcalls"):
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
        "leadership":          LEADERSHIP.get(icpsr),
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
        "state": r["state"], "chamber": r["chamber"], "leadership": r["leadership"],
        "dissents": dissents_by_member[r["icpsr"]][:MAX_DISSENTS],
        "missed": {"eligible": r["eligible_votes"], "missed": r["missed_votes"],
                   "pct": r["missed_pct"]},
    }))

# History snapshot — one per UTC day, overwritten on re-run.
# Summary only. A per-member series was measured and isn't worth the bytes: the
# median member moves 0.02 points week over week and the only members who swing
# are new arrivals thrashing on a small denominator. The signal is at caucus
# level, which is exactly what summary holds — and this keeps a weekly commit
# to ~60 lines instead of ~4500 for the rest of the congress.
hist_dir = docs / "history"
hist_dir.mkdir(parents=True, exist_ok=True)
today = now.strftime("%Y-%m-%d")
# No updated_at in the snapshot: it would differ on every run and make a same-day
# re-run look like new history. "date" identifies it; data.json has the build time.
(hist_dir / f"{today}.json").write_text(json.dumps({
    "date": today,
    "congress": CONGRESS,
    "summary": summary,
}, indent=2))
# Each entry carries its congress so a trend view can select one without opening
# every snapshot — the 119th's archive must not bleed into the 120th's chart.
(hist_dir / "index.json").write_text(json.dumps(
    [{"date": p.stem, "congress": json.loads(p.read_text())["congress"]}
     for p in sorted(hist_dir.glob("*.json")) if p.name != "index.json"]))

# A party code we don't map (Voteview's 328 for anyone who isn't Sanders or King)
# drops that member from the index entirely. Name them rather than quietly shrink.
unscored = sorted((int(m["icpsr"]), m["bioname"]) for m in members_all.values()
                  if m["caucus"] not in ("D", "R"))
if unscored:
    print(f"! Unscored — no mapped caucus: {', '.join(n for _, n in unscored)}")
# Anyone new needs a person to decide which caucus, if any, they sit with — a log
# line nobody reads is how a member quietly vanishes from the index for a term.
# Already-reviewed cases stay off the report so the weekly run doesn't cry wolf.
newly_unscored = [n for i, n in unscored if i not in ACCEPTED_UNSCORED]
if newly_unscored:
    note_drift(f"unscored — no mapped caucus, decide and add to "
               f"DEM_CAUCUSING_INDEPENDENTS or ACCEPTED_UNSCORED: "
               f"{', '.join(newly_unscored)}")
if delegates:
    print(f"! Excluded — non-voting delegates: {', '.join(sorted(delegates))}")
if not LEADERSHIP:
    print(f"! No leadership table for Congress {CONGRESS} — nobody flagged. "
          f"Add one to LEADERSHIP_BY_CONGRESS in {Path(__file__).name}.")
    # Every new Congress needs this filled in once. Surfacing it as an issue is
    # what makes the January rollover a task rather than a silent omission.
    note_drift(f"no leadership table for Congress {CONGRESS} — nobody is flagged; "
               f"add one to LEADERSHIP_BY_CONGRESS in {Path(__file__).name}")

# The build still publishes on drift — the numbers are usually fine and stale is
# worse than slightly-off. CI turns a non-empty report into a GitHub issue.
if drift:
    SCHEMA_REPORT.write_text("\n".join(f"- {d}" for d in drift) + "\n")
    print(f"! Upstream schema drift ({len(drift)}) — wrote {SCHEMA_REPORT}")
    for d in drift:
        print(f"    {d}")
elif SCHEMA_REPORT.exists():
    SCHEMA_REPORT.unlink()   # clean run — don't let a stale report reopen an issue

print(f"✓ Wrote {len(members)} members to {docs / 'data.json'}")
print(f"  Per-member files: {len(members)} · History snapshot: {today}")
print(f"  Updated: {output['updated_at']}")
