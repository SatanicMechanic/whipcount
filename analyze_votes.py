#!/usr/bin/env python3
"""
Congressional Voting Independence Index
Downloads Voteview data, scores each member, outputs JSON for GitHub Pages site.

Nothing is pinned to one Congress: which one is in scope comes from today's date,
so the weekly job rolls itself over every odd January. Set VOTES_CONGRESS to pin
a specific one for rebuilding an old term.
"""

import csv
import json
import os
import re
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
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Voteview is one university host and a transient 5xx used to skip the whole
    # weekly publish. 404 stays un-retried: it is the "not published yet" answer
    # the fallback below depends on.
    http = requests.Session()
    http.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}))))

    def download(congress):
        """Fetch all six CSVs. False if Voteview has none for this Congress yet."""
        for kind in ("votes", "members", "rollcalls"):
            for ch in ("H", "S"):
                url = f"{VOTEVIEW_BASE}/{kind}/{ch}{congress}_{kind}.csv"
                r = http.get(url, timeout=60)
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
    "members":   {"icpsr", "chamber", "state_abbrev", "party_code", "bioname"},
    "votes":     {"icpsr", "chamber", "rollnumber", "cast_code"},
    "rollcalls": {"chamber", "rollnumber", "date", "bill_number",
                  "vote_question", "vote_desc", "vote_result"},
}
KNOWN_CHAMBERS = {"House", "Senate", "President"}
KNOWN_CAST_CODES = set(range(10))   # 0 not a member · 1-6 Yea/Nay · 7-8 Present · 9 Not Voting
KNOWN_PARTY_CODES = {"100", "200", "328"}
SCHEMA_REPORT = Path(os.environ.get("VOTES_SCHEMA_REPORT", "schema-drift.txt"))
MAX_DRIFT_LINES = 25   # backstop: an issue body has a size limit, and so does a reader
drift = {}   # key -> first message seen; insertion-ordered


def note_drift(key, msg):
    """Record one line per distinct `key`, keeping the first example seen.

    Keyed rather than deduped by message text: the messages carry an example
    rollcall or member, so a whole cast_code family changing would otherwise
    emit a line per row — hundreds of thousands of them, and an issue body far
    past GitHub's limit.
    """
    drift.setdefault(key, msg)


def fail_schema(msg):
    """Unrecoverable drift: record it for CI, then stop."""
    SCHEMA_REPORT.write_text(msg + "\n")
    raise SystemExit(f"! {msg}")


def as_int(raw, field, context):
    """int(raw), or None plus one drift line — never an exception.

    A traceback here is the worst outcome available: it writes no schema-drift.txt,
    so the workflow's report step finds nothing, files no issue, and skips the
    publish. The site then serves the last good build indefinitely with no signal.
    Keyed by field, not by value: a column that changes format changes it on every
    row, and one line per row would be hundreds of thousands of them.
    """
    try:
        return int(raw)
    except (ValueError, TypeError):
        note_drift(("non-numeric", field),
                   f"{field}: expected a number, got {raw!r} (e.g. {context}) "
                   f"— those rows are skipped")
        return None


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
# The Speaker is NOT here: it is elected on the floor, so it comes out of the
# rollcall data further down. These are the posts both parties fill in closed
# conference and caucus meetings, which produce no vote for Voteview to record.
#
# Keyed by Congress on purpose. ICPSR numbers follow the person, not the post, so
# one flat table carried forward would keep calling a former whip "Whip" for the
# rest of his career. An unlisted Congress flags nobody and says so at the end of
# the build — a missing badge is recoverable, a wrong one is a false claim.
LEADERSHIP_BY_CONGRESS = {
    119: {
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
            note_drift(("chamber", m["chamber"]),
                       f"members: unrecognised chamber {m['chamber']!r} "
                       f"(e.g. {m['bioname']}) — skipped")
        if m["party_code"] not in KNOWN_PARTY_CODES:
            note_drift(("party_code", m["party_code"]),
                       f"members: unrecognised party_code {m['party_code']!r} "
                       f"(e.g. {m['bioname']}) — member left unscored")
        if m["chamber"] not in ("House", "Senate"):
            continue
        if m["state_abbrev"] in DELEGATE_STATES:
            delegates.append(m["bioname"])
            continue
        icpsr = as_int(m["icpsr"], "members.icpsr", m["bioname"])
        if icpsr is None:
            continue
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
# Keyed by (icpsr, chamber), not icpsr: rollnumbers restart per chamber, so a
# member who serves in both during one Congress (a House member appointed to a
# Senate vacancy) would otherwise get a span mixing the two numbering schemes and
# a cast_count spanning both — making missed = eligible - cast_count negative.
span = {}                      # (icpsr, chamber) -> [first roll, last roll]
cast_count = Counter()         # (icpsr, chamber) -> decisive votes actually cast

for ch in ("H", "S"):
    for v in read_csv(f"{ch}{CONGRESS}_votes.csv", "votes"):
        chamber, roll = v["chamber"], as_int(v["rollnumber"], "votes.rollnumber",
                                             f"{v['chamber']} {v['rollnumber']!r}")
        if roll is None:
            continue
        if chamber not in KNOWN_CHAMBERS:
            # Not fatal and not skippable: dropping the rows would empty the index.
            # The damage is the join against rollcalls.csv, which is keyed by that
            # file's chamber value — dissents come out with no date, bill number,
            # question or result, and nothing else goes visibly wrong.
            note_drift(("votes chamber", chamber),
                       f"votes: unrecognised chamber {chamber!r} "
                       f"(e.g. rollcall {roll}) — dissents lose their bill context")
        rolls_seen[chamber].add(roll)
        raw = as_int(v["cast_code"], "votes.cast_code", f"{chamber} rollcall {roll}")
        if raw is None:
            continue
        if raw not in KNOWN_CAST_CODES:
            # normalize() passes an unknown code straight through, where it lands
            # in the "missed" bucket. Silent miscounting is the failure to catch.
            note_drift(("cast_code", raw),
                       f"votes: unrecognised cast_code {raw} "
                       f"(e.g. {chamber} rollcall {roll}) — counted as a missed vote")
        code = normalize(raw)
        if code == 0:
            continue
        icpsr = as_int(v["icpsr"], "votes.icpsr", f"{chamber} rollcall {roll}")
        if icpsr is None:
            continue
        key = (icpsr, chamber)
        if key in span:
            s = span[key]
            s[0], s[1] = min(s[0], roll), max(s[1], roll)
        else:
            span[key] = [roll, roll]
        if code in (1, 6):
            cast_count[key] += 1
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
for (icpsr, chamber), (first, last) in span.items():
    # Summed across chambers, so a member who switched mid-Congress is measured
    # against each chamber's own rollcalls over the stretch they served in it.
    held = sum(1 for r in rolls_seen[chamber] if first <= r <= last)
    eligible[icpsr] += held
    missed[icpsr] += held - cast_count[(icpsr, chamber)]

# ── Rollcall context ───────────────────────────────────────────────────────────
rollcalls = {}
for ch in ("H", "S"):
    for r in read_csv(f"{ch}{CONGRESS}_rollcalls.csv", "rollcalls"):
        roll = as_int(r["rollnumber"], "rollcalls.rollnumber",
                      f"{r['chamber']} {r['rollnumber']!r}")
        if roll is None:
            continue
        rollcalls[(r["chamber"], roll)] = {
            "date": r["date"],
            "bill_number": r["bill_number"],
            "vote_question": r["vote_question"],
            "vote_desc": r["vote_desc"][:240],
            "vote_result": r["vote_result"],
        }

# ── Speaker, derived from the floor ────────────────────────────────────────────
# The Speaker is the one leadership post decided by a recorded vote, so Voteview
# already has it: vote_question "Election of the Speaker", winner in vote_result
# as "Surname (ST)". Taking the latest such vote also makes a mid-Congress change
# — 2023's McCarthy to Johnson — resolve itself with no edit.
#
# Floor leaders and whips are not derivable at any price: both parties choose
# them in closed conference and caucus meetings that never reach a rollcall.
# There is exactly one "Election of the Speaker" question in the 119th and no
# Senate equivalent, so those eight stay in LEADERSHIP_BY_CONGRESS by hand.
SPEAKER_QUESTION = "Election of the Speaker"
elections = sorted((r["date"], r["vote_result"]) for r in rollcalls.values()
                   if r["vote_question"] == SPEAKER_QUESTION)
if elections:
    won = elections[-1][1]                       # most recent election wins
    surname, _, state = won.partition(" (")
    hits = [i for i, m in members_all.items()
            if m["chamber"] == "House" and m["state_abbrev"] == state.rstrip(")")
            and m["bioname"].split(",")[0].strip().casefold() == surname.strip().casefold()]
    if len(hits) == 1:
        LEADERSHIP = {**LEADERSHIP, hits[0]: "Speaker"}
        print(f"  Speaker (from rollcall): {members_all[hits[0]]['bioname']}")
    else:
        note_drift("speaker", f"could not match Speaker {won!r} to exactly one member "
                   f"({len(hits)} matched) — Speaker not flagged")
else:
    note_drift("speaker", f"no {SPEAKER_QUESTION!r} rollcall found — Speaker not flagged")

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

# ── Score bands ────────────────────────────────────────────────────────────────
# The one place the bands are defined. It ships in data.json as "tiers", and the
# site builds its legend, filter options, histogram axis, ranges and badge colors
# from that — so a rename lands here and nowhere else. Colors are var(--s{id}) in
# the stylesheet, keyed by id for the same reason.
# "max" is exclusive: a score landing exactly on it takes the next tier up.
# Everything downstream keys on the id, not the name: history snapshots archive
# label_dist by tier, so a rename can never desync an old snapshot from a new one.
TIERS = [
    {"id": 0, "max": 1.0,  "name": "Mindless Drone"},
    {"id": 1, "max": 5.0,  "name": "Bobblehead"},
    {"id": 2, "max": 10.0, "name": "Squeaky Wheel"},
    {"id": 3, "max": 20.0, "name": "Loose Cannon"},
    {"id": 4, "max": 30.0, "name": "Heretic"},
    {"id": 5, "max": None, "name": "Lone Wolf"},
]

def independence_tier(score_pct):
    s = round(score_pct, 2)   # round first: the label must agree with the number shown
    for t in TIERS:
        if t["max"] is None or s < t["max"]:
            return t["id"]

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
                # Not rendered, deliberately kept: it is the per-vote audit trail for
                # the one number the whole method turns on, and the self-check reads it
                # to prove the formula. The 310KB it costs is spread across 543 files
                # fetched one at a time — 585 bytes per visitor, which buys a lot here.
                "weight": round(weight, 4),
            })

    party_unity = 1 - n_defect / n_part if n_part else None
    p_dev = w_defect / w_total if w_total else None
    c_dev = n_cons_defect / n_cons if n_cons else None

    # The headline score is cohesion-weighted partisan deviation alone. It used to be
    # the mean of that and consensus deviation, but the two are not the same behavior
    # and do not share a scale: consensus deviation runs ~6x larger (median 3.5% vs
    # 0.6%), so the mean tracked it at r=0.96 and the party-discipline signal — the
    # actual subject of the site — contributed almost nothing. Consensus deviation is
    # published beside the score instead of blended into it.
    # No fallback to c_dev when a member has no partisan votes: scoring them on the
    # other scale is the same mix in miniature, and 30 consensus-only votes is a
    # reachable way to clear MIN_VOTES. They go out unscored; the site renders that.
    ind = p_dev

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
        "independence_score":  pct(ind),
        # The id only: "tiers" is three lines up in the same file, so the name is a
        # local join, and a stored copy of a derived value is the thing this index
        # deliberately does not publish.
        "independence_tier":   independence_tier(ind * 100) if ind is not None else None,
        # party_unity_pct is the one derived rate that stays: the page documents it
        # by name as the unweighted alternative to the score.
        "party_unity_pct":     pct(party_unity),
        "partisan_votes":      n_part,
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
        # A list indexed by tier id, not a dict keyed by name: this is archived
        # weekly and must survive a rename of the bands.
        "label_dist": [sum(1 for r in subset if r["independence_tier"] == t["id"])
                       for t in TIERS]
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
    "tiers":      TIERS,
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
# Skip anything unreadable rather than letting one bad file kill every future
# build: this runs before the workflow commits, so an unguarded raise here would
# fail identically every week until someone hand-edited the archive.
# The date is also the only snapshot field the site renders as text, so it is
# checked, not trusted: docs/history/ is committed to main, which makes a merged
# pull request a write path into the published page. A name that is not exactly
# YYYY-MM-DD, or a file whose own "date" disagrees with its name, is left out.
SNAPSHOT_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

history_index = []
for p in sorted(hist_dir.glob("*.json")):
    if p.name == "index.json":
        continue
    try:
        entry = json.loads(p.read_text())
        if not SNAPSHOT_DATE.match(p.stem) or entry.get("date") != p.stem:
            print(f"! Skipping misnamed snapshot {p.name}")
            note_drift(("snapshot", p.name),
                       f"history/{p.name} is not a YYYY-MM-DD snapshot of its own date "
                       f"(carries {entry.get('date')!r}) — left out of index.json")
            continue
        history_index.append({"date": p.stem, "congress": entry["congress"]})
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"! Skipping unreadable snapshot {p.name}: {type(e).__name__}")
        note_drift(("snapshot", p.name), f"history/{p.name} is unreadable "
                                         f"({type(e).__name__}) — left out of index.json")
(hist_dir / "index.json").write_text(json.dumps(history_index))

# A party code we don't map (Voteview's 328 for anyone who isn't Sanders or King)
# drops that member from the index entirely. Name them rather than quietly shrink.
unscored = sorted((i, m["bioname"]) for i, m in members_all.items()
                  if m["caucus"] not in ("D", "R"))
if unscored:
    print(f"! Unscored — no mapped caucus: {', '.join(n for _, n in unscored)}")
# Anyone new needs a person to decide which caucus, if any, they sit with — a log
# line nobody reads is how a member quietly vanishes from the index for a term.
# Already-reviewed cases stay off the report so the weekly run doesn't cry wolf.
newly_unscored = [n for i, n in unscored if i not in ACCEPTED_UNSCORED]
if newly_unscored:
    note_drift("unscored", f"unscored — no mapped caucus, decide and add to "
               f"DEM_CAUCUSING_INDEPENDENTS or ACCEPTED_UNSCORED: "
               f"{', '.join(newly_unscored)}")
if delegates:
    print(f"! Excluded — non-voting delegates: {', '.join(sorted(delegates))}")
if CONGRESS not in LEADERSHIP_BY_CONGRESS:
    print(f"! No leaders/whips table for Congress {CONGRESS} — Speaker still "
          f"derived from the floor vote. Add one to LEADERSHIP_BY_CONGRESS.")
    # Conference-elected posts need filling in once per Congress. Surfacing it as
    # an issue is what makes the January rollover a task, not a silent omission.
    note_drift("leadership", f"no leaders/whips table for Congress {CONGRESS} — the four floor "
               f"leaders and whips are unflagged (the Speaker is derived); add one "
               f"to LEADERSHIP_BY_CONGRESS in {Path(__file__).name}")

# The build still publishes on drift — the numbers are usually fine and stale is
# worse than slightly-off. CI turns a non-empty report into a GitHub issue.
if drift:
    msgs = list(drift.values())
    shown, rest = msgs[:MAX_DRIFT_LINES], len(msgs) - MAX_DRIFT_LINES
    lines = [f"- {d}" for d in shown]
    if rest > 0:
        lines.append(f"- ...and {rest} more kinds of drift (see the run log)")
    SCHEMA_REPORT.write_text("\n".join(lines) + "\n")
    print(f"! Upstream schema drift ({len(msgs)}) — wrote {SCHEMA_REPORT}")
    for d in msgs:
        print(f"    {d}")
elif SCHEMA_REPORT.exists():
    SCHEMA_REPORT.unlink()   # clean run — don't let a stale report reopen an issue

print(f"✓ Wrote {len(members)} members to {docs / 'data.json'}")
print(f"  Per-member files: {len(members)} · History snapshot: {today}")
print(f"  Updated: {output['updated_at']}")
