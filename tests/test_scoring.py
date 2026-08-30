#!/usr/bin/env python3
"""
Self-check for analyze_votes.py. Builds a tiny synthetic Voteview dataset,
runs the real script against it offline, and asserts on the JSON it writes.

    python3 tests/test_scoring.py

Fixture: 5 Democrats (icpsr 1-5), 6 Republicans (icpsr 6-11), 40 House rollcalls.
  rolls  1-20  partisan, D=Yea R=Nay. On roll 1 icpsr 9 breaks ranks (R 5-1).
  rolls 21-30  partisan, D=Nay (cast_code 4) vs R split 3-3 (tie -> Yea).
  rolls 31-40  consensus, everyone Yea. icpsr 4 dissents on roll 31,
               icpsr 5 on rolls 31-32. icpsr 11 is absent for rolls 38-40.
Some Yea/Nay votes use cast codes 2/4/5 so the normalization is exercised.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMS = [1, 2, 3, 4, 5]
REPS = [6, 7, 8, 9, 10, 11]


def build_fixture(data_dir):
    with open(data_dir / "H119_members.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "icpsr", "district_code", "state_abbrev",
                    "party_code", "bioname"])
        for i in DEMS + REPS:
            w.writerow([119, "House", i, i, "CA", 100 if i in DEMS else 200, f"MEMBER, No{i}"])

    with open(data_dir / "H119_rollcalls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "rollnumber", "date", "bill_number",
                    "vote_result", "vote_desc", "vote_question"])
        for n in range(1, 41):
            w.writerow([119, "House", n, f"2025-01-{n:02d}", f"HR{n}", "Passed",
                        "x" * 300, "On Passage"])

    rows = []
    for n in range(1, 21):                       # unanimous-ish partisan
        for i in DEMS:
            rows.append((n, i, 2 if (n, i) == (3, 1) else 1))
        for i in REPS:
            yea = (n == 1 and i == 9)
            rows.append((n, i, 1 if yea else (5 if (n, i) == (4, 6) else 6)))
    for n in range(21, 31):                      # 50/50 R split, D on the other side
        for i in DEMS:
            rows.append((n, i, 4))
        for i in REPS:
            rows.append((n, i, 1 if i in (6, 7, 8) else 6))
    for n in range(31, 41):                      # consensus
        for i in DEMS:
            dissent = (i == 4 and n == 31) or (i == 5 and n in (31, 32))
            rows.append((n, i, 6 if dissent else 1))
        for i in REPS:
            if i == 11 and n >= 38:
                rows.append((n, i, 7 if n == 38 else 9))   # present / not voting
            else:
                rows.append((n, i, 1))

    with open(data_dir / "H119_votes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "rollnumber", "icpsr", "cast_code"])
        for n, i, code in rows:
            w.writerow([119, "House", n, i, code])

    # Empty Senate files — the script reads both chambers.
    with open(data_dir / "S119_members.csv", "w") as f:
        f.write("congress,chamber,icpsr,district_code,state_abbrev,party_code,bioname\n")
    with open(data_dir / "S119_rollcalls.csv", "w") as f:
        f.write("congress,chamber,rollnumber,date,bill_number,vote_result,vote_desc,vote_question\n")
    with open(data_dir / "S119_votes.csv", "w") as f:
        f.write("congress,chamber,rollnumber,icpsr,cast_code\n")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir, out_dir = tmp / "data", tmp / "out"
        data_dir.mkdir()
        build_fixture(data_dir)

        env = {**os.environ, "VOTES_DATA_DIR": str(data_dir), "VOTES_OUT_DIR": str(out_dir)}
        subprocess.run([sys.executable, "analyze_votes.py"], cwd=ROOT, env=env, check=True)

        data = json.loads((out_dir / "data.json").read_text())
        by_icpsr = {m["icpsr"]: m for m in data["members"]}

        assert len(by_icpsr) == 11, by_icpsr.keys()

        # ── cast-code normalization ───────────────────────────────────────────
        # Rolls 21-30 are only classifiable if cast_code 4 counts as Nay, and
        # rolls 3/4 only stay unanimous if 2 counts as Yea and 5 as Nay.
        assert by_icpsr[1]["partisan_votes"] == 30, by_icpsr[1]["partisan_votes"]
        assert by_icpsr[1]["consensus_votes"] == 10
        assert by_icpsr[1]["party_unity_pct"] == 100.0
        assert by_icpsr[6]["party_unity_pct"] == 100.0   # the cast_code 5 voter

        # ── weight formula ────────────────────────────────────────────────────
        d9 = json.loads((out_dir / "members" / "9.json").read_text())["dissents"]
        weights = {(d["rollnumber"], d["kind"]): d["weight"] for d in d9}
        # 50/50 party split -> weight 0.0
        assert weights[(21, "partisan")] == 0.0, weights[(21, "partisan")]
        # 5 of 6 on the majority position -> (5/6 - 0.5) * 2
        assert weights[(1, "partisan")] == 0.6667, weights[(1, "partisan")]
        # icpsr 9 defected 11 of 30 partisan votes, but 10 of those carried zero
        # weight. Denominator = 0.6667 (roll 1) + 19 unanimous rolls at weight 1.0.
        assert by_icpsr[9]["party_unity_pct"] == 63.33, by_icpsr[9]["party_unity_pct"]
        expected = round(100 * (2 / 3) / (2 / 3 + 19), 2)
        assert by_icpsr[9]["weighted_partisan_deviation_pct"] == expected == 3.39
        # A defection on a zero-cohesion vote costs nothing at all.
        assert by_icpsr[10]["party_unity_pct"] == 66.67
        assert by_icpsr[10]["weighted_partisan_deviation_pct"] == 0.0
        # Consensus dissents are unweighted.
        d4 = json.loads((out_dir / "members" / "4.json").read_text())["dissents"]
        assert [d["kind"] for d in d4] == ["consensus"] and d4[0]["weight"] == 1.0

        # ── missed_pct ────────────────────────────────────────────────────────
        assert by_icpsr[11]["eligible_votes"] == 40
        assert by_icpsr[11]["missed_votes"] == 3          # one 7, two 9s
        assert by_icpsr[11]["missed_pct"] == 7.5
        assert by_icpsr[1]["missed_pct"] == 0.0
        assert json.loads((out_dir / "members" / "11.json").read_text())["missed"] == {
            "eligible": 40, "missed": 3, "pct": 7.5}
        assert data["summary"]["all"]["avg_missed_pct"] == round(7.5 / 11, 2)

        # ── label thresholds (boundaries are exclusive-below) ─────────────────
        assert by_icpsr[1]["independence_score"] == 0.0
        assert by_icpsr[1]["independence_label"] == "Mindless Drone"
        assert by_icpsr[9]["independence_score"] == 1.69
        assert by_icpsr[9]["independence_label"] == "Yes Man"
        assert by_icpsr[4]["independence_score"] == 5.0      # exactly on 5
        assert by_icpsr[4]["independence_label"] == "Reluctant Rebel"
        assert by_icpsr[5]["independence_score"] == 10.0     # exactly on 10
        assert by_icpsr[5]["independence_label"] == "Frequent Dissenter"

        # ── bill context joined onto dissents ─────────────────────────────────
        assert d4[0]["bill_number"] == "HR31" and d4[0]["vote_result"] == "Passed"
        assert d4[0]["vote_question"] == "On Passage"
        assert len(d4[0]["vote_desc"]) == 240                # truncated
        assert d4[0]["member_vote"] == "Nay" and d4[0]["party_position"] == "Yea"
        # dissents sorted by date descending
        dates = [d["date"] for d in d9]
        assert dates == sorted(dates, reverse=True)

        # ── history snapshot ──────────────────────────────────────────────────
        dates = json.loads((out_dir / "history" / "index.json").read_text())
        assert len(dates) == 1
        snap = json.loads((out_dir / "history" / f"{dates[0]}.json").read_text())
        assert snap["date"] == dates[0] and snap["congress"] == 119
        assert set(snap["members"][0]) == {
            "icpsr", "name", "party", "chamber", "independence_score", "missed_pct"}

    print("all good")


main()
