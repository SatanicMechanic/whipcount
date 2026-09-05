#!/usr/bin/env python3
"""
Self-check for analyze_votes.py. Builds a tiny synthetic Voteview dataset,
runs the real script against it offline, and asserts on the JSON it writes.

    python3 tests/test_scoring.py

Fixture: 7 Democrats (icpsr 1-5, 12, 13), 6 Republicans (icpsr 6-11), 40 House
rollcalls.
  rolls  1-20  partisan, D=Yea R=Nay. On roll 1 icpsr 9 breaks ranks (R 5-1).
  rolls 21-30  partisan, D=Nay (cast_code 4) vs R split 3-3 (tie -> Yea).
  rolls 31-40  consensus, everyone Yea. icpsr 4 dissents on roll 31,
               icpsr 5 on rolls 31-32. icpsr 11 is absent for rolls 38-40.
Three members have rows Voteview omits outright rather than coding 9, which is
how it records a member who simply didn't vote: icpsr 11 vanishes mid-tenure on
rolls 35-37 (the Speaker case — must count as missed), icpsr 12 arrives at roll
11 and icpsr 13 departs after roll 30 (neither may be charged for votes held
outside their service).
Some Yea/Nay votes use cast codes 2/4/5 so the normalization is exercised.
A president (icpsr 99912, chamber "President") casts Nay on rolls 1-20 the way
Voteview records announced positions; nothing about him may reach the output.
A non-voting delegate (icpsr 14, Guam, R) votes the Republican line on all 40
rollcalls — more than MIN_VOTES, and enough to move the R cohesion weights on
rolls 1 and 21 if he were ever counted. He must not be.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMS = [1, 2, 3, 4, 5, 12, 13]
REPS = [6, 7, 8, 9, 10, 11]
PRESIDENT = 99912
DELEGATE = 14


def build_fixture(data_dir):
    with open(data_dir / "H119_members.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "icpsr", "district_code", "state_abbrev",
                    "party_code", "bioname"])
        # Distinct surnames: the Speaker is matched from the election rollcall by
        # "Surname (ST)", so a fixture where everyone shares one would only ever
        # exercise the ambiguity guard.
        for i in DEMS + REPS:
            w.writerow([119, "House", i, i, "CA", 100 if i in DEMS else 200, f"MEMBER{i}, No{i}"])
        w.writerow([119, "President", PRESIDENT, 0, "USA", 200, "PRESIDENT, Fake"])
        w.writerow([119, "House", DELEGATE, 0, "GU", 200, "DELEGATE, Fake"])

    with open(data_dir / "H119_rollcalls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "rollnumber", "date", "bill_number",
                    "vote_result", "vote_desc", "vote_question"])
        for n in range(1, 41):
            # Rolls 1 and 20 are Speaker elections — the later one must win, the
            # way 2023's McCarthy-to-Johnson change resolves without an edit.
            if n in (1, 20):
                w.writerow([119, "House", n, f"2025-01-{n:02d}", "",
                            f"MEMBER{6 if n == 1 else 7} (CA)", "x" * 300,
                            "Election of the Speaker"])
            else:
                w.writerow([119, "House", n, f"2025-01-{n:02d}", f"HR{n}", "Passed",
                            "x" * 300, "On Passage"])

    rows = []
    for n in range(1, 21):                       # unanimous-ish partisan
        for i in DEMS:
            rows.append((n, i, 2 if (n, i) == (3, 1) else 1))
        for i in REPS:
            yea = (n == 1 and i == 9)
            rows.append((n, i, 1 if yea else (5 if (n, i) == (4, 6) else 6)))
        rows.append((n, PRESIDENT, 6))       # announced position, not a vote
        rows.append((n, DELEGATE, 6))        # with the R line — never counted
    for n in range(21, 31):                      # 50/50 R split, D on the other side
        for i in DEMS:
            rows.append((n, i, 4))
        for i in REPS:
            rows.append((n, i, 1 if i in (6, 7, 8) else 6))
        rows.append((n, DELEGATE, 1))            # would break the 3-3 tie if counted
    for n in range(31, 41):                      # consensus
        for i in DEMS:
            dissent = (i == 4 and n == 31) or (i == 5 and n in (31, 32))
            rows.append((n, i, 6 if dissent else 1))
        for i in REPS:
            if i == 11 and n >= 38:
                rows.append((n, i, 7 if n == 38 else 9))   # present / not voting
            else:
                rows.append((n, i, 1))
        rows.append((n, DELEGATE, 1))

    # Rollcalls a member has no row for at all. Voteview drops the row rather
    # than coding it 9, so the attendance denominator can't be a row count.
    OMITTED = {11: range(35, 38), 12: range(1, 11), 13: range(31, 41)}
    rows = [(n, i, c) for n, i, c in rows if n not in OMITTED.get(i, ())]

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


def check_congress_rollover(data_dir, out_dir):
    """The Congress is derived from the date, so the weekly job rolls itself over.

    analyze_votes.py is a top-to-bottom script — importing it runs the whole
    pipeline — so point it at the fixture and a scratch output dir first.
    Otherwise the import rewrites the real docs/ and drops a history snapshot
    into the repo.
    """
    os.environ.update(VOTES_OFFLINE="1", VOTES_CONGRESS="119",
                      VOTES_DATA_DIR=str(data_dir), VOTES_OUT_DIR=str(out_dir))
    sys.dont_write_bytecode = True   # don't leave a __pycache__ entry behind
    sys.path.insert(0, str(ROOT))
    from importlib import import_module
    congress_on = import_module("analyze_votes").congress_on

    cases = [
        (date(2025, 1, 2), 118),   # 119th has not convened yet
        (date(2025, 1, 3), 119),   # convenes January 3
        (date(2026, 12, 31), 119),
        (date(2027, 1, 2), 119),   # still the outgoing Congress
        (date(2027, 1, 3), 120),   # the 120th
        (date(2028, 12, 31), 120),
        (date(2029, 1, 3), 121),
    ]
    for day, want in cases:
        assert congress_on(day) == want, f"{day}: got {congress_on(day)}, want {want}"


def run_script(data_dir, out_dir, report):
    """Run analyze_votes.py offline; return (exit code, report text)."""
    env = {**os.environ, "VOTES_DATA_DIR": str(data_dir), "VOTES_OUT_DIR": str(out_dir),
           "VOTES_OFFLINE": "1", "VOTES_CONGRESS": "119",
           "VOTES_SCHEMA_REPORT": str(report)}
    p = subprocess.run([sys.executable, "analyze_votes.py"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode, (report.read_text() if report.exists() else "")


def check_schema_drift(tmp):
    """Voteview can change its files. Neither failure mode may be silent."""
    def fixture(mutate):
        d = tmp / f"drift{next(check_schema_drift.n)}"
        d.mkdir()
        build_fixture(d)
        mutate(d)
        return d

    def rewrite(path, fn):
        lines = path.read_text().splitlines()
        path.write_text("\n".join(fn(lines)) + "\n")

    # A renamed or dropped column is unrecoverable — stop, don't publish garbage.
    def rename_column(d):
        rewrite(d / "H119_votes.csv",
                lambda ls: [ls[0].replace("cast_code", "vote_cast")] + ls[1:])
    code, report = run_script(fixture(rename_column), tmp / "o1", tmp / "r1.txt")
    assert code != 0, "a renamed column must fail the build"
    assert "cast_code" in report and "renamed" in report, report

    # An unknown cast_code still publishes — normalize() drops it into "missed",
    # which is wrong but not catastrophic. It must not be silent.
    def odd_cast_code(d):
        rewrite(d / "H119_votes.csv",
                lambda ls: ls[:1] + [ls[1].rsplit(",", 1)[0] + ",11"] + ls[2:])
    code, report = run_script(fixture(odd_cast_code), tmp / "o2", tmp / "r2.txt")
    assert code == 0, "an unknown cast_code should still publish"
    assert "cast_code 11" in report, report

    # Same for a chamber or party_code Voteview has not used before.
    def odd_member(d):
        def fn(ls):
            f = ls[1].split(",")
            f[1], f[5] = "Territory", "999"     # chamber, party_code
            return ls[:1] + [",".join(f)] + ls[2:]
        rewrite(d / "H119_members.csv", fn)
    code, report = run_script(fixture(odd_member), tmp / "o3", tmp / "r3.txt")
    assert code == 0, "an unknown chamber should still publish"
    assert "chamber 'Territory'" in report and "party_code '999'" in report, report

    # A Speaker we can't match to exactly one member must be reported, not
    # guessed at and not fatal — the rest of the index is still correct.
    def unmatchable_speaker(d):
        rewrite(d / "H119_rollcalls.csv",
                lambda ls: [ls[0]] + [l.replace("MEMBER7 (CA)", "NOBODY (ZZ)") for l in ls[1:]])
    code, report = run_script(fixture(unmatchable_speaker), tmp / "o5", tmp / "r5.txt")
    assert code == 0, "an unmatchable Speaker should still publish"
    assert "Speaker" in report and "NOBODY (ZZ)" in report, report

    # One unreadable snapshot must not brick every future build. index.json is
    # rebuilt from the whole archive on each run, before the workflow commits, so
    # an unguarded raise here would fail identically every week.
    d = tmp / "corrupt"; d.mkdir(); build_fixture(d)
    out = tmp / "o6"; (out / "history").mkdir(parents=True)
    (out / "history" / "2020-01-01.json").write_text("{ not json")
    (out / "history" / "2020-01-02.json").write_text('{"date": "2020-01-02"}')   # no congress
    code, report = run_script(d, out, tmp / "r6.txt")
    assert code == 0, "a corrupt snapshot must not stop the build"
    index = json.loads((out / "history" / "index.json").read_text())
    assert all(e["congress"] == 119 for e in index), index
    assert len(index) == 1, f"only today's good snapshot belongs in the index: {index}"
    assert "2020-01-01.json" in report and "2020-01-02.json" in report, report

    # Drift is one line per distinct value, not per row: a whole cast_code family
    # changing would otherwise emit hundreds of thousands of lines.
    def many_bad_codes(d):
        def fn(ls):
            out = [ls[0]]
            for l in ls[1:]:
                f = l.split(",")
                f[4] = "11"          # same unknown code on every single row
                out.append(",".join(f))
            return out
        rewrite(d / "H119_votes.csv", fn)
    code, report = run_script(fixture(many_bad_codes), tmp / "o7", tmp / "r7.txt")
    assert code == 0
    hits = [l for l in report.splitlines() if "cast_code 11" in l]
    assert len(hits) == 1, f"expected one line for the code, got {len(hits)}"

    # A clean run must clear a stale report, or CI reopens an issue forever.
    d = tmp / "clean"; d.mkdir(); build_fixture(d)
    stale = tmp / "r4.txt"
    stale.write_text("left over from last week\n")
    code, report = run_script(d, tmp / "o4", stale)
    assert code == 0 and report == "", f"stale report not cleared: {report!r}"


check_schema_drift.n = iter(range(100))


SWITCHER = 60


def check_chamber_switcher(tmp):
    """A member serving in both chambers in one Congress.

    Rollnumbers restart per chamber, so a single span per member would mix the
    two numbering schemes while cast_count summed across both — driving
    missed = eligible - cast_count negative. Unreachable in the 119th (no icpsr
    votes in both chambers), so this fixture is the only thing that proves the
    per-(member, chamber) spans work at all.
    """
    d = tmp / "switch"; d.mkdir()
    build_fixture(d)                      # House: members 1-13, 40 rollcalls

    # Put the switcher in both chambers' member files, voting every rollcall.
    with open(d / "H119_members.csv", "a", newline="") as f:
        csv.writer(f).writerow([119, "House", SWITCHER, 1, "CA", 100, "SWITCHER, Pat"])
    with open(d / "H119_votes.csv", "a", newline="") as f:
        w = csv.writer(f)
        for n in range(1, 41):
            w.writerow([119, "House", n, SWITCHER, 1])

    # A real Senate: 2 D, 2 R over 40 partisan rollcalls, plus the switcher.
    with open(d / "S119_members.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "icpsr", "district_code", "state_abbrev",
                    "party_code", "bioname"])
        for i, party in ((51, 100), (52, 100), (53, 200), (54, 200)):
            w.writerow([119, "Senate", i, 0, "NY", party, f"SENATOR{i}, No{i}"])
        w.writerow([119, "Senate", SWITCHER, 0, "CA", 100, "SWITCHER, Pat"])
    with open(d / "S119_rollcalls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "rollnumber", "date", "bill_number",
                    "vote_result", "vote_desc", "vote_question"])
        for n in range(1, 41):
            w.writerow([119, "Senate", n, f"2025-02-{n % 28 + 1:02d}", f"S{n}",
                        "Passed", "x" * 50, "On Passage"])
    with open(d / "S119_votes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["congress", "chamber", "rollnumber", "icpsr", "cast_code"])
        for n in range(1, 41):
            for i in (51, 52, SWITCHER):
                w.writerow([119, "Senate", n, i, 1])
            for i in (53, 54):
                w.writerow([119, "Senate", n, i, 6])

    code, _ = run_script(d, tmp / "osw", tmp / "rsw.txt")
    assert code == 0
    members = {m["icpsr"]: m for m in
               json.loads((tmp / "osw" / "data.json").read_text())["members"]}

    # The invariant that used to break: you cannot miss votes you never cast.
    for m in members.values():
        assert m["missed_votes"] >= 0, f"{m['name']}: missed {m['missed_votes']}"
        assert m["missed_votes"] <= m["eligible_votes"], m["name"]

    # 40 House + 40 Senate rollcalls, present for all of them. Keyed by icpsr
    # alone this read 40 eligible against 80 cast — missed of -40.
    sw = members[SWITCHER]
    assert sw["eligible_votes"] == 80, sw["eligible_votes"]
    assert sw["missed_votes"] == 0, sw["missed_votes"]
    assert sw["missed_pct"] == 0.0


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir, out_dir = tmp / "data", tmp / "out"
        data_dir.mkdir()
        build_fixture(data_dir)
        check_congress_rollover(data_dir, tmp / "import_out")
        check_schema_drift(tmp)
        check_chamber_switcher(tmp)

        # VOTES_CONGRESS pins the fixture's Congress. Without it the script
        # derives one from today's date and this test would start looking for
        # H120_*.csv on 2027-01-03.
        env = {**os.environ, "VOTES_DATA_DIR": str(data_dir),
               "VOTES_OUT_DIR": str(out_dir), "VOTES_OFFLINE": "1",
               "VOTES_CONGRESS": "119"}
        subprocess.run([sys.executable, "analyze_votes.py"], cwd=ROOT, env=env, check=True)

        data = json.loads((out_dir / "data.json").read_text())
        by_icpsr = {m["icpsr"]: m for m in data["members"]}

        assert len(by_icpsr) == 13, by_icpsr.keys()

        # ── the president is not a member of Congress ─────────────────────────
        assert PRESIDENT not in by_icpsr
        assert not (out_dir / "members" / f"{PRESIDENT}.json").exists()
        # ── nor is a non-voting delegate, despite a full 40-vote record ───────
        # The weight assertions below are the real check that he never reached
        # the R tally: roll 1 would read 6/7 and roll 21 would be 4-3, not 3-3.
        assert DELEGATE not in by_icpsr
        assert not (out_dir / "members" / f"{DELEGATE}.json").exists()
        # group_stats returns {} for an empty group — the fixture has no Senate
        assert data["summary"]["all"]["count"] == sum(
            data["summary"][g].get("count", 0) for g in ("house", "senate"))

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
        # 5 of 6 on the majority position -> (5/6 - 0.5) * 2. Also proves the
        # president never joined the R tally: 6 of 7 would give 0.7143.
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
        # Denominator is every rollcall the chamber held between a member's
        # first and last appearance, not the rows they happen to have.
        assert by_icpsr[11]["eligible_votes"] == 40
        assert by_icpsr[11]["missed_votes"] == 6      # one 7, two 9s, three no-rows
        assert by_icpsr[11]["missed_pct"] == 15.0
        assert by_icpsr[1]["missed_pct"] == 0.0
        assert json.loads((out_dir / "members" / "11.json").read_text())["missed"] == {
            "eligible": 40, "missed": 6, "pct": 15.0}
        # Arriving late / leaving early is not absenteeism.
        assert by_icpsr[12]["eligible_votes"] == 30 and by_icpsr[12]["missed_pct"] == 0.0
        assert by_icpsr[13]["eligible_votes"] == 30 and by_icpsr[13]["missed_pct"] == 0.0
        # 13 left before any consensus vote — score falls back to partisan alone
        assert by_icpsr[13]["consensus_votes"] == 0
        assert by_icpsr[13]["consensus_deviation_pct"] is None
        assert by_icpsr[13]["independence_score"] == 0.0
        assert data["summary"]["all"]["avg_missed_pct"] == round(15.0 / 13, 2)

        # ── leadership flag ───────────────────────────────────────────────────
        # The Speaker is read off the floor vote, not a hand-kept table, so a
        # mid-Congress replacement needs no edit: roll 20 beats roll 1.
        assert by_icpsr[7]["leadership"] == "Speaker", by_icpsr[7]["leadership"]
        assert by_icpsr[6]["leadership"] is None, "the ousted Speaker must not keep the badge"
        assert json.loads(
            (out_dir / "members" / "7.json").read_text())["leadership"] == "Speaker"
        # No fixture icpsr is in LEADERSHIP_BY_CONGRESS, so the conference-elected
        # posts are all None — the flag never filters anyone out of the index.
        assert all("leadership" in m for m in data["members"])
        assert sum(m["leadership"] is not None for m in data["members"]) == 1

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
        index = json.loads((out_dir / "history" / "index.json").read_text())
        assert len(index) == 1
        assert index[0]["congress"] == 119   # lets a trend view pick one congress
        snap = json.loads((out_dir / "history" / f"{index[0]['date']}.json").read_text())
        assert snap["date"] == index[0]["date"] and snap["congress"] == 119
        # no build timestamp, so an unchanged re-run produces an identical file
        assert "updated_at" not in snap
        # summary only — the per-member series was dropped as noise
        assert set(snap) == {"date", "congress", "summary"}
        assert snap["summary"]["house_dem"]["count"] == 7

    print("all good")


if __name__ == "__main__":
    main()
