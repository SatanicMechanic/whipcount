# Congressional Independence Index

A weekly-updated dashboard scoring every member of Congress on their independence from party leadership and bipartisan consensus.

**Live site:** <https://congressionalindependence.deathbutt.org>

Nothing here is pinned to one Congress. Which one is in scope is derived from the date — the 1st convened in 1789 and each runs two years — so the weekly job rolls itself over on January 3 of every odd year. A Congress convenes weeks before Voteview publishes its first files, so until that data lands the build falls back one Congress and keeps serving the outgoing one, then switches on its own. `VOTES_CONGRESS` pins a specific one for rebuilding an old term.

## How scores work

Each member receives an **Independence Score** — the average of:

1. **Partisan deviation** — how often they voted against their party majority on contested partisan votes, *weighted by how united the party was*. Breaking ranks on a vote where your party was unanimous is real independence; being on the losing side of a 50/50 party split is not. Each vote's weight is `(party's share on the majority position - 0.5) x 2` — 1.0 when the party votes as a bloc, 0.0 when it's evenly divided. The plain unweighted number is still published as `party_unity_pct` if you prefer it.
2. **Consensus deviation** — how often they voted against bipartisan consensus (both parties agreed but they didn't). Not weighted.

| Score | Label |
|-------|-------|
| < 1%  | Mindless Drone |
| 1–5%  | Yes Man |
| 5–10% | Reluctant Rebel |
| 10–20%| Frequent Dissenter |
| 20–30%| Rebellious Streak |
| 30%+  | Lone Wolf |

**Attendance** is tracked separately and does *not* feed into the score. Over every rollcall their chamber held between their first and last recorded vote, `missed %` counts the ones with no Yea or Nay from them — Present, Not Voting, and the ones Voteview has no row for at all. That last case matters: the Speaker votes at his own discretion and is simply absent from a large share of House rollcalls, which a row count would score as perfect attendance. Bounding by first and last vote keeps members who arrived or left mid-congress from being charged for votes held outside their service. Showing up is not the same thing as being independent, so the two numbers stay apart.

Yea and Nay each cover a range of Voteview cast codes (1–3 and 4–6); all of them are counted. Only decisive Yea/Nay votes go into the score denominators.

**Floor leaders are flagged, not excluded.** The Speaker plus each party's leader and whip in each chamber carry a `leadership` field; the table badges them and the filter can hide or isolate them. They schedule the votes they then vote on, so their loyalty is partly loyalty to an agenda they set themselves — in the 119th the Speaker came out the single most loyal member of his caucus, on the subset of votes he chose to cast. Excluding them outright would drop the first names anyone looks up, so the call is left to the reader.

## History

Each weekly run archives a summary-only snapshot to `docs/history/`, tagged with its Congress. There is no per-member series: it was measured, and the median member moves 0.02 points week over week while the only large swings are new arrivals thrashing on a small denominator. The signal is at caucus level, so that is all the snapshots keep — about 110 lines a week instead of 4,500.

The **Caucus drift** chart plots average independence per caucus over time, House and Senate as separate panels. It appears on its own once the archive actually covers a term: at least three snapshots with scored members, the first within 90 days of the Congress convening, and no gap longer than 45 days. The gate is coverage rather than a Congress number because the problem it guards against is real: a cumulative average that had already converged before archiving began is a flat line by construction. Any term archived from its start gets a chart; one the archive only caught the tail of does not, and neither case needs a code change. (Archiving here began partway through the 119th, so that term shows no chart.)

**The Speaker is derived, not configured.** It is the only leadership post decided by a recorded vote, so Voteview already has it: `vote_question` is `Election of the Speaker` and the winner sits in `vote_result` as `Surname (ST)`. The build takes the most recent such vote and matches it to a member, which means a mid-Congress replacement — 2023's McCarthy to Johnson — resolves with no edit at all. If the name doesn't match exactly one member, the Speaker goes unflagged and it is reported rather than guessed.

The other four posts per chamber are not derivable at any price: both parties elect their floor leaders and whips in closed conference and caucus meetings that never produce a rollcall. `Election of the Speaker` is the only leadership question Voteview records, and the Senate has no equivalent at all. Those four stay in `LEADERSHIP_BY_CONGRESS`, keyed by Congress because ICPSR numbers follow the person and not the post — one flat table carried forward would keep calling a former whip "Whip" for the rest of his career. An unlisted Congress leaves them unflagged and files an issue, which is how each new term gets the table filled in.

## Watching the data source

Voteview is a third party that can change its files whenever it likes, so the build checks what it gets. A missing or renamed column stops the run — there is no sensible way to score without it. A `chamber`, `party_code`, or `cast_code` the script doesn't recognise is the more dangerous case, because nothing would otherwise go wrong: an unknown cast code just falls into the "missed" bucket and the index is quietly incorrect. Those are recorded and the build still publishes.

Anything unexpected lands in `schema-drift.txt`, and the weekly workflow turns a non-empty report into a GitHub issue — commenting on the open one rather than filing a new issue every Monday. A clean run deletes the file, so a fixed problem stops nagging. Two maintenance conditions ride the same channel: an unmapped member who needs a caucus decision, and a Congress with no leadership table. That last one is what makes the January rollover a task in the tracker rather than a silent omission. Cases already reviewed and accepted go in `ACCEPTED_UNSCORED` so the weekly run stays quiet.

Data sourced from [Voteview.com](https://voteview.com). Sanders and King are scored against the Democratic caucus they align with, but displayed as Independent. Anyone else Voteview codes as Independent has no caucus to measure deviation against and is left out of the index — the build prints their names so the omission is never silent. The President appears in Voteview's vote files (announced positions are recorded as cast codes) and is excluded entirely. So are the six non-voting delegates (DC, PR, VI, GU, AS, MP): barred from final-passage votes, their record is a Committee-of-the-Whole subset of a few dozen votes against a chamber norm of hundreds, and the resulting "missed" rate — well past 80% — is the franchise rather than attendance. Ranking them beside voting members would compare two different things, so they are dropped from the index and from the party-cohesion tallies — the build prints their names.

## Running your own copy

This one is already deployed; the steps below are for standing up a fork.

### 1. Create repo and push files

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### 2. Enable GitHub Pages

- Go to repo **Settings → Pages**
- Source: **Deploy from a branch**
- Branch: `site` · Folder: `/ (root)`
- The `site` branch is created by the Action on its first run — run the Action before setting this.
- Click **Save**

### 3. Run the Action once to generate initial data

- Go to **Actions → Update Voting Index**
- Click **Run workflow**

After ~30 seconds, `docs/data.json` will be committed and the site will be live.

The Action runs automatically every Monday at 8am UTC thereafter.

## Local development

```bash
uv run --with requests python3 analyze_votes.py          # generates docs/data.json, docs/members/, docs/history/
python3 tests/test_scoring.py     # scoring self-check, no network needed
node tests/test_site.mjs          # site self-check: empty-Congress render, trend chart
cd docs && python3 -m http.server 8000
# open http://localhost:8000
```

---
# disclaimer

The author has zero background in data science, python, or anything else.  This may all be hallucinatory AI slop.  Enter at your own risk. ooooh, scaaaaary...
