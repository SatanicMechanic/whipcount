# 119th Congress · Independence Index

A weekly-updated dashboard scoring every member of Congress on their independence from party leadership and bipartisan consensus.

**Live site:** `https://<your-username>.github.io/<repo-name>/`

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

**Attendance** is tracked separately and does *not* feed into the score. Over every rollcall their chamber held between their first and last recorded vote, `missed %` counts the ones with no Yea or Nay from them — Present, Not Voting, and the ones Voteview has no row for at all. That last case matters: the Speaker votes at his own discretion and is simply absent from about a quarter of House rollcalls, which a row count would score as perfect attendance. Bounding by first and last vote keeps members who arrived or left mid-congress from being charged for votes held outside their service. Showing up is not the same thing as being independent, so the two numbers stay apart.

Yea and Nay each cover a range of Voteview cast codes (1–3 and 4–6); all of them are counted. Only decisive Yea/Nay votes go into the score denominators.

**Floor leaders are flagged, not excluded.** The Speaker plus each party's leader and whip in each chamber carry a `leadership` field; the table badges them and the filter can hide or isolate them. They schedule the votes they then vote on, so their loyalty is partly loyalty to an agenda they set themselves — the Speaker scores as the most loyal House Republican, on the subset of votes he chose to cast. Excluding them outright would drop the first names anyone looks up, so the call is left to the reader.

Data sourced from [Voteview.com](https://voteview.com). Sanders and King are scored against the Democratic caucus they align with, but displayed as Independent. Anyone else Voteview codes as Independent has no caucus to measure deviation against and is left out of the index — the build prints their names so the omission is never silent. The President appears in Voteview's vote files (announced positions are recorded as cast codes) and is excluded entirely. So are the six non-voting delegates (DC, PR, VI, GU, AS, MP): barred from final-passage votes, their record is a Committee-of-the-Whole subset of a few dozen votes against a chamber norm of hundreds, and their 85%+ "missed" rate is the franchise rather than attendance. Ranking them beside voting members would compare two different things, so they are dropped from the index and from the party-cohesion tallies — the build prints their names.

## Setup

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
python3 tests/test_scoring.py     # self-check, no network needed
cd docs && python3 -m http.server 8000
# open http://localhost:8000
```

---
# disclaimer

The author has zero background in data science, python, or anything else.  This may all be hallucinatory AI slop.  Enter at your own risk. ooooh, scaaaaary...
