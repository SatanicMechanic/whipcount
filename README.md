# 119th Congress · Independence Index

A weekly-updated dashboard scoring every member of Congress on their independence from party leadership and bipartisan consensus.

**Live site:** `https://<your-username>.github.io/<repo-name>/`

## How scores work

Each member receives an **Independence Score** — the average of:
1. **Partisan deviation** — how often they voted against their party majority on contested partisan votes
2. **Consensus deviation** — how often they voted against bipartisan consensus (both parties agreed but they didn't)

| Score | Label |
|-------|-------|
| < 1%  | Mindless Drone |
| 1–5%  | Yes Man |
| 5–10% | Reluctant Rebel |
| 10–20%| Frequent Dissenter |
| 20–30%| Rebellious Streak |
| 30%+  | Lone Wolf |

Data sourced from [Voteview.com](https://voteview.com). Sanders and King are scored against the Democratic caucus they align with, but displayed as Independent.

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
- Branch: `main` · Folder: `/docs`
- Click **Save**

### 3. Run the Action once to generate initial data

- Go to **Actions → Update Voting Index**
- Click **Run workflow**

After ~30 seconds, `docs/data.json` will be committed and the site will be live.

The Action runs automatically every Monday at 8am UTC thereafter.

## Local development

```bash
pip install pandas requests
python analyze_votes.py        # generates docs/data.json
cd docs && python -m http.server 8000
# open http://localhost:8000
```

---
# disclaimer

The author has zero background in data science, python, or anything else.  This may all be hallucinatory AI slop.  Enter at your own risk. ooooh, scaaaaary...
