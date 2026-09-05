#!/usr/bin/env node
/**
 * Self-check for docs/index.html. Runs the page's own JS against synthetic
 * payloads under a stub DOM — no browser, no dependencies.
 *
 *     node tests/test_site.mjs
 *
 * Covers what breaks at the turn of a Congress:
 *   1. January payload — 0 members, every summary group {}. renderSummaryCards
 *      runs before the loading screen is hidden, so a throw there is a blank site.
 *   2. Trend chart — snapshots from the previous Congress must not be plotted,
 *      and snapshots taken before anyone cleared MIN_VOTES carry no values.
 *   3. End labels must not collide when the two caucus averages converge.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const GROUPS = ["all", "house", "senate", "house_dem", "house_rep", "senate_dem", "senate_rep"];
const SERIES = ["house_dem", "house_rep", "senate_dem", "senate_rep"];
const LABELS = ["Mindless Drone", "Yes Man", "Reluctant Rebel",
                "Frequent Dissenter", "Rebellious Streak", "Lone Wolf"];

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures++;
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? "  " + detail : ""}`);
};

// ── Stub DOM ─────────────────────────────────────────────────────────────────
const els = {};
const el = id => (els[id] ??= {
  id, innerHTML: "", textContent: "", value: "", style: {}, dataset: {},
  querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
});
globalThis.document = { getElementById: el, querySelectorAll: () => [], addEventListener: () => {} };
globalThis.window = { addEventListener: () => {} };
globalThis.location = { hash: "" };

let archive = {};                       // url -> object, for the stubbed fetch
globalThis.fetch = async url => {
  if (!(url in archive)) throw new Error(`404 ${url}`);
  return { json: async () => archive[url] };
};

const html = fs.readFileSync(path.join(ROOT, "docs", "index.html"), "utf8");
const src = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join("\n")
              .replace(/init\(\)\.catch\([\s\S]*$/, "");   // don't boot the page
const page = await import("data:text/javascript;base64," + Buffer.from(
  src + "\nexport {renderSummaryCards, renderTable, renderTrend, trendPanel, seriesPoints};" +
        "\nexport function setMembers(m){ allMembers = m; }").toString("base64"));

const group = avg => ({
  count: 100, avg_independence: avg, min_independence: 0, max_independence: avg * 3,
  avg_missed_pct: 3.1, label_dist: Object.fromEntries(LABELS.map(l => [l, 16])),
});
const snapshot = (date, congress, avgs) => ({
  date, congress,
  summary: Object.fromEntries(GROUPS.map(g => [g, avgs && SERIES.includes(g) ? group(avgs[g]) : {}])),
});

// ── 1. The January payload ───────────────────────────────────────────────────
console.log("\nJanuary payload (0 members, every summary group empty)");
const empty = { congress: 120, members: [], summary: Object.fromEntries(GROUPS.map(g => [g, {}])) };
try {
  page.setMembers(empty.members);
  page.renderSummaryCards(empty.summary);
  check("renderSummaryCards survives empty groups", true);
} catch (e) {
  check("renderSummaryCards survives empty groups", false, e.message);
}
check("cards show a zero count", els["summary-cards"].innerHTML.includes("0 members"));
check("no NaN/undefined in cards", !/NaN|undefined/.test(els["summary-cards"].innerHTML));
try {
  page.renderTable();
  check("renderTable survives zero members", els["tbody"].innerHTML.includes("NO MEMBERS SCORED YET"));
} catch (e) {
  check("renderTable survives zero members", false, e.message);
}

// ── 2. Trend chart ───────────────────────────────────────────────────────────
// The gate is coverage of the term, not a Congress number — so it must switch
// itself on for any Congress archived from the start, and stay off for one we
// only caught the tail of.
const DAY = 86400000;
const convened = n => Date.UTC(1789 + 2 * (n - 1), 0, 3);
const weekly = (startMs, n) => Array.from({ length: n }, (_, i) =>
  new Date(startMs + i * 7 * DAY).toISOString().slice(0, 10));

/** Load `dates` for `congress`; first `blank` snapshots score nobody. */
function loadArchive(congress, dates, blank = 0, extra = []) {
  archive = { "history/index.json": [
    ...dates.map(d => ({ date: d, congress })), ...extra.map(e => ({ ...e })),
  ] };
  dates.forEach((d, i) => {
    archive[`history/${d}.json`] = i < blank ? snapshot(d, congress, null)
      : snapshot(d, congress, { house_dem: 2 + i * .2, house_rep: 6 - i * .2,
                                senate_dem: 7 + i * .6, senate_rep: 5 + i * .05 });
  });
  for (const e of extra)
    archive[`history/${e.date}.json`] =
      snapshot(e.date, e.congress, Object.fromEntries(SERIES.map(k => [k, 99])));
}

const reset = () => { els["trend"] = undefined; els["trend-grid"] = undefined;
                      els["trend-sub"] = undefined; };
const shown = () => els["trend"]?.style.display === "block";

console.log("\nTrend chart · archived from the start of the term");
const dates120 = weekly(convened(120) + 8 * DAY, 10);   // first run 8 days in
loadArchive(120, dates120, 2, [{ date: "2026-12-21", congress: 119 },
                               { date: "2026-12-28", congress: 119 }]);
reset();
await page.renderTrend(120);
const grid = els["trend-grid"].innerHTML;
check("shown when the archive covers the term", shown());
check("two panels", (grid.match(/trend-panel/g) || []).length === 2);
check("four series drawn", (grid.match(/class="series"/g) || []).length === 4);
check("no NaN/undefined in the SVG", !/NaN|Infinity|undefined/.test(grid));
check("previous Congress not plotted", !grid.includes("99.0"));
check("empty snapshots dropped, not plotted as zero",
  (grid.match(/d="[^"]*"/g) || []).every(d => (d.match(/[ML]/g) || []).length === 8),
  "expect 8 points per line (10 snapshots - 2 empty)");
check("subtitle counts plotted points, not snapshots",
  els["trend-sub"].textContent.includes("8 weekly"), els["trend-sub"].textContent);

console.log("\nTrend chart · coverage gate");
// The 119th case: archiving began ~20 months into the term, so the cumulative
// averages had already converged and the line would be flat by construction.
reset();
loadArchive(119, weekly(convened(119) + 600 * DAY, 10));
await page.renderTrend(119);
check("hidden when archiving started mid-term", !shown());

// A dormant stretch is not a complete record.
reset();
loadArchive(120, [...weekly(convened(120) + 8 * DAY, 4),
                  ...weekly(convened(120) + 200 * DAY, 4)]);
await page.renderTrend(120);
check("hidden when the archive has a long gap", !shown());

// Nothing is pinned to the 120th — a well-archived 121st must light up too.
reset();
loadArchive(121, weekly(convened(121) + 12 * DAY, 8), 2);
await page.renderTrend(121);
check("shown for the 121st with no code change", shown());

reset();
loadArchive(120, weekly(convened(120) + 8 * DAY, 2));
await page.renderTrend(120);
check("hidden with only two points", !shown());

// ── 3. Label collision ───────────────────────────────────────────────────────
console.log("\nConverged caucuses (averages 0.04 apart)");
const conv = [0, 1, 2, 3].map(i => snapshot(`2027-0${i + 2}-01`, 120,
  Object.fromEntries(SERIES.map(k => [k, 5 + (k.endsWith("dem") ? .02 : -.02) * i]))));
const panel = page.trendPanel({ title: "House", d: "house_dem", r: "house_rep" }, conv);
const ys = [...panel.matchAll(/class="end-lbl" x="[\d.]+" y="([\d.]+)"/g)].map(m => +m[1]);
check("end labels stay >= 10px apart", Math.abs(ys[0] - ys[1]) >= 10,
  `gap ${Math.abs(ys[0] - ys[1]).toFixed(1)}px`);

const { W, P } = JSON.parse(panel.match(/data-geom='([^']+)'/)[1]);
const dots = [...panel.matchAll(/cx="([\d.]+)" cy="([\d.]+)"/g)].map(m => [+m[1], +m[2]]);
check("marks stay inside the plot box", dots.every(([x]) => x >= P.l - .5 && x <= W - P.r + .5));

console.log(failures ? `\n${failures} failure(s)` : "\nall good");
process.exit(failures ? 1 : 0);
