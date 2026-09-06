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

// The score bands are defined once, in analyze_votes.py, and shipped in data.json.
// Read them from there rather than restating them: that is the whole point of the
// tier table, and it means a rename in the Python is exercised here for free.
const TIERS = [...fs.readFileSync(path.join(ROOT, "analyze_votes.py"), "utf8")
  .matchAll(/\{"id": (\d+),\s+"max": ([\d.]+|None),\s+"name": "([^"]+)"\}/g)]
  .map(m => ({ id: +m[1], max: m[2] === "None" ? null : +m[2], name: m[3] }));
if (TIERS.length !== 6) throw new Error(`parsed ${TIERS.length} tiers from analyze_votes.py, want 6`);

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures++;
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? "  " + detail : ""}`);
};

// ── Stub DOM ─────────────────────────────────────────────────────────────────
const els = {};
const el = id => (els[id] ??= {
  id, innerHTML: "", textContent: "", value: "", style: {}, dataset: {}, attrs: {},
  querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
  setAttribute(k, v) { this.attrs[k] = v; },
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
// Anchored to line starts: the page's prose mentions <script> and <style> inside
// an HTML comment, and a loose match swallows the stylesheet along with it.
const src = [...html.matchAll(/^<script>$([\s\S]*?)^<\/script>$/gm)].map(m => m[1]).join("\n")
              .replace(/init\(\)\.catch\([\s\S]*$/, "");   // don't boot the page
const page = await import("data:text/javascript;base64," + Buffer.from(
  src + "\nexport {renderSummaryCards, renderTable, renderTrend, trendPanel, seriesPoints, updateSortHeaders};" +
        "\nexport function setMembers(m){ allMembers = m; }" +
        "\nexport function setSort(c,d){ sortCol = c; sortDir = d; }" +
        "\nexport function setTiers(t){ TIERS = t; }" +
        "\nexport {tierOptions};" +
        "\nexport {filteredMembers};").toString("base64"));

page.setTiers(TIERS);

const group = avg => ({
  count: 100, avg_independence: avg, min_independence: 0, max_independence: avg * 3,
  avg_missed_pct: 3.1, label_dist: TIERS.map(() => 16),   // indexed by tier id, not keyed by name
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
check("distribution survives zero members", els["hist-sub"].textContent === "no scored members");
check("no NaN/undefined in empty distribution", !/NaN|undefined/.test(els["hist-grid"].innerHTML));

// ── 1b. The distribution histogram ───────────────────────────────────────────
// The pile-up is the finding, so the chart has to survive one bin holding nearly
// everyone: bars scale to the largest bin, not the total.
console.log("\nDistribution histogram");
const member = (score, tier) => ({
  icpsr: Math.random(), name: "X", party: "D", state: "NY", chamber: "House",
  independence_score: score, independence_tier: tier, missed_pct: 0,
  party_unity_pct: 100, consensus_deviation_pct: 0, partisan_votes: 100,
  consensus_votes: 10, leadership: null,
});
const skewed = [
  ...Array.from({ length: 90 }, () => member(0.4, 0)),
  ...Array.from({ length: 9 },  () => member(3.0, 1)),
  member(33.0, 5),
];
page.setMembers(skewed);
page.renderTable();
const distGrid = els["hist-grid"].innerHTML;
check("all six bins rendered", (distGrid.match(/hist-bar/g) || []).length === 6);
check("largest bin is full height", /height:100%/.test(distGrid));
check("counts and shares shown", distGrid.includes("90 · 90%") && distGrid.includes("1 · 1%"));
check("empty bins render at zero, not NaN", distGrid.includes("height:0%") && !/NaN|undefined/.test(distGrid));
check("subtitle counts scored members", els["hist-sub"].textContent.startsWith("100 scored members"));
check("empty bins draw nothing at all", /height:0%;min-height:0px/.test(distGrid));
// Ranges are derived from the same cut points the tiers carry, so re-thresholding
// cannot leave a stale literal here. The first is escaped — a bare "<0.5%" would
// start a tag.
const firstRange = `&lt;${+TIERS[0].max}%`, lastRange = `${+TIERS.at(-2).max}%+`;
check("axis carries chip, range and every tier name",
  TIERS.every(t => els["hist-x"].innerHTML.includes(t.name)) &&
  els["hist-x"].innerHTML.includes(firstRange) &&
  els["hist-x"].innerHTML.includes(lastRange),
  els["hist-x"].innerHTML.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim());
check("bars have a text alternative", /Mindless Drone 90/.test(els["hist-grid"].attrs["aria-label"]));
check("filter options are generated from the same table",
  page.tierOptions() === TIERS.map(t => `<option value="${t.id}">${t.name}</option>`).join(""),
  page.tierOptions());

// Picking one bin from the label filter must not collapse the chart to that bin —
// the histogram is the context for the filtered table, so it ignores that one filter.
els["filter-label"].value = "1";   // tier id, not the display name
page.renderTable();
check("label filter narrows the table", els["tbody"].innerHTML.match(/<tr/g).length === 9);
check("label filter leaves the distribution whole",
  els["hist-sub"].textContent.startsWith("100 scored members") &&
  els["hist-grid"].innerHTML.includes("90 · 90%"));
els["filter-label"].value = "";

// ── 1c. Consensus deviation is never ranked across chambers ─────────────────
// The Senate's consensus bucket is a ~65-vote procedural residue against the
// House's ~191 suspension bills, so a combined leaderboard on that column would
// compare two different things. Sorting it groups by chamber instead.
console.log("\nConsensus deviation sort");
const mixed = [
  { ...member(1, 1), name: "SenHigh", chamber: "Senate", consensus_deviation_pct: 60, consensus_votes: 65 },
  { ...member(1, 1), name: "HouseHigh", chamber: "House", consensus_deviation_pct: 40, consensus_votes: 191 },
  { ...member(1, 1), name: "HouseLow", chamber: "House", consensus_deviation_pct: 5, consensus_votes: 191 },
  { ...member(1, 1), name: "SenLow", chamber: "Senate", consensus_deviation_pct: 2, consensus_votes: 65 },
];
page.setMembers(mixed);
page.setSort("consensus_deviation_pct", -1);
const order = page.filteredMembers().map(m => m.name);
check("chambers grouped, not interleaved", order.join() === "HouseHigh,HouseLow,SenHigh,SenLow", order.join());
check("still ranked within each chamber",
  order.indexOf("HouseHigh") < order.indexOf("HouseLow") &&
  order.indexOf("SenHigh") < order.indexOf("SenLow"));
page.renderTable();
check("denominator shown beside the rate", els["tbody"].innerHTML.includes("of 65"));
page.setSort("independence_score", -1);

// ── 1d. Sort headers ─────────────────────────────────────────────────────────
// A member page leaves its own <thead><th> in the DOM after you navigate back —
// they have no .sort-arrow, so anything selecting "thead th" globally throws on
// the next header click and the table silently stops sorting.
console.log("\nSort headers");
const th = (col) => ({
  dataset: col ? { col } : {},
  classList: { toggle(){} },
  attrs: {}, setAttribute(k, v) { this.attrs[k] = v; },
  querySelector: sel => col && sel === ".sort-arrow" ? { textContent: "" } : null,
});
const mainThs = [th("name"), th("independence_score")];
const detailTh = th(null);                       // a member page's Date column
document.querySelectorAll = sel =>
  sel.includes("#main-table") ? mainThs : [...mainThs, detailTh];
let threw = null;
try { page.updateSortHeaders(); } catch (e) { threw = e.message; }
check("survives a member page's headers still in the DOM", threw === null, threw || "");
check("aria-sort stamped on the sorted column",
  mainThs[1].attrs["aria-sort"] === "descending", JSON.stringify(mainThs[1].attrs));
check("aria-sort not stamped on unrelated tables",
  detailTh.attrs["aria-sort"] === undefined, JSON.stringify(detailTh.attrs));
document.querySelectorAll = () => [];

// ── 1e. No third-party origins ───────────────────────────────────────────────
// The CSP is meta-only (GitHub Pages sets no headers), so it is the whole defence
// and it says 'self'. A resource pointing anywhere else is dead on arrival — the
// page would silently lose its fonts rather than fall back to them.
console.log("\nThird-party resources");
const external = [...html.matchAll(/(?:src|href)="(https?:\/\/[^"]+)"/g)]
  .map(m => m[1])
  .filter(u => !/^https:\/\/voteview\.com/.test(u));   // the data credit in the footer, an <a>
check("no third-party resources loaded", external.length === 0, external.join(" "));
check("CSP declared", /http-equiv="Content-Security-Policy"/.test(html));
check("fonts are served from this repo",
  (html.match(/src: url\(fonts\/[^)]+\.woff2\)/g) || []).length >= 4);

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
const shown = () => els["trend"]?.dataset.state === "ready";

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

// A snapshot whose own date field is unparseable makes every comparison in the
// coverage gate NaN — which is false, so it used to walk through both gates and
// plot NaN coordinates. The gate has to fail closed on an archive it can't read.
reset();
const good120 = weekly(convened(120) + 8 * DAY, 6);
loadArchive(120, good120);
archive[`history/${good120[2]}.json`].date = "not-a-date";
await page.renderTrend(120);
check("hidden when a snapshot's date is unparseable", !shown());
check("no NaN plotted from a bad date",
  !/NaN/.test(els["trend-grid"]?.innerHTML ?? ""));   // untouched when the gate holds

// docs/history/ is committed to main, so a merged pull request is a write path
// into the page. Date.parse is not a filter — V8 skips parenthesised comments, so
// this payload parses fine — which is why the gate checks the shape instead.
reset();
const XSS = "Jan 1 2026 (<img src=x onerror=alert(1)>)";
loadArchive(120, good120);
archive[`history/${good120[2]}.json`].date = XSS;
await page.renderTrend(120);
check("a date that parses but isn't YYYY-MM-DD is refused", !shown());
check("payload never reaches the page",
  !(els["trend-grid"]?.innerHTML ?? "").includes("onerror"));

// ── 3. Label collision ───────────────────────────────────────────────────────
console.log("\nConverged caucuses (averages 0.04 apart)");
const conv = [0, 1, 2, 3].map(i => snapshot(`2027-0${i + 2}-01`, 120,
  Object.fromEntries(SERIES.map(k => [k, 5 + (k.endsWith("dem") ? .02 : -.02) * i]))));
const panel = page.trendPanel({ title: "House", d: "house_dem", r: "house_rep" }, conv);
const ys = [...panel.matchAll(/class="end-lbl" x="[\d.]+" y="([\d.]+)"/g)].map(m => +m[1]);
check("end labels stay >= 10px apart", Math.abs(ys[0] - ys[1]) >= 10,
  `gap ${Math.abs(ys[0] - ys[1]).toFixed(1)}px`);

// data-* are HTML-escaped now (a snapshot date is untrusted text); the browser
// decodes them on dataset access, so the test has to decode them too.
const unesc = t => t.replace(/&quot;/g, '"').replace(/&lt;/g, "<")
                    .replace(/&gt;/g, ">").replace(/&amp;/g, "&");
check("chart data attributes are escaped", /data-geom="[^']*"/.test(panel));
const { W, P } = JSON.parse(unesc(panel.match(/data-geom="([^"]+)"/)[1]));
const dots = [...panel.matchAll(/cx="([\d.]+)" cy="([\d.]+)"/g)].map(m => [+m[1], +m[2]]);
check("marks stay inside the plot box", dots.every(([x]) => x >= P.l - .5 && x <= W - P.r + .5));

console.log(failures ? `\n${failures} failure(s)` : "\nall good");
process.exit(failures ? 1 : 0);
