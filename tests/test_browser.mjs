/**
 * Renders the real page in a real browser and asserts it actually works.
 *
 *     node tests/test_browser.mjs
 *
 * test_site.mjs evaluates the page's functions against a stub DOM, which cannot
 * see anything the browser decides: whether the Content-Security-Policy blocks
 * the page's own inline script, whether a font 404s, whether init() throws on
 * real data.json. Those are exactly the failures that ship silently, because the
 * weekly job publishes on a cron with nobody watching.
 *
 * No dependencies, deliberately — the other two suites have none and CI installs
 * nothing for them. Chrome speaks its own remote-debugging protocol over a
 * WebSocket, and node has had a WebSocket client built in since v22, so driving
 * it directly costs less than adding a browser-automation dependency and the
 * ~300MB of browser binaries it downloads. Set CHROME_PATH to override discovery.
 */
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const TIMEOUT_MS = 60_000;

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures++;
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? "  " + detail : ""}`);
};

if (typeof WebSocket !== "function") {
  console.error(`node ${process.version} has no WebSocket; this check needs v22+`);
  process.exit(1);
}

// ── Find a Chromium ──────────────────────────────────────────────────────────
// Any Chromium serves: the protocol is the same and the point is to exercise the
// page's own behaviour, not one vendor's rendering.
function findBrowser() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const apps = fs.existsSync("/Applications") ? fs.readdirSync("/Applications") : [];
  const rank = a => ["Google Chrome", "Chromium", "Microsoft Edge", "Brave"]
    .findIndex(p => a.startsWith(p));
  const candidates = [
    ...apps.filter(a => rank(a) >= 0).sort((a, b) => rank(a) - rank(b))
           .map(a => `/Applications/${a}/Contents/MacOS/${a.replace(/\.app$/, "")}`),
    "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge",
  ];
  return candidates.find(c => fs.existsSync(c));
}

// ── Build a real site to serve ───────────────────────────────────────────────
// The same synthetic Voteview data the scoring self-check uses, run through the
// real script, next to the real index.html. Nothing is mocked.
function buildSite() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "votes-browser-"));
  const data = path.join(tmp, "data"), site = path.join(tmp, "site");
  fs.mkdirSync(data);
  const python = (args, env) => {
    try {
      execFileSync("python3", args, { env: { ...process.env, ...env }, stdio: "pipe" });
    } catch (e) {
      console.error(`\nfixture build failed:\n${e.stderr?.toString() ?? e.message}`);
      process.exit(1);
    }
  };
  python(["-c",
    "import sys, pathlib; sys.path.insert(0, sys.argv[1]);" +
    "from test_scoring import build_fixture; build_fixture(pathlib.Path(sys.argv[2]))",
    path.join(ROOT, "tests"), data]);
  python([path.join(ROOT, "analyze_votes.py")],
    { VOTES_OFFLINE: "1", VOTES_CONGRESS: "119", VOTES_DATA_DIR: data,
      VOTES_OUT_DIR: site, VOTES_SCHEMA_REPORT: path.join(tmp, "drift.txt") });
  for (const f of ["index.html", "changes.html"])
    fs.copyFileSync(path.join(ROOT, "docs", f), path.join(site, f));
  fs.cpSync(path.join(ROOT, "docs", "fonts"), path.join(site, "fonts"), { recursive: true });
  return { tmp, site };
}

const MIME = { ".html": "text/html", ".json": "application/json",
               ".woff2": "font/woff2", ".svg": "image/svg+xml", ".css": "text/css" };

function serve(dir) {
  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "") || "index.html";
    const file = path.join(dir, rel);
    // The page is served from a real origin so 'self' means something; keep it
    // inside the served tree anyway.
    if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(fs.readFileSync(file));
  });
  return new Promise(r => server.listen(0, "127.0.0.1", () => r(server)));
}

// ── Minimal CDP client ───────────────────────────────────────────────────────
async function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const pending = new Map();
  const events = [];
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) {
      const { res, rej } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? rej(new Error(m.error.message)) : res(m.result);
    } else if (m.method) {
      events.push(m);
    }
  };
  let id = 0;
  const send = (method, params = {}) => new Promise((res, rej) => {
    pending.set(++id, { res, rej });
    ws.send(JSON.stringify({ id, method, params }));
  });
  return { send, events, close: () => ws.close() };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function poll(fn, what, ms = 15_000) {
  const until = Date.now() + ms;
  for (;;) {
    if (await fn()) return true;
    if (Date.now() > until) throw new Error(`timed out waiting for ${what}`);
    await sleep(150);
  }
}

// ── Run ──────────────────────────────────────────────────────────────────────
const browser = findBrowser();
if (!browser) {
  console.error("No Chromium found. Install Chrome/Chromium or set CHROME_PATH.");
  process.exit(1);
}

const guard = setTimeout(() => {
  console.error(`\nhung for ${TIMEOUT_MS}ms — failing rather than blocking CI`);
  process.exit(1);
}, TIMEOUT_MS).unref?.() ?? null;

const { tmp, site } = buildSite();
const server = await serve(site);
const origin = `http://127.0.0.1:${server.address().port}`;
const profile = path.join(tmp, "profile");

const chrome = spawn(browser, [
  "--headless=new", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
  `--user-data-dir=${profile}`, "--remote-debugging-port=0", "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });

// The chosen port is announced on stderr; asking for 0 avoids colliding with
// whatever else is listening on a developer's machine.
const browserWs = await new Promise((res, rej) => {
  let buf = "";
  chrome.stderr.on("data", d => {
    buf += d;
    const m = buf.match(/ws:\/\/[^\s]+/);
    if (m) res(m[0]);
  });
  chrome.on("exit", c => rej(new Error(`browser exited (${c}) before listening`)));
});

// That endpoint is the browser target, which has no Page or Runtime domain.
// The tab's own endpoint does, and /json lists it.
const port = new URL(browserWs).port;
let pageWs;
await poll(async () => {
  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  pageWs = targets.find(t => t.type === "page")?.webSocketDebuggerUrl;
  return !!pageWs;
}, "a page target to appear", 10_000);

const cdp = await connect(pageWs);
await cdp.send("Page.enable");
await cdp.send("Runtime.enable");
await cdp.send("Network.enable");
await cdp.send("Log.enable");        // CSP violations arrive here, not on console

await cdp.send("Page.navigate", { url: origin });

const evaluate = async expr => (await cdp.send("Runtime.evaluate", {
  expression: expr, returnByValue: true, awaitPromise: true,
})).result.value;

console.log(`\nBrowser: ${path.basename(browser)} · serving ${site}`);

try {
  await poll(async () => await evaluate(`document.getElementById("app").style.display === "block"`),
             "init() to finish and reveal #app");
  check("page boots and reveals the app", true);
} catch (e) {
  check("page boots and reveals the app", false, e.message);
}

// Give late-arriving violations and requests a moment to land.
await sleep(400);

// ── What the stub DOM cannot see ─────────────────────────────────────────────
const logs = cdp.events.filter(e => e.method === "Log.entryAdded").map(e => e.params.entry);
const csp = logs.filter(l => l.source === "security" || /Content Security Policy/i.test(l.text));
check("no CSP violations", csp.length === 0, csp.map(l => l.text).join(" | "));

const errors = logs.filter(l => l.level === "error" && !csp.includes(l));
const thrown = cdp.events.filter(e => e.method === "Runtime.exceptionThrown")
  .map(e => e.params.exceptionDetails.exception?.description ?? e.params.exceptionDetails.text);
check("no console errors", errors.length === 0, errors.map(l => l.text).join(" | "));
check("no uncaught exceptions", thrown.length === 0, thrown.join(" | "));

const failed = cdp.events.filter(e => e.method === "Network.loadingFailed")
  .map(e => e.params.errorText);
const notOk = cdp.events.filter(e => e.method === "Network.responseReceived"
  && e.params.response.status >= 400)
  .map(e => `${e.params.response.status} ${e.params.response.url}`);
check("every request succeeded", failed.length === 0 && notOk.length === 0,
  [...failed, ...notOk].join(" | "));

// Fonts are the reason a request check matters: they were third-party until
// today, and a missing woff2 degrades silently to a system font.
const fonts = await evaluate(`(async () => { await document.fonts.ready;
  return [...document.fonts].filter(f => f.status === "loaded").map(f => f.family); })()`);
// All three are used on the page: Bebas in the h1, DM Mono in every number,
// DM Sans in the body. Fewer means one 404'd and fell back silently.
check("all three self-hosted fonts loaded", new Set(fonts).size === 3,
  [...new Set(fonts)].join(", "));

// ── What actually rendered ───────────────────────────────────────────────────
const view = await evaluate(`JSON.stringify({
  rows:   document.querySelectorAll("#tbody tr[data-icpsr]").length,
  cols:   document.querySelectorAll("#main-table thead th").length,
  bars:   document.querySelectorAll(".hist-bar").length,
  axis:   document.querySelectorAll(".hist-x > div").length,
  opts:   document.querySelectorAll("#filter-label option").length,
  badges: document.querySelectorAll("#tbody .badge-dot").length,
  sticky: getComputedStyle(document.querySelector("#main-table thead th")).position,
  ghost:  !!document.querySelector(".trend-ghost"),
  banner: !document.getElementById("banner").hidden,
  blink:  document.querySelector("#banner a")?.getAttribute("href") ?? "",
  jump:   (() => { const a = document.querySelector("a.jump");
            return !!a && !!document.querySelector(a.getAttribute("href")); })(),
  trend:  document.getElementById("trend").dataset.state ?? "",
  hpad:   document.body.scrollWidth <= document.documentElement.clientWidth,
})`).then(JSON.parse);

check("table rendered rows", view.rows > 0, `${view.rows} rows · ${view.cols} columns`);
check("distribution rendered every tier", view.bars === 6 && view.axis === 6,
  `${view.bars} bars, ${view.axis} axis cells`);
check("tier filter built from the published table", view.opts === 7, `${view.opts} options`);
check("every row carries a tier dot", view.rows > 0 && view.badges === view.rows,
  `${view.badges} dots for ${view.rows} rows`);
check("header is sticky at desktop width", view.sticky === "sticky", view.sticky);
check("trend shows its ghost, not a chart", view.ghost && view.trend !== "ready");
check("the explainer link points at something that exists", view.jump);
// A fresh profile has no dismissal stored and the retire date has not passed, so
// a first-time visitor must see it.
check("change banner shown to a first-time visitor", view.banner);
check("banner links to the changes page", view.blink === "changes.html", view.blink);

await evaluate(`document.getElementById("banner-x").click()`);
check("dismissing the banner hides it",
  await evaluate(`document.getElementById("banner").hidden === true`));
check("dismissal is remembered",
  await evaluate(`localStorage.getItem("seen-changes")`) !== null);
check("page does not scroll sideways", view.hpad);

// ── The member route ─────────────────────────────────────────────────────────
const icpsr = await evaluate(`document.querySelector("#tbody tr[data-icpsr]").dataset.icpsr`);
await evaluate(`location.hash = "#/member/${icpsr}"`);
try {
  await poll(async () => await evaluate(`document.querySelector("#detail-view h2") !== null`),
             "the member page to render", 8000);
  const detail = await evaluate(`JSON.stringify({
    stats: document.querySelectorAll("#detail-view .stat").length,
    main:  document.getElementById("main-view").style.display,
  })`).then(JSON.parse);
  check("member page renders", detail.stats >= 4 && detail.main === "none",
    `${detail.stats} stats`);
} catch (e) {
  check("member page renders", false, e.message);
}

// A bad id must fall through to the list rather than resolve to a prefix.
await evaluate(`location.hash = "#/member/${icpsr}abc"`);
await sleep(200);
check("a malformed member hash falls back to the list",
  await evaluate(`document.getElementById("main-view").style.display !== "none"`));

// ── The changes page ─────────────────────────────────────────────────────────
const before = cdp.events.length;
await cdp.send("Page.navigate", { url: `${origin}/changes.html` });
await sleep(800);
const changes = await evaluate(`JSON.stringify({
  title: document.title,
  heads: document.querySelectorAll("h2").length,
  back:  document.querySelector("a.back")?.getAttribute("href") ?? "",
  wide:  document.body.scrollWidth <= document.documentElement.clientWidth,
})`).then(JSON.parse);
const newLogs = cdp.events.slice(before).filter(e => e.method === "Log.entryAdded")
  .map(e => e.params.entry);
const newFailed = cdp.events.slice(before)
  .filter(e => e.method === "Network.loadingFailed" ||
    (e.method === "Network.responseReceived" && e.params.response.status >= 400));
check("changes page renders", changes.heads >= 5 && /What changed/.test(changes.title),
  `${changes.heads} sections`);
check("changes page links back to the index", changes.back === "./", changes.back);
check("changes page is clean", newLogs.length === 0 && newFailed.length === 0,
  newLogs.map(l => l.text).join(" | "));
check("changes page does not scroll sideways", changes.wide);

cdp.close();
chrome.kill();
await new Promise(r => chrome.on("exit", r));   // it is still writing its profile
server.close();
fs.rmSync(tmp, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
if (guard) clearTimeout(guard);

console.log(failures ? `\n${failures} failure(s)` : "\nall good");
process.exit(failures ? 1 : 0);
