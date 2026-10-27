// AEGIS-SLAM web UI (no build step). Talks to the FastAPI service.
const $ = (id) => document.getElementById(id);
let jobId = null, poll = null, seen = new Set();

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 200));
  return r.json();
}

// ---- version badge ----
fetchJSON("/version").then(v => {
  const arm = v.is_arm_graviton ? " · Graviton (arm64)" : "";
  const inst = v.ec2_instance_type ? ` · ${v.ec2_instance_type}` : "";
  $("version-badge").textContent = `OpenCV ${v.opencv_version}${arm}${inst}`;
  $("version-badge").title = "Click for build info";
}).catch(() => $("version-badge").textContent = "OpenCV version unavailable");

// ---- start panel wiring ----
$("source").addEventListener("change", () => {
  $("upload-row").style.display = $("source").value === "upload" ? "flex" : "none";
});

// ---- judge mode: hide uploads, show presets + token ----
fetchJSON("/api/config").then(c => {
  if (!c.judge_mode) return;
  [...$("source").options].forEach(o => { if (o.value === "upload") o.disabled = true; });
  $("token-row").style.display = "flex";
  $("preset-row").style.display = "flex";
  document.querySelector(".privacy").textContent =
    "Judge mode: read-only demo on the built-in dataset. No uploads, no arbitrary paths.";
}).catch(() => {});

// ---- demo presets: show recorded fault-injected runs (docs/demo) ----
const PRESETS = { "preset-blur": "blur", "preset-cls": "framedrop", "preset-clean": "clean" };
document.querySelectorAll(".preset").forEach(btn => btn.addEventListener("click", async () => {
  const name = PRESETS[btn.id];
  try {
    const r = await fetchJSON(`/preloaded/${name}`);
    jobId = null; stopPoll();
    $("panel-start").style.display = "none";
    $("panel-live").style.display = "block";
    $("panel-result").style.display = "block";
    $("job-id").textContent = `preloaded demo: ${name} (${(r.result.mode || "supervised")}, ${r.result.frames} frames)`;
    $("status").textContent = "done";
    $("progress-fill").style.width = "100%";
    $("progress-txt").textContent = `${r.result.frames}/${r.result.frames} · 100%`;
    seen = new Set();
    renderFeed(r.trace || []);
    if (r.trajectory && (r.trajectory.est || []).length) drawTraj(r.trajectory, r.result.untrusted_segments || []);
    const ate = (r.result.ate && Number.isFinite(r.result.ate.rmse)) ? r.result.ate.rmse.toFixed(3) : "n/a";
    $("advice").textContent = buildAdvice(r.result) + ` ATE RMSE ${ate} m · coverage ${(100*(r.result.coverage_trusted||0)).toFixed(0)}% · interventions ${(r.result.trace_summary||{}).num_decisions}`;
    $("report").textContent = JSON.stringify(r.result, null, 2);
  } catch (e) {
    alert("This demo run is not loaded on the server yet (run scripts/preload_demo.py). " + e.message);
  }
}));

$("start").addEventListener("click", async () => {
  const mode = $("mode").value;
  const scenario = $("scenario") ? $("scenario").value : "clean";
  const fd = new FormData();
  fd.append("mode", mode);
  fd.append("fault", scenario);
  if ($("source").value === "upload") {
    const f = $("file").files[0];
    if (!f) { alert("Choose a video file"); return; }
    fd.append("file", f);
  } else {
    fd.append("dataset", "");
  }
  const tok = $("token") && $("token").value;
  if (tok) fd.append("token", tok);
  $("start").disabled = true; $("start").textContent = "Starting…";
  try {
    const r = await fetch("/jobs", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "failed");
    jobId = j.job_id;
    beginLive(`${mode} · ${scenario}`);
  } catch (e) {
    alert("Could not start job: " + e.message);
  } finally {
    $("start").disabled = false; $("start").textContent = "Run pipeline";
  }
});

function beginLive(label) {
  $("panel-start").style.display = "none";
  $("panel-live").style.display = "block";
  $("job-id").textContent = label ? `${jobId} · ${label}` : jobId;
  seen = new Set();
  poll = setInterval(refresh, 1500);
  refresh();
}

async function refresh() {
  if (!jobId) return;
  try {
    const s = await fetchJSON(`/jobs/${jobId}`);
    $("status").textContent = s.status;
    $("progress-fill").style.width = (s.progress || 0) + "%";
    $("progress-txt").textContent = `${s.frame || 0}/${s.total || "?"} · ${s.progress || 0}%`;
    drawHealth(s.health_summary || {});
    renderPending(s.pending_decisions || []);
    const tr = await fetchJSON(`/jobs/${jobId}/trace`);
    renderFeed(tr.decisions || []);
    if (s.status === "done") { finish(); }
    if (s.status === "error") { stopPoll(); $("advice").textContent = "Run error: " + s.error; }
  } catch (e) { /* transient */ }
}

function stopPoll(){ if (poll) { clearInterval(poll); poll = null; } }

// ---- decision feed ----
function renderFeed(decisions) {
  const feed = $("feed");
  decisions.forEach(d => {
    if (seen.has(d.decision_id)) return;
    seen.add(d.decision_id);
    const verdict = (d.outcome && d.outcome.verdict) || "pending";
    const el = document.createElement("div");
    el.className = "decision";

    const head = document.createElement("div");
    head.className = "d";
    const strong = document.createElement("strong");
    strong.textContent = `@frame ${d.frame}`;
    const pill = document.createElement("span");
    pill.className = "pill " + verdict;
    pill.textContent = verdict;
    head.append(strong, pill);

    const diag = document.createElement("div");
    diag.append("diagnosis: ");
    const b = document.createElement("b");
    b.textContent = d.diagnosis;
    diag.append(b, ` · by: ${d.decided_by}`);

    const act = document.createElement("div");
    const code = document.createElement("code");
    code.textContent = d.action.tool;
    act.append("action: ", code, ` ${JSON.stringify(d.action.args || {})}`);

    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "evidence";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(d.evidence, null, 1);
    det.append(sum, pre);

    el.append(head, diag, act, det);
    feed.prepend(el);
  });
}

// ---- approvals ----
let pendingTimer = 0;
function renderPending(list) {
  const box = $("approval");
  if (!list.length) { box.style.display = "none"; return; }
  const req = list[0];
  box.style.display = "block";
  box.dataset.decision = req.decision_id;
  $("approval-audit").textContent = "";
  $("approval-frame").src = `/jobs/${jobId}/frame/${req.frame}`;
  $("approval-reason").textContent = req.reason;
  $("approval-evidence").textContent = JSON.stringify(req.evidence, null, 1);
  const elapsed = (Date.now()/1000 - req.created_at);
  const left = Math.max(0, Math.round(req.timeout_s - elapsed));
  $("approval-timer").textContent = ` auto-mark-untrusted in ${left}s`;
}
$("approve").addEventListener("click", () => decide(true));
$("skip").addEventListener("click", () => decide(false));
async function decide(approve) {
  const id = $("approval").dataset.decision;
  const body = {
    approve,
    reviewer: ($("reviewer") && $("reviewer").value) || "web-session",
  };
  const tok = $("token") && $("token").value;
  if (tok) body.token = tok;
  const r = await fetch(`/jobs/${jobId}/decisions/${id}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!r.ok) { $("approval-timer").textContent = " " + (await r.text()); return; }
  const rec = await r.json();
  // audit line: who answered, and when — the trace carries the same fields
  const when = rec.resolved_at ? new Date(rec.resolved_at * 1000).toLocaleTimeString() : "—";
  $("approval-audit").textContent =
    `${rec.status} by ${rec.decided_by || "—"} at ${when}`;
  setTimeout(() => { $("approval").style.display = "none"; }, 1500);
  refresh();
}

// ---- health timeline (blur / brightness / inlier ratio) ----
function drawHealth(h) {
  const c = $("health"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#8b949e"; ctx.font = "12px system-ui";
  const rows = [
    ["inlier ratio " + (h.recent_mean_inlier_ratio ?? "—"), h.recent_mean_inlier_ratio, "#2f81f7"],
    ["blur var " + (h.recent_blur_var ?? "—"), h.recent_blur_var/200, "#d29922"],
    ["brightness " + (h.recent_brightness ?? "—"), h.recent_brightness/255, "#3fb950"],
  ];
  rows.forEach((r, i) => {
    const y = 20 + i * 30;
    ctx.fillStyle = "#8b949e"; ctx.fillText(r[0], 8, y);
    const v = Math.max(0, Math.min(1, r[1] || 0));
    ctx.fillStyle = "#0b0f14"; ctx.fillRect(150, y - 10, 260, 12);
    ctx.fillStyle = r[2]; ctx.fillRect(150, y - 10, 260 * v, 12);
  });
  ctx.fillStyle = "#e6edf3";
  ctx.fillText("lost frames: " + (h.lost_frames ?? 0), 8, 110);
}

// ---- results ----
async function finish() {
  stopPoll();
  const res = await fetchJSON(`/jobs/${jobId}/result`);
  $("panel-result").style.display = "block";
  const r = res.result || {};
  const ate = (r.ate && Number.isFinite(r.ate.rmse)) ? r.ate.rmse.toFixed(3) : "n/a";
  const advice = buildAdvice(r);
  $("advice").textContent = advice;
  $("report").textContent = JSON.stringify(r, null, 2);
  $("trace-dl").href = `/jobs/${jobId}/trace`;
  if (res.trajectory) drawTraj(res.trajectory, r.untrusted_segments || []);
}

function buildAdvice(r) {
  const parts = [];
  const ts = r.trace_summary || {};
  const dg = ts.by_diagnosis || {};
  if (dg.motion_blur) parts.push("Motion blur detected — slow your pan/tilt or raise shutter speed.");
  if (dg.low_light) parts.push("Low light detected — add lighting or avoid dark segments.");
  if (dg.textureless) parts.push("Textureless regions detected — capture areas with more surface detail.");
  if (dg.tracking_lost || dg.tracking_lost_unrecoverable) parts.push("Tracking was lost — re-record the flagged segment with steadier motion and overlap.");
  if (r.untrusted_segments && r.untrusted_segments.length) parts.push(`Untrusted segments: ${JSON.stringify(r.untrusted_segments)} — do not use these poses.`);
  if (!parts.length) parts.push("No faults detected — this capture looks reliable.");
  return parts.join(" ");
}

function drawTraj(traj, untrusted) {
  const c = $("traj"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const est = traj.est || [], gt = traj.gt || [];
  const all = est.concat(gt);
  if (!all.length) return;
  let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
  all.forEach(p=>{const x=p[0],y=p[2];minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);});
  const pad=20, sc=Math.min((c.width-2*pad)/Math.max(1e-3,maxX-minX),(c.height-2*pad)/Math.max(1e-3,maxY-minY));
  const T=(p)=>[pad+(p[0]-minX)*sc, c.height-pad-(p[2]-minY)*sc];
  if (gt.length){ ctx.strokeStyle="#8b949e";ctx.lineWidth=1.5;ctx.beginPath();gt.forEach((p,i)=>{const[x,y]=T(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke(); }
  ctx.strokeStyle="#2f81f7";ctx.lineWidth=2;ctx.beginPath();
  est.forEach((p,i)=>{const[x,y]=T(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  est.forEach(p=>{const[x,y]=T(p);ctx.fillStyle="#3fb950";ctx.fillRect(x-1.5,y-1.5,3,3);});
}
