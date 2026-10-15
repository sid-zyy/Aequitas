"use strict";

// ---------- helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const CATEGORIES = {
  person_down: "Person down",
  fighting: "Fighting",
  tailgating: "Tailgating",
  crowd_incident: "Crowd incident",
  suspicious_activity: "Suspicious activity",
};
const SUBTYPES = { loitering: "Loitering", abandoned_object: "Abandoned object", overcrowding: "Overcrowding", dispersal: "Dispersal" };
const STATUS_LABEL = {
  confirmed: "Confirmed", disputed: "Disputed", rejected: "Rejected", unverified: "Unverified",
  queued: "Queued", processing: "Processing", verifying: "AI verifying", done: "Complete", failed: "Failed", cancelled: "Cancelled",
};
const SUGGESTIONS = [
  "Show all tailgating events from camera 2 today",
  "How many people fell this week?",
  "Any fights in the last 2 hours?",
  "Was a bag left unattended near the entrance?",
  "Show disputed and unverified events from today",
];

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function fmtTime(ms, withDate = true) {
  const d = new Date(ms);
  const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  if (!withDate) return t;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const day = new Date(d); day.setHours(0, 0, 0, 0);
  const diff = (today - day) / 86400000;
  const label = diff === 0 ? "Today" : diff === 1 ? "Yesterday" : d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
  return `${label}, ${t}`;
}
const fmtDur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`);
const label = (ev) => SUBTYPES[ev.subtype] || CATEGORIES[ev.category] || ev.category;
const catChip = (ev) => `<span class="cat" style="--cc: var(--c-${esc(ev.category)})"><i></i>${esc(label(ev))}</span>`;
const badge = (status) => `<span class="badge b-${esc(status)}">${esc(STATUS_LABEL[status] || status)}</span>`;

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3200);
}

const ICON_EMPTY = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>';
const empty = (text) => `<div class="empty">${ICON_EMPTY}<div>${esc(text)}</div></div>`;

// ---------- state ----------
const state = { status: null, cameras: [], mode: "triggered", file: null, jobsTimer: null, eventsCache: new Map() };

// ---------- routing ----------
const pages = ["overview", "analyze", "ask", "events"];
function route() {
  const name = (location.hash.replace("#/", "") || "overview").split("?")[0];
  const page = pages.includes(name) ? name : "overview";
  if ($("#modal").open) $("#modal").close();
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.dataset.page === page));
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === page));
  ({ overview: loadOverview, analyze: loadAnalyze, ask: () => $("#question").focus(), events: loadEvents })[page]();
}
window.addEventListener("hashchange", route);

// ---------- system status ----------
async function loadStatus() {
  try {
    const s = await api("/api/status");
    state.status = s;
    $("#sys-vlm-dot").className = "dot " + (s.vlm.online ? "on" : "off");
    $("#sys-vlm").textContent = `${s.vlm.model} ${s.vlm.online ? "online" : "offline"}`;
    $("#sys-device").textContent = `${s.device.toUpperCase()} · ${s.sample_fps} fps sampling`;
    const sel = $("#model");
    sel.innerHTML = s.vision_models.map((m) => `<option value="${esc(m)}" ${m === s.vlm.model ? "selected" : ""}>${esc(m)}</option>`).join("");
  } catch {
    $("#sys-vlm-dot").className = "dot off";
    $("#sys-vlm").textContent = "Server unreachable";
  }
}

async function loadCameras() {
  state.cameras = await api("/api/cameras");
  const opts = state.cameras.map((c) => `<option value="${esc(c.camera_id)}">${esc(c.name)} (${esc(c.camera_id)})</option>`).join("");
  const cur = $("#camera").value;
  $("#camera").innerHTML = opts + `<option value="__new">+ New camera…</option>`;
  if (cur) $("#camera").value = cur;
  if (!state.cameras.length) $("#camera").value = "__new";
  $("#new-camera").hidden = $("#camera").value !== "__new";
  const fc = $("#f-camera").value;
  $("#f-camera").innerHTML = `<option value="">All cameras</option>` + state.cameras.map((c) => `<option value="${esc(c.camera_id)}">${esc(c.name)}</option>`).join("");
  $("#f-camera").value = fc;
}

// ---------- overview ----------
const KPI_ICONS = {
  today: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18M8 2v4M16 2v4"/></svg>',
  total: '<svg viewBox="0 0 24 24"><path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-4z"/><path d="M9 12l2 2 4-4"/></svg>',
  videos: '<svg viewBox="0 0 24 24"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="M16 10l6-3v10l-6-3"/></svg>',
  cams: '<svg viewBox="0 0 24 24"><path d="M3 7h3l2-3h8l2 3h3v13H3z"/><circle cx="12" cy="13" r="4"/></svg>',
};

async function loadOverview() {
  const [stats, recent] = await Promise.all([api("/api/stats"), api("/api/events?status=confirmed&limit=6")]);
  const total = stats.by_status.confirmed || 0;
  const kpis = [
    ["Confirmed today", stats.confirmed_today, KPI_ICONS.today],
    ["Confirmed incidents", total, KPI_ICONS.total],
    ["Videos analyzed", stats.videos_analyzed, KPI_ICONS.videos],
    ["Cameras", stats.cameras, KPI_ICONS.cams],
  ];
  $("#kpis").innerHTML = kpis.map(([l, v, i]) => `<div class="card kpi"><div class="kpi-label">${i}${esc(l)}</div><div class="kpi-value">${Number(v).toLocaleString()}</div></div>`).join("");

  const max = Math.max(1, ...Object.values(stats.by_category));
  $("#cat-bars").innerHTML = Object.entries(CATEGORIES).map(([k, name]) => {
    const n = stats.by_category[k] || 0;
    return `<div class="bar-row"><span class="cat" style="--cc: var(--c-${k})"><i></i>${esc(name)}</span>
      <div class="bar-track"><div class="bar-fill" style="--cc: var(--c-${k}); width:${(n / max) * 100}%"></div></div>
      <span class="bar-n">${n}</span></div>`;
  }).join("");

  recent.events.forEach((e) => state.eventsCache.set(e.id, e));
  $("#recent").innerHTML = recent.events.length
    ? recent.events.map((e) => `<div class="list-item" data-ev="${e.id}" tabindex="0" role="button">
        <img class="thumb" src="${esc(e.keyframe_url)}" alt="" loading="lazy">
        <div style="min-width:0"><div class="li-title">${catChip(e)}</div>
          <div class="li-sub">${esc(e.camera_name)} · ${esc(fmtTime(e.start_ts))}</div></div>
        <span class="conf">${Math.round(e.confidence * 100)}%</span></div>`).join("")
    : empty("No confirmed incidents yet. Analyze a video to get started.");
}

// ---------- analyze ----------
function setFile(f) {
  state.file = f || null;
  const drop = $("#drop");
  drop.classList.toggle("has-file", !!f);
  $("#drop-title").innerHTML = f ? esc(f.name) : "Drop a video here or <u>browse</u>";
  $("#drop-sub").textContent = f ? `${(f.size / 1048576).toFixed(1)} MB · ready to analyze` : "MP4, MOV, AVI, MKV or WebM · up to 4 GB";
  validateForm();
}
function validateForm() {
  const camOk = $("#camera").value !== "__new" || $("#new-camera").value.trim().length > 0;
  $("#submit").disabled = !(state.file && camOk);
}
function formError(msg) {
  const el = $("#form-error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

function initUploadForm() {
  const drop = $("#drop"), input = $("#file");
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
  $("#camera").addEventListener("change", () => {
    const isNew = $("#camera").value === "__new";
    $("#new-camera").hidden = !isNew;
    if (isNew) $("#new-camera").focus();
    validateForm();
  });
  $("#new-camera").addEventListener("input", validateForm);
  $("#mode").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-value]");
    if (!b) return;
    state.mode = b.dataset.value;
    $("#mode").querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  });

  $("#upload-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!state.file) return;
    formError("");
    const fd = new FormData();
    fd.append("file", state.file);
    if ($("#camera").value === "__new") fd.append("new_camera_name", $("#new-camera").value.trim());
    else fd.append("camera_id", $("#camera").value);
    if ($("#recorded").value) fd.append("recorded_at", $("#recorded").value);
    fd.append("mode", state.mode);
    fd.append("vlm_model", $("#model").value || "");

    const bar = $("#upload-bar"), btn = $("#submit");
    bar.hidden = false; btn.disabled = true;
    btn.lastChild.textContent = " Uploading…";
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) bar.firstElementChild.style.width = `${(ev.loaded / ev.total) * 100}%`; };
    xhr.onload = async () => {
      bar.hidden = true; bar.firstElementChild.style.width = "0";
      btn.lastChild.textContent = " Start analysis";
      if (xhr.status >= 200 && xhr.status < 300) {
        toast("Upload complete. Analysis started.");
        setFile(null); $("#file").value = ""; $("#new-camera").value = "";
        await loadCameras();
        const job = JSON.parse(xhr.responseText);
        $("#camera").value = job.camera_id; $("#new-camera").hidden = true;
        loadJobs();
      } else {
        let msg = "Upload failed.";
        try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
        formError(typeof msg === "string" ? msg : "Upload failed.");
      }
      validateForm();
    };
    xhr.onerror = () => { bar.hidden = true; btn.lastChild.textContent = " Start analysis"; formError("Network error during upload."); validateForm(); };
    xhr.send(fd);
  });
}

function jobCard(j) {
  const active = ["queued", "processing", "verifying"].includes(j.status);
  const pct = Math.round(j.progress * 100);
  let detail;
  if (j.status === "queued") detail = "Waiting for the previous job to finish";
  else if (j.status === "processing") detail = `Scanning frames ${j.frames_done.toLocaleString()} / ${j.frames_total.toLocaleString()} · ${pct}%${j.vlm_backlog ? ` · ${j.vlm_backlog} awaiting AI review` : ""}`;
  else if (j.status === "verifying") {
    const unit = j.mode === "periodic" ? "window" : "candidate";
    detail = j.vlm_backlog ? `AI reviewing ${j.vlm_backlog} remaining ${unit}${j.vlm_backlog === 1 ? "" : "s"}` : "Finishing AI review";
  }
  else if (j.status === "done") detail = `${j.events_found} event${j.events_found === 1 ? "" : "s"} recorded`;
  else detail = STATUS_LABEL[j.status] || j.status;
  const modeName = j.mode === "periodic" ? "VLM review" : "Tracking-triggered";
  const indeterminate = j.status === "queued" || j.status === "verifying";
  return `<div class="job ${esc(j.status)}">
    <div class="job-top">
      <div style="min-width:0"><div class="job-name">${esc(j.filename)}</div>
        <div class="job-meta">${esc(j.camera_name || j.camera_id)} · ${esc(fmtDur(j.duration_s))} · ${esc(modeName)} · ${esc(j.vlm_model || "")}</div></div>
      ${badge(j.status)}
    </div>
    ${active ? `<div class="progress ${indeterminate ? "indeterminate" : ""}"><div style="width:${pct}%"></div></div>` : ""}
    <div class="job-foot" style="${active ? "" : "margin-top:10px"}">
      <span>${esc(detail)}</span>
      <span class="actions">
        ${active ? `<button class="btn btn-sm" data-cancel="${esc(j.id)}">Cancel</button>` : ""}
        ${j.status === "done" && j.events_found ? `<button class="btn btn-sm" data-view-cam="${esc(j.camera_id)}">View events</button>` : ""}
      </span>
    </div>
    ${j.error ? `<div class="job-error">${esc(j.error)}</div>` : ""}
  </div>`;
}

async function loadJobs() {
  let jobs;
  try { jobs = await api("/api/uploads"); } catch { return; }
  const active = jobs.filter((j) => ["queued", "processing", "verifying"].includes(j.status)).length;
  const nb = $("#nav-jobs");
  nb.hidden = !active; nb.textContent = active;
  $("#jobs-count").textContent = jobs.length ? `${jobs.length} total` : "";
  $("#jobs").innerHTML = jobs.length ? jobs.map(jobCard).join("") : empty("No videos analyzed yet.");
  clearTimeout(state.jobsTimer);
  if (active) state.jobsTimer = setTimeout(loadJobs, 1500);
  else if (loadJobs._wasActive) { loadOverviewIfVisible(); }
  loadJobs._wasActive = !!active;
}
function loadOverviewIfVisible() { if ($('.page[data-page="overview"]').classList.contains("active")) loadOverview(); }

async function loadAnalyze() {
  await Promise.all([loadCameras(), loadJobs()]);
  validateForm();
}

// ---------- ask ----------
function addMsg(role, html) {
  $("#chat-empty").hidden = true;
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = role === "user"
    ? `<div class="bubble">${html}</div>`
    : `<div class="avatar"><svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg></div><div class="bot-body">${html}</div>`;
  $("#thread").appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

function linkRefs(text) {
  return esc(text).replace(/\[#(\d+)\]/g, (_, id) => `<button class="ref" data-ev="${id}">#${id}</button>`);
}

function eventRow(e) {
  return `<button class="ev-row" data-ev="${e.id}">
    <img src="${esc(e.keyframe_url)}" alt="" loading="lazy">
    <div style="min-width:0">
      <div class="ev-row-top"><span class="ref">#${e.id}</span>${catChip(e)}${e.status !== "confirmed" ? badge(e.status) : ""}</div>
      <div class="ev-row-sum">${esc(e.summary || "No AI summary for this event.")}</div>
      <div class="ev-row-meta">${esc(e.camera_name)} · ${esc(fmtTime(e.start_ts))} · ${Math.round(e.confidence * 100)}% confidence</div>
    </div></button>`;
}

function renderAnswer(r) {
  r.events.forEach((e) => state.eventsCache.set(e.id, e));
  const lines = r.text.split("\n");
  const head = r.intent === "count"
    ? lines.find((l) => /^\d+ /.test(l)) || lines[lines.length - 1]
    : lines.find((l) => l.startsWith("Found ") || l.startsWith("No ")) || "";
  let html = r.warnings.map((w) => `<div class="answer-warn">${esc(w)}</div>`).join("");
  if (r.summary) html += `<div class="answer-summary">${linkRefs(r.summary)}</div>`;
  html += `<div class="answer-head">${esc(head.replace(/:$/, ""))}</div>`;
  if (r.events.length) html += `<div class="ev-list">${r.events.map(eventRow).join("")}</div>`;
  const how = r.parsed_with === "llm" ? "Understood by the local LLM" : "Understood by keyword matching";
  html += `<div class="answer-meta">${how} · ${(r.elapsed_ms / 1000).toFixed(1)} s · answers come only from recorded events</div>`;
  return html;
}

async function ask(q) {
  q = q.trim();
  if (!q) return;
  $("#question").value = ""; autosize();
  addMsg("user", esc(q));
  const pending = addMsg("bot", `<div class="typing"><span></span><span></span><span></span></div>`);
  $("#send").disabled = true;
  try {
    const r = await api("/api/ask", { method: "POST", body: JSON.stringify({ question: q }) });
    pending.querySelector(".bot-body").innerHTML = renderAnswer(r);
  } catch (err) {
    pending.querySelector(".bot-body").innerHTML = `<div class="answer-warn">Sorry, that failed: ${esc(err.message)}</div>`;
  } finally {
    $("#send").disabled = false;
    pending.scrollIntoView({ behavior: "smooth", block: "end" });
    $("#question").focus();
  }
}
function autosize() { const t = $("#question"); t.style.height = "auto"; t.style.height = Math.min(160, t.scrollHeight) + "px"; }

function initChat() {
  $("#suggestions").innerHTML = SUGGESTIONS.map((s) => `<button class="chip" type="button">${esc(s)}</button>`).join("");
  $("#suggestions").addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) ask(c.textContent); });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask($("#question").value); });
  $("#question").addEventListener("input", autosize);
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("#question").value); } });
  $("#chat-clear").addEventListener("click", () => {
    $("#thread").querySelectorAll(".msg").forEach((m) => m.remove());
    $("#chat-empty").hidden = false;
  });
}

// ---------- events ----------
let eventsSeq = 0;
async function loadEvents() {
  const seq = ++eventsSeq;
  await loadCameras();
  const p = new URLSearchParams({ status: $("#f-status").value, limit: "60" });
  if ($("#f-camera").value) p.set("camera", $("#f-camera").value);
  if ($("#f-category").value) p.set("category", $("#f-category").value);
  if ($("#f-q").value.trim()) p.set("q", $("#f-q").value.trim());
  const grid = $("#event-grid");
  if (!grid.children.length) grid.innerHTML = Array(6).fill('<div class="ev"><div class="skeleton" style="aspect-ratio:16/10"></div><div class="ev-body"><div class="skeleton" style="height:14px;width:60%"></div><div class="skeleton" style="height:32px"></div></div></div>').join("");
  const r = await api(`/api/events?${p}`);
  if (seq !== eventsSeq) return;
  r.events.forEach((e) => state.eventsCache.set(e.id, e));
  $("#events-sub").textContent = `${r.total.toLocaleString()} event${r.total === 1 ? "" : "s"}${r.total > r.events.length ? ` · showing the latest ${r.events.length}` : ""}`;
  grid.innerHTML = r.events.length ? r.events.map((e) => `<button class="ev" data-ev="${e.id}">
      <img src="${esc(e.keyframe_url)}" alt="" loading="lazy">
      <div class="ev-body">
        <div class="ev-top">${catChip(e)}${badge(e.status)}</div>
        <div class="ev-sum">${esc(e.summary || "No AI summary for this event.")}</div>
        <div class="ev-meta"><span>${esc(e.camera_name)} · ${esc(fmtTime(e.start_ts))}</span><span class="conf">#${e.id}</span></div>
      </div></button>`).join("") : `<div style="grid-column:1/-1">${empty("No events match these filters.")}</div>`;
}

function initEvents() {
  $("#f-category").innerHTML = `<option value="">All categories</option>` + Object.entries(CATEGORIES).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  ["#f-camera", "#f-category", "#f-status"].forEach((s) => $(s).addEventListener("change", loadEvents));
  let t; $("#f-q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(loadEvents, 300); });
}

// ---------- event modal ----------
async function openEvent(id) {
  let e = state.eventsCache.get(id);
  if (!e) { try { e = await api(`/api/events/${id}`); } catch (err) { toast(err.message); return; } }
  const m = $("#modal");
  $("#m-title").innerHTML = `${catChip(e)} ${badge(e.status)}`;
  $("#m-sub").textContent = `Event #${e.id} · ${e.camera_name} · ${fmtTime(e.start_ts)}`;
  $("#m-summary").textContent = e.summary || "No AI summary for this event.";
  const rows = [
    ["Camera", `${e.camera_name} (${e.camera_id})`],
    ["Started", fmtTime(e.start_ts)],
    ["Duration", fmtDur(Math.max(1, (e.end_ts - e.start_ts) / 1000))],
    ["Category", label(e)],
    ["Confidence", `${Math.round(e.confidence * 100)}%`],
    ["AI label", e.vlm_category ? (CATEGORIES[e.vlm_category] || e.vlm_category) : "—"],
    ["Rule strength", e.trigger_score ? `${Math.round(e.trigger_score * 100)}%` : "—"],
    ["Zone", e.zone || "—"],
    ["Tracks", e.track_ids && e.track_ids.length ? e.track_ids.join(", ") : "—"],
    ["Models", [e.models.vlm, e.models.detector].filter(Boolean).join(" · ") || "—"],
  ];
  $("#m-meta").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  const img = $("#m-img"), vid = $("#m-video"), tabs = $("#m-tabs");
  img.src = e.keyframe_url;
  vid.pause(); vid.removeAttribute("src"); vid.load();
  tabs.hidden = !e.video;
  showTab("frames");
  if (e.video) {
    vid.dataset.offset = e.video.offset_s;
    vid.dataset.src = e.video.url;
  }
  m.showModal();
}
function showTab(name) {
  document.querySelectorAll("#m-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  const vid = $("#m-video");
  $("#m-img").hidden = name !== "frames";
  vid.hidden = name !== "video";
  if (name === "video" && vid.dataset.src) {
    if (!vid.getAttribute("src")) {
      vid.src = vid.dataset.src;
      vid.addEventListener("loadedmetadata", () => { vid.currentTime = Math.max(0, Number(vid.dataset.offset) - 1); }, { once: true });
    }
    vid.play().catch(() => {});
  } else vid.pause();
}

document.addEventListener("click", (e) => {
  const ev = e.target.closest("[data-ev]");
  if (ev) { openEvent(Number(ev.dataset.ev)); return; }
  const cancel = e.target.closest("[data-cancel]");
  if (cancel) { api(`/api/uploads/${cancel.dataset.cancel}/cancel`, { method: "POST" }).then(loadJobs).catch((err) => toast(err.message)); return; }
  const view = e.target.closest("[data-view-cam]");
  if (view) { $("#f-camera").dataset.pending = view.dataset.viewCam; location.hash = "#/events"; return; }
  const tab = e.target.closest("#m-tabs .tab");
  if (tab) showTab(tab.dataset.tab);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.matches(".list-item[data-ev]")) openEvent(Number(e.target.dataset.ev));
});
$("#m-close").addEventListener("click", () => $("#modal").close());
$("#modal").addEventListener("click", (e) => { if (e.target === $("#modal")) $("#modal").close(); });
$("#modal").addEventListener("close", () => $("#m-video").pause());

// "View events" from a job pre-selects that camera.
const _loadEvents = loadEvents;
loadEvents = async function () {
  const pend = $("#f-camera").dataset.pending;
  if (pend) {
    await loadCameras();
    $("#f-camera").value = pend;
    $("#f-status").value = "all";
    delete $("#f-camera").dataset.pending;
  }
  return _loadEvents();
};

// ---------- boot ----------
initUploadForm();
initChat();
initEvents();
loadStatus();
loadJobs();
route();
setInterval(loadStatus, 30000);
