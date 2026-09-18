(function () {
  "use strict";

  var state = {
    view: "search",
    results: [],
    total: 0,
    offset: 0,
    lastParams: {},
    selected: new Set(),
    people: [],
    progressTimer: null,
    folder: { root: null, name: null, indexed: false },
    recent: [],
    categories: { fixed: {}, discovered: {} },
    classifyTimer: null,
    findPath: null,
    exportDest: null,
    savedPeople: [],
    peopleUnticked: new Set(),
    catTicked: new Set(),       // tile keys ticked for "Export ticked categories": a fixed name, or "discovered:" + name
    catSeen: new Set(),         // tile keys already given their default tick (all but "unclassified")
  };

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  // The status line holds the text span plus the "name this person" form, so only the span is written.
  var statusEl = $("#status-text");
  var statusTimer = null;
  function setStatus(msg, hold) {
    statusEl.textContent = msg || "";
    if (statusTimer) { clearTimeout(statusTimer); statusTimer = null; }
    if (msg && !hold) {
      statusTimer = setTimeout(function () { statusEl.textContent = ""; }, 6000);
    }
  }

  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          var err = new Error((body && body.detail) || (r.status + " " + r.statusText));
          err.status = r.status;
          throw err;
        });
      }
      return r.json();
    });
  }

  // ---------- tabs ----------
  function showView(name) {
    state.view = name;
    $$("header nav button").forEach(function (b) {
      b.classList.toggle("on", b.dataset.view === name);
    });
    var hasFolder = !!(state.folder && state.folder.root);
    $$("main > section").forEach(function (s) {
      if (s.id === "view-nofolder") { s.hidden = hasFolder; return; }
      s.hidden = !hasFolder || s.id !== "view-" + name;
    });
    if (!hasFolder) return;
    if (name === "people" && state.people.length === 0) loadPeople();
    if (name === "people") loadReferences();
    if (name === "categories") loadCategories();
  }
  $$("header nav button").forEach(function (b) {
    b.addEventListener("click", function () { showView(b.dataset.view); });
  });

  // ---------- stats ----------
  var lastErrorCount = null;
  function loadStats() {
    return api("/api/stats").then(function (s) {
      var bits = [s.photos + " photos", s.faces + " faces", s.people + " people"];
      if (s.last_index) bits.push("indexed " + s.last_index);
      if (s.indexing) bits.push("indexing…");
      if (s.errors) bits.push(s.errors + " failed");
      $("#stats").textContent = bits.join("  ·  ");
      if (s.errors !== lastErrorCount) {
        lastErrorCount = s.errors;
        loadErrors();
      }
      return s;
    });
  }

  function loadErrors() {
    return api("/api/errors").then(function (data) {
      var list = (data && data.errors) || [];
      var box = $("#index-errors");
      box.hidden = list.length === 0;
      $("#index-errors-title").textContent = list.length + " file(s) could not be read. They are skipped; tick retry and Index again once fixed.";
      $("#index-errors-list").textContent = list.slice(0, 200).map(function (e) { return e.rel; }).join("\n") + (list.length > 200 ? "\n… " + (list.length - 200) + " more" : "");
    }).catch(function () { /* non-fatal */ });
  }

  // ---------- folder ----------
  var folderNameEl = $("#folder-name");
  var recentSelect = $("#recent-folders");

  function applyFolderInfo(info) {
    state.folder = info || { root: null, name: null, indexed: false };
    folderNameEl.textContent = state.folder.name || "no folder open";
    folderNameEl.title = state.folder.root || "";
  }

  function loadFolder() {
    return api("/api/folder").then(function (info) {
      applyFolderInfo(info);
      return info;
    });
  }

  function loadRecent() {
    return api("/api/folder/recent").then(function (data) {
      state.recent = (data && data.recent) || [];
      renderRecent();
    }).catch(function () { /* non-fatal */ });
  }

  function renderRecent() {
    recentSelect.innerHTML = '<option value="">recent&hellip;</option>';
    state.recent.forEach(function (r) {
      var opt = document.createElement("option");
      opt.value = r.path;
      opt.textContent = r.name || r.path;
      recentSelect.appendChild(opt);
    });
    recentSelect.value = "";
  }

  // Reached after any folder switch: reset per-folder UI state, then either land
  // on the Index tab (fresh folder, nothing indexed yet) or refresh the current view.
  function settleFolder(info) {
    state.people = [];
    state.results = [];
    state.selected = new Set();
    state.categories = { fixed: {}, discovered: {} };
    state.catTicked = new Set(); state.catSeen = new Set();
    state.findPath = null; syncSaveForm();            // a reference from the previous shoot must not be saved into this one
    state.savedPeople = []; state.peopleUnticked = new Set(); renderSavedPeople();
    renderGrid();
    peopleEl.innerHTML = "";
    personSelect.innerHTML = '<option value="">anyone</option>';
    progressEl.textContent = "";
    $("#index-errors").hidden = true;
    lastErrorCount = null;
    $("#category-filter").value = ""; $("#cluster-filter").value = "";
    showCategoryChip(null);
    loadRecent();
    if (!info.root) {
      showView(state.view);
      return;
    }
    if (!info.indexed) {
      showView("index");
      var btn = $("#start-index");
      if (btn) btn.focus();
      loadStats().then(function (s) { if (s.indexing) pollProgress(); });   // first index of a fresh folder, page reloaded mid-run
      return;
    }
    loadStats().then(function (s) { loadErrors(); if (s.indexing) pollProgress(); });
    loadPeople();
    runSearch();
    showView(state.view === "index" ? "search" : state.view);
  }

  function openFolderPicker() {
    setStatus("waiting for the folder picker…", true);
    return fetch("/api/folder/choose", { method: "POST" }).then(function (r) {
      if (r.status === 204) { setStatus("folder pick cancelled"); return null; }
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          throw new Error((body && body.detail) || (r.status + " " + r.statusText));
        });
      }
      return r.json();
    }).then(function (info) {
      if (!info) return;
      applyFolderInfo(info);
      setStatus("opened " + (info.name || info.root));
      settleFolder(info);
    }).catch(function (err) {
      setStatus("could not open folder: " + err.message);
    });
  }

  function switchFolder(path) {
    setStatus("opening " + path + "…", true);
    return api("/api/folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path }),
    }).then(function (info) {
      applyFolderInfo(info);
      setStatus("opened " + (info.name || info.root));
      settleFolder(info);
    }).catch(function (err) {
      setStatus("could not open folder: " + err.message);
    });
  }

  // export destination (another disk)
  var exportDestEl = $("#export-dest");
  var exportDestReset = $("#export-dest-reset");

  function renderExportDest() {
    var d = state.exportDest;
    if (!d) { exportDestEl.textContent = ""; exportDestEl.title = ""; exportDestReset.hidden = true; return; }
    exportDestEl.textContent = "exports go to " + d.path + (d.mounted ? " (" + d.free_gb + " GB free)" : " (not mounted)");
    exportDestEl.title = d.path;
    exportDestReset.hidden = !!d.default;
  }

  function loadExportDest() {
    return api("/api/export/destination").then(function (d) {
      state.exportDest = d;
      renderExportDest();
      return d;
    }).catch(function () { /* non-fatal */ });
  }

  $("#export-dest-change").addEventListener("click", function () {
    setStatus("waiting for the folder picker…", true);
    fetch("/api/export/destination/choose", { method: "POST" }).then(function (r) {
      if (r.status === 204) { setStatus("destination unchanged"); return null; }
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          throw new Error((body && body.detail) || (r.status + " " + r.statusText));
        });
      }
      return r.json();
    }).then(function (d) {
      if (!d) return;
      state.exportDest = d;
      renderExportDest();
      setStatus("exports now go to " + d.path);
    }).catch(function (err) {
      setStatus("could not change the destination: " + err.message);
    });
  });

  exportDestReset.addEventListener("click", function () {
    api("/api/export/destination", { method: "DELETE" }).then(function (d) {
      state.exportDest = d;
      renderExportDest();
      setStatus("exports go back to " + d.path);
    }).catch(function (err) {
      setStatus("could not reset the destination: " + err.message);
    });
  });

  $("#open-folder").addEventListener("click", openFolderPicker);
  $("#open-folder-main").addEventListener("click", openFolderPicker);
  recentSelect.addEventListener("change", function () {
    var path = recentSelect.value;
    if (path) switchFolder(path);
  });

  // ---------- search ----------
  var form = $("#q");
  var sharpInput = form.querySelector('[name="sharp"]');
  var sharpOut = $("#sharp-out");
  sharpInput.addEventListener("input", function () { sharpOut.textContent = sharpInput.value; });
  sharpInput.addEventListener("change", function () { runSearch(); });

  function currentFilters() {
    var fd = new FormData(form);
    var params = {};
    var q = (fd.get("q") || "").trim();
    if (q) params.q = q;
    var sharp = fd.get("sharp");
    if (sharp && Number(sharp) > 0) params.sharp = sharp;
    var faces = fd.get("faces");
    if (faces) params.faces = faces;
    var person = fd.get("person");
    if (person) params.person = person;
    var category = fd.get("category");
    if (category) params.category = category;
    var cluster = fd.get("cluster");
    if (cluster) params.cluster = cluster;
    return params;
  }

  var PAGE = 200;
  function runSearch(extra, append) {
    var params = currentFilters();
    Object.assign(params, extra || {});
    if (!append) { state.offset = 0; state.results = []; }
    params.limit = PAGE; params.offset = state.offset;
    state.lastParams = params;
    var qs = new URLSearchParams(params).toString();
    return api("/api/search?" + qs).then(function (data) {
      state.results = append ? state.results.concat(data.results || []) : (data.results || []);
      state.total = data.total || 0;
      state.offset = state.results.length;
      renderGrid();
    }).catch(function (err) {
      if (err.status === 404) {
        state.results = []; state.total = 0;
        renderGrid();
        setStatus("that photo has no embedding to compare against");
      } else {
        setStatus("search failed: " + err.message);
      }
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    runSearch();
  });
  form.querySelectorAll("select").forEach(function (sel) {
    sel.addEventListener("change", function () { runSearch(); });
  });
  $("#show-more").addEventListener("click", function () { runSearch(state.lastParams, true); });
  $("#select-matching").addEventListener("click", function () {
    var p = Object.assign({}, state.lastParams); delete p.limit; delete p.offset;
    api("/api/search/ids?" + new URLSearchParams(p).toString()).then(function (data) {
      (data.ids || []).forEach(function (id) { state.selected.add(id); });
      $$(".card", gridEl).forEach(function (c) { if (state.selected.has(Number(c.dataset.id))) c.classList.add("selected"); });
      updateSelbar();
      setStatus("selected all " + data.total + " matching photo(s)");
    }).catch(function (err) { setStatus("could not select: " + err.message); });
  });

  // n_faces is null when the photo was indexed with faces off: not "0", just unknown
  function facesLabel(n) { return n == null ? "?" : String(n); }

  var gridEl = $("#grid");
  function renderGrid() {
    gridEl.innerHTML = "";
    var divided = false;
    state.results.forEach(function (r) {
      // Results arrive sure first, then the "less sure" band by confidence: one divider before the first
      // unsure one. The whole list is rebuilt from state.results, so "Show more" keeps a single divider.
      var unsure = r.sure === false;
      if (unsure && !divided) {
        divided = true;
        var div = document.createElement("div");
        div.className = "grid-divider mono";
        div.textContent = "less sure, sorted by confidence";
        gridEl.appendChild(div);
      }
      var card = document.createElement("div");
      card.className = "card";
      if (unsure) card.classList.add("unsure");
      if (state.selected.has(r.id)) card.classList.add("selected");
      card.dataset.id = r.id;

      var img = document.createElement("img");
      img.loading = "lazy";
      img.alt = r.rel;
      img.src = "/api/thumb/" + r.qhash + "?size=grid";
      card.appendChild(img);

      var tag = document.createElement("div");
      tag.className = "tag mono";
      var sharpPct = r.sharp_pct != null ? Math.round(r.sharp_pct) : 0;
      tag.textContent = sharpPct + "%  " + facesLabel(r.n_faces) + "f";
      if (unsure) tag.textContent += "  " + Math.round((r.confidence || 0) * 100) + "% sure";
      card.appendChild(tag);

      card.addEventListener("click", function () {
        toggleSelect(r.id, card);
      });
      card.addEventListener("dblclick", function () {
        openLightbox(r);
      });
      gridEl.appendChild(card);
    });
    var more = $("#more-row");
    more.hidden = state.results.length === 0;
    // A text or image query ranks the whole shoot, so "total" is the shoot size and
    // "select all matching" would select everything: only filter-only searches get it.
    var ranked = !!(state.lastParams.q || state.lastParams.image_id);
    $("#shown-count").textContent = ranked ? state.results.length + " shown" : state.results.length + " of " + state.total + " shown";
    $("#show-more").hidden = state.results.length >= state.total;
    $("#select-matching").hidden = ranked;
    updateSelbar();
  }

  function toggleSelect(id, card) {
    if (state.selected.has(id)) {
      state.selected.delete(id);
      card.classList.remove("selected");
    } else {
      state.selected.add(id);
      card.classList.add("selected");
    }
    updateSelbar();
  }

  var selbar = $("#selbar");
  var selcount = $("#selcount");
  var selectAllTop = $("#selectall-top");
  function updateSelbar() {
    var n = state.selected.size;
    var shown = state.results.length;
    selbar.hidden = shown === 0 && n === 0;
    selectAllTop.hidden = shown === 0;
    selcount.textContent = n + " selected" + (shown ? " of " + shown + " shown" : "");
  }

  function selectAllShown() {
    state.results.forEach(function (r) { state.selected.add(r.id); });
    $$(".card", gridEl).forEach(function (c) { c.classList.add("selected"); });
    updateSelbar();
    setStatus("selected " + state.results.length + " shown photo(s)");
  }
  $("#selectall").addEventListener("click", selectAllShown);
  selectAllTop.addEventListener("click", selectAllShown);

  $("#clearsel").addEventListener("click", function () {
    state.selected.clear();
    $$(".card.selected", gridEl).forEach(function (c) { c.classList.remove("selected"); });
    updateSelbar();
  });

  var exportTimer = null;
  var EXPORT_POLL_MAX_FAILS = 5;
  function pollExportProgress(label) {
    if (exportTimer) clearInterval(exportTimer);
    var fails = 0;
    exportTimer = setInterval(function () {
      api("/api/export/progress").then(function (p) {
        fails = 0;
        if (p.running) {
          setStatus(label + " " + p.done + "/" + p.total + (p.failed ? ", " + p.failed + " failed" : "") + (p.skipped ? ", " + p.skipped + " already there" : ""), true);
          return;
        }
        clearInterval(exportTimer); exportTimer = null;
        if (p.error) { setStatus("export failed: " + p.error, true); return; }
        var written = p.done - p.failed - (p.skipped || 0);
        var msg = "exported " + written + " of " + p.total + " to " + p.path;
        if (p.skipped) msg += ", " + p.skipped + " already there";
        if (p.failed) msg += " (" + p.failed + " failed, see failed.txt)";
        setStatus(msg, true);
      }).catch(function () {
        fails += 1;
        if (fails < EXPORT_POLL_MAX_FAILS) return;     // one dropped poll is not a lost server
        clearInterval(exportTimer); exportTimer = null;
        setStatus("lost contact with the server; check the terminal", true);
      });
    }, 800);
  }
  function startExport(ids, name, mode, label) {
    return api("/api/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: ids, name: name, mode: mode }),
    }).then(function () { pollExportProgress(label); })
      .catch(function (err) { setStatus("export failed: " + err.message, true); });
  }

  $("#export").addEventListener("click", function () {
    var ids = Array.from(state.selected);
    if (!ids.length) { setStatus("select some photos first"); return; }
    var name = $("#exportname").value.trim() || "export";
    var mode = $("#exportmode").value;
    startExport(ids, name, mode, "exporting " + ids.length + " photo(s)");
  });

  // ---------- lightbox ----------
  var lightbox = $("#lightbox");
  var lbImg = $("#lb-img");
  var lbMeta = $("#lb-meta");
  var lbLike = $("#lb-like");
  var currentLb = null;

  function openLightbox(r) {
    currentLb = r;
    lbImg.src = "/api/thumb/" + r.qhash + "?size=full";
    var sharpPct = r.sharp_pct != null ? Math.round(r.sharp_pct) : 0;
    var bits = [r.rel, r.width + "×" + r.height, "sharp " + sharpPct + "%", facesLabel(r.n_faces) + " faces"];
    if (r.taken_at) bits.push(r.taken_at);
    lbMeta.textContent = bits.join("  ·  ");
    lightbox.hidden = false;
  }
  function closeLightbox() {
    lightbox.hidden = true;
    currentLb = null;
  }
  $("#lb-close").addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", function (e) {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
  });
  lbLike.addEventListener("click", function () {
    if (!currentLb) return;
    var id = currentLb.id;
    closeLightbox();
    showView("search");
    form.querySelector('[name="q"]').value = "";
    runSearch({ image_id: id });
    setStatus("showing photos similar to that one");
  });

  // ---------- people ----------
  var peopleEl = $("#people");
  var personSelect = $("#person-select");

  function loadPeople() {
    return api("/api/people").then(function (list) {
      state.people = list;
      renderPeople();
      fillPersonSelect();
      return list;
    });
  }

  function renderPeople() {
    peopleEl.innerHTML = "";
    state.people.forEach(function (p) {
      var card = document.createElement("div");
      card.className = "card person";

      if (p.cover_qhash) {
        var img = document.createElement("img");
        img.loading = "lazy";
        img.alt = p.name || ("person " + p.id);
        img.src = "/api/thumb/" + p.cover_qhash + "?size=full";
        img.addEventListener("load", function () { cropToFace(img, p.cover_box); }, { once: true });
        card.appendChild(img);
      }

      var tag = document.createElement("div");
      tag.className = "tag mono";
      tag.textContent = p.n + " photo" + (p.n === 1 ? "" : "s");
      card.appendChild(tag);

      var nameWrap = document.createElement("div");
      nameWrap.className = "pname";
      var nameInput = document.createElement("input");
      nameInput.value = p.name || "";
      nameInput.placeholder = "person_" + String(p.id).padStart(2, "0");
      nameInput.className = "mono";
      nameInput.addEventListener("click", function (e) { e.stopPropagation(); });
      nameInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") { e.preventDefault(); nameInput.blur(); }   // blur does the save, once
      });
      nameInput.addEventListener("blur", function () {
        var val = nameInput.value.trim();
        if (val === (p.name || "")) return;
        api("/api/people/" + p.id + "/name", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: val }),
        }).then(function () {
          p.name = val;
          fillPersonSelect();
          setStatus("named person " + p.id + ": " + (val || "(cleared)"));
        }).catch(function (err) {
          setStatus("could not save name: " + err.message);
        });
      });
      nameWrap.appendChild(nameInput);
      card.appendChild(nameWrap);

      card.addEventListener("click", function () {
        showView("search");
        personSelect.value = String(p.id);
        runSearch({ person: p.id });
      });

      peopleEl.appendChild(card);
    });
  }

  // Draw the face box (padded 1.6x, clamped to the image) into a square canvas with
  // object-fit: cover semantics, then swap the img to that crop. cover_box is in the
  // same pixel space as the full-size thumb the img loaded.
  var COVER_EDGE = 320;
  function cropToFace(img, box) {
    box = box || [0, 0, 0, 0];
    var W = img.naturalWidth, H = img.naturalHeight;
    if (!W || !H || !box[2] || !box[3]) return;
    var pad = 1.6;
    var cx = box[0] + box[2] / 2, cy = box[1] + box[3] / 2;
    var side = Math.max(box[2], box[3]) * pad;
    side = Math.min(side, W, H);
    var sx = Math.min(Math.max(cx - side / 2, 0), W - side);
    var sy = Math.min(Math.max(cy - side / 2, 0), H - side);
    try {
      var canvas = document.createElement("canvas");
      canvas.width = COVER_EDGE; canvas.height = COVER_EDGE;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(img, sx, sy, side, side, 0, 0, COVER_EDGE, COVER_EDGE);
      img.src = canvas.toDataURL("image/jpeg", 0.85);
    } catch (e) {
      /* canvas unavailable or tainted: keep the plain thumb */
    }
  }

  function fillPersonSelect() {
    var current = personSelect.value;
    personSelect.innerHTML = '<option value="">anyone</option>';
    state.people.forEach(function (p) {
      var opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = (p.name || "person_" + String(p.id).padStart(2, "0")) + " (" + p.n + ")";
      personSelect.appendChild(opt);
    });
    personSelect.value = current || "";
  }

  $("#cluster").addEventListener("click", function () {
    var eps = parseFloat($("#eps").value) || 0.5;
    setStatus("grouping faces…", true);
    api("/api/people/cluster", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ eps: eps }),
    }).then(function (list) {
      state.people = list;
      renderPeople();
      fillPersonSelect();
      setStatus("found " + list.length + " group(s)");
      loadStats();
    }).catch(function (err) {
      setStatus("grouping failed: " + err.message);
    });
  });

  $("#export-people").addEventListener("click", function () {
    setStatus("exporting people, groups and solo shots…", true);
    api("/api/export/people", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "symlink" }),
    }).then(function (res) {
      setStatus("exported links to " + res.path);
    }).catch(function (err) {
      setStatus("export failed: " + err.message);
    });
  });

  // find a person from a reference photo
  var findSim = $("#find-sim");
  var findSimOut = $("#find-sim-out");
  var FIND_SIM_DEFAULT = findSim.value;

  function showFindResults(data, who) {
    // Not a paged search: every match is already here, so the "show more" and
    // "select all matching" affordances are hidden after renderGrid() re-shows them.
    // who: "that person" for a picked photo, or a saved name (that payload has no reference keys).
    state.results = data.results || [];
    state.total = data.total || 0;
    state.offset = state.results.length;
    state.lastParams = {};
    showView("search");
    renderGrid();
    $("#show-more").hidden = true;
    $("#select-matching").hidden = true;
    $("#shown-count").textContent = state.results.length + " shown";
    if ("faces_in_reference" in data && !data.faces_in_reference) {
      setStatus("no face found in that photo");
      return;
    }
    if (data.reference_face_too_small) {
      setStatus("the face in that photo is too small to match, pick a closer shot");
      return;
    }
    setStatus("found " + state.total + " photo(s) of " + (who || "that person") + (data.person_id ? ", person " + data.person_id : ""));
  }

  $("#find-person").addEventListener("click", function () {
    setStatus("waiting for the photo picker…", true);
    fetch("/api/people/find/choose", { method: "POST" }).then(function (r) {
      if (r.status === 204) { setStatus("photo pick cancelled"); return null; }
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          throw new Error((body && body.detail) || (r.status + " " + r.statusText));
        });
      }
      return r.json();
    }).then(function (data) {
      if (!data) return;
      // Only a usable face can be re-matched or saved under a name.
      var usable = !!(data.path && data.faces_in_reference && !data.reference_face_too_small);
      state.findPath = usable ? data.path : null;
      syncSaveForm();
      // A fresh pick matches at the default threshold, so the slider shows that too.
      findSim.value = FIND_SIM_DEFAULT; findSimOut.textContent = FIND_SIM_DEFAULT;
      showFindResults(data);
    }).catch(function (err) {
      setStatus("could not find that person: " + err.message);
    });
  });

  findSim.addEventListener("input", function () { findSimOut.textContent = findSim.value; });
  findSim.addEventListener("change", function () {
    if (state.savedPeople.length) loadReferences();    // counts follow the slider
    if (!state.findPath) return;
    setStatus("matching at " + findSim.value + "…", true);
    api("/api/people/find", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.findPath, min_sim: parseFloat(findSim.value) }),
    }).then(function (data) { showFindResults(data); }).catch(function (err) {
      setStatus("could not find that person: " + err.message);
    });
  });

  // ---------- named people (saved reference photos) ----------
  var savePersonForm = $("#save-person-form");
  var savePersonName = $("#save-person-name");
  var savedPeopleEl = $("#saved-people");
  var peopleExportRow = $("#people-export-row");

  function syncSaveForm() {
    savePersonForm.hidden = !state.findPath;
  }

  function savePerson() {
    var name = savePersonName.value.trim();
    if (!state.findPath) { setStatus("find a person from a photo first"); return; }
    if (!name) { setStatus("type a name for this person"); savePersonName.focus(); return; }
    setStatus("saving " + name + "…", true);
    api("/api/people/references", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, path: state.findPath }),
    }).then(function (res) {
      savePersonName.value = "";
      state.peopleUnticked.delete(res.name);
      setStatus("saved " + res.name + "; the People tab lists everyone saved");
      return loadReferences();
    }).catch(function (err) {
      setStatus("could not save that person: " + err.message);
    });
  }
  $("#save-person").addEventListener("click", savePerson);
  savePersonName.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); savePerson(); }
  });

  function loadReferences() {
    if (!(state.folder && state.folder.root)) return Promise.resolve([]);
    var qs = new URLSearchParams({ min_sim: findSim.value }).toString();
    return api("/api/people/references?" + qs).then(function (data) {
      state.savedPeople = (data && data.people) || [];
      renderSavedPeople();
      return state.savedPeople;
    }).catch(function (err) {
      setStatus("could not load saved people: " + err.message);
    });
  }

  function findSaved(name) {
    setStatus("matching " + name + " at " + findSim.value + "…", true);
    api("/api/people/references/" + encodeURIComponent(name) + "/find", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ min_sim: parseFloat(findSim.value) }),
    }).then(function (data) { showFindResults(data, name); }).catch(function (err) {
      setStatus("could not show " + name + ": " + err.message);
    });
  }

  function renameSaved(p, newName) {
    return api("/api/people/references/" + encodeURIComponent(p.name) + "/rename", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName }),
    }).then(function () {
      if (state.peopleUnticked.has(p.name)) { state.peopleUnticked.delete(p.name); state.peopleUnticked.add(newName); }
      setStatus("renamed " + p.name + " to " + newName);
      return loadReferences();
    }).catch(function (err) {
      setStatus("could not rename: " + err.message);
      return loadReferences();
    });
  }

  function removeSaved(p) {
    setStatus("removing " + p.name + "…", true);
    return Promise.all(p.reference_ids.map(function (id) {
      return api("/api/people/references/" + id, { method: "DELETE" });
    })).then(function () {
      state.peopleUnticked.delete(p.name);
      setStatus("removed " + p.name);
      return loadReferences();
    }).catch(function (err) {
      setStatus("could not remove " + p.name + ": " + err.message);
      return loadReferences();
    });
  }

  function renderSavedPeople() {
    savedPeopleEl.innerHTML = "";
    peopleExportRow.hidden = !state.savedPeople.length;
    if (!state.savedPeople.length) {
      var empty = document.createElement("p");
      empty.className = "mono";
      empty.textContent = "no saved people yet: Find a person from a photo, then name them in the status line";
      savedPeopleEl.appendChild(empty);
      return;
    }
    state.savedPeople.forEach(function (p) {
      var row = document.createElement("div");
      row.className = "ref-row";

      var tickLabel = document.createElement("label");
      tickLabel.title = "include in Export ticked people";
      var tick = document.createElement("input");
      tick.type = "checkbox";
      tick.className = "ref-tick";
      tick.value = p.name;
      tick.checked = !state.peopleUnticked.has(p.name);
      tick.addEventListener("change", function () {
        if (tick.checked) state.peopleUnticked.delete(p.name); else state.peopleUnticked.add(p.name);
      });
      tickLabel.appendChild(tick);
      row.appendChild(tickLabel);

      var nameInput = document.createElement("input");
      nameInput.type = "text";
      nameInput.className = "mono ref-name";
      nameInput.value = p.name;
      nameInput.title = p.reference_ids.length + " reference photo" + (p.reference_ids.length === 1 ? "" : "s") + ": " + p.sources.join(", ");
      nameInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") { e.preventDefault(); nameInput.blur(); }   // blur does the save, once
      });
      nameInput.addEventListener("blur", function () {
        var val = nameInput.value.trim();
        if (!val) { nameInput.value = p.name; return; }
        if (val === p.name) return;
        renameSaved(p, val);
      });
      row.appendChild(nameInput);

      var count = document.createElement("span");
      count.className = "mono ref-count";
      count.textContent = p.count + " photo" + (p.count === 1 ? "" : "s");
      row.appendChild(count);

      var show = document.createElement("button");
      show.type = "button";
      show.textContent = "Show";
      show.addEventListener("click", function () { findSaved(p.name); });
      row.appendChild(show);

      // Two clicks to remove, no browser dialog: the first arms the button, the second deletes.
      // A second click that lands within 400ms of the arming one is a double-click, not a
      // deliberate confirm, so it is ignored rather than treated as the delete.
      var remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "x";
      remove.title = "remove " + p.name + " (two clicks)";
      var armTimer = null;
      var armedAt = 0;
      function disarm() {
        if (armTimer) { clearTimeout(armTimer); armTimer = null; }
        remove.classList.remove("armed");
        remove.textContent = "x";
      }
      remove.addEventListener("click", function () {
        if (!remove.classList.contains("armed")) {
          remove.classList.add("armed");
          remove.textContent = "really remove?";
          armedAt = Date.now();
          armTimer = setTimeout(disarm, 5000);
          return;
        }
        if (Date.now() - armedAt <= 400) return;   // a double-click landed as the confirm, not a real one
        disarm();
        removeSaved(p);
      });
      remove.addEventListener("blur", disarm);
      row.appendChild(remove);

      savedPeopleEl.appendChild(row);
    });
  }

  $("#people-export-refs").addEventListener("click", function () {
    var names = $$(".ref-tick", savedPeopleEl).filter(function (b) { return b.checked; }).map(function (b) { return b.value; });
    if (!names.length) { setStatus("tick at least one person"); return; }
    var mode = $("#people-export-mode").value;
    var includeRaw = $("#people-include-raw").checked;
    setStatus("exporting " + names.length + " " + (names.length === 1 ? "person" : "people") + "…", true);
    api("/api/export/references", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: names, mode: mode, include_raw: includeRaw, min_sim: parseFloat(findSim.value) }),
    }).then(function () { pollExportProgress("exporting people"); })
      .catch(function (err) { setStatus("export failed: " + err.message, true); });
  });

  // ---------- categories ----------
  // Two rows: the fixed CATEGORIES (a filter on photos.category) and the ones discovered in this shoot
  // (k-means clusters named from the vocabulary, a filter on photos.cluster). A tile key is the fixed
  // name, or "discovered:" + name, so the tick state of the two rows never collides.
  var catTilesEl = $("#cat-tiles");
  var discTilesEl = $("#disc-tiles");
  var catProgressEl = $("#cat-progress");
  var categoryChip = $("#category-chip");

  function showCategoryChip(text) {
    if (!text) { categoryChip.hidden = true; return; }
    $("#category-chip-name").textContent = text;
    categoryChip.hidden = false;
  }
  $("#category-chip-clear").addEventListener("click", function () {
    $("#category-filter").value = ""; $("#cluster-filter").value = "";
    showCategoryChip(null);
    runSearch();
  });

  // One of the two filters at a time: a fixed category, or a discovered one (cluster).
  function filterByCategory(cat, cluster) {
    $("#category-filter").value = cluster ? "" : cat;
    $("#cluster-filter").value = cluster ? cat : "";
    showCategoryChip((cluster ? "discovered: " : "category: ") + cat);
    showView("search");
    runSearch();
  }

  function loadCategories() {
    return api("/api/categories").then(function (counts) {
      state.categories = { fixed: (counts && counts.fixed) || {}, discovered: (counts && counts.discovered) || {} };
      renderCategoryTiles();
    }).catch(function (err) {
      setStatus("could not load categories: " + err.message);
    });
  }

  var catExportRow = $("#cat-export-row");

  function makeTile(cat, count, cluster) {
    var key = cluster ? "discovered:" + cat : cat;
    var tile = document.createElement("div");
    tile.className = "cat-tile";

    var tick = document.createElement("label");
    tick.className = "cat-tile-tick";
    tick.title = "include in Export ticked categories";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.className = cluster ? "disc-tick" : "cat-tick";
    box.value = cat;
    if (!state.catSeen.has(key)) {                 // first sight: everything but "unclassified" starts ticked
      state.catSeen.add(key);
      if (cat !== "unclassified") state.catTicked.add(key);
    }
    box.checked = state.catTicked.has(key);
    box.addEventListener("change", function () {
      if (box.checked) state.catTicked.add(key); else state.catTicked.delete(key);
    });
    tick.appendChild(box);
    tile.appendChild(tick);

    var label = document.createElement("button");
    label.type = "button";
    label.className = "cat-tile-main mono";
    label.textContent = cat + "  " + count;
    label.addEventListener("click", function () { filterByCategory(cat, cluster); });
    tile.appendChild(label);

    var exportBtn = document.createElement("button");
    exportBtn.type = "button";
    exportBtn.className = "mono";
    exportBtn.textContent = "Export links";
    exportBtn.addEventListener("click", function () { exportCategory(cat, cluster); });
    tile.appendChild(exportBtn);
    return tile;
  }

  function renderTileRow(el, counts, cluster, emptyText) {
    el.innerHTML = "";
    var names = Object.keys(counts);
    if (!names.length) {
      var p = document.createElement("p");
      p.className = "mono";
      p.textContent = emptyText;
      el.appendChild(p);
      return 0;
    }
    names.forEach(function (cat) { el.appendChild(makeTile(cat, counts[cat], cluster)); });
    return names.length;
  }

  function renderCategoryTiles() {
    var nFixed = renderTileRow(catTilesEl, state.categories.fixed, false, "no categories yet, run Categorise");
    var nDisc = renderTileRow(discTilesEl, state.categories.discovered, true, "no discovered categories yet, run Categorise");
    catExportRow.hidden = !(nFixed || nDisc);
  }

  function exportCategory(cat, cluster) {
    setStatus("gathering " + cat + " photos…", true);
    var params = cluster ? { cluster: cat } : { category: cat };
    if (!$("#cat-include-unsure").checked) params.sure_only = 1;
    var qs = new URLSearchParams(params).toString();
    return api("/api/search/ids?" + qs).then(function (data) {
      var ids = data.ids || [];
      if (!ids.length) { setStatus("no photos in " + cat); return null; }
      setStatus("exporting " + ids.length + " " + cat + " photo(s)…", true);
      return startExport(ids, "categories/" + (cluster ? "discovered/" : "") + cat, "symlink", "exporting " + cat);
    }).catch(function (err) {
      setStatus("export failed: " + err.message);
    });
  }

  $("#cat-export-all").addEventListener("click", function () {
    var ticked = function (sel) { return $$(sel).filter(function (b) { return b.checked; }).map(function (b) { return b.value; }); };
    var cats = ticked(".cat-tick");
    var disc = ticked(".disc-tick");
    var n = cats.length + disc.length;
    if (!n) { setStatus("tick at least one category"); return; }
    var mode = $("#cat-export-mode").value;
    var includeRaw = $("#cat-include-raw").checked;
    var includeUnsure = $("#cat-include-unsure").checked;
    setStatus("exporting " + n + " categor" + (n === 1 ? "y" : "ies") + "…", true);
    api("/api/export/categories", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ categories: cats, discovered: disc, mode: mode, include_raw: includeRaw, include_unsure: includeUnsure }),
    }).then(function () { pollExportProgress("exporting categories"); })
      .catch(function (err) { setStatus("export failed: " + err.message, true); });
  });

  function stopClassifyPoll() {
    if (state.classifyTimer) {
      clearInterval(state.classifyTimer);
      state.classifyTimer = null;
    }
  }

  function pollClassifyProgress() {
    stopClassifyPoll();
    state.classifyTimer = setInterval(function () {
      api("/api/classify/progress").then(function (p) {
        catProgressEl.textContent = p.running ? "categorising…" : "";
        if (!p.running) {
          stopClassifyPoll();
          loadCategories();           // both rows, fixed and discovered, from the same endpoint the tab opens with
          setStatus(p.error ? ("categorising failed: " + p.error) : "categorising finished");
        }
      }).catch(function () { stopClassifyPoll(); });
    }, 800);
  }

  $("#categorise").addEventListener("click", function () {
    api("/api/classify", { method: "POST" }).then(function () {
      setStatus("categorising…", true);
      catProgressEl.textContent = "categorising…";
      pollClassifyProgress();
    }).catch(function (err) {
      if (err.status === 409) {
        setStatus("already categorising");
        pollClassifyProgress();
      } else if (err.status === 501) {
        setStatus("categorisation isn't available yet");
      } else {
        setStatus("could not start categorising: " + err.message);
      }
    });
  });

  // ---------- index ----------
  var progressEl = $("#progress");

  function stopProgressPoll() {
    if (state.progressTimer) {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
    }
  }

  function formatProgress(p) {
    if (p.stage === "error") return "indexing failed: " + (p.error || "unknown error");
    var done = p.done || 0, total = p.total || 0;
    var line = p.stage + "  " + done + "/" + total;
    if (p.running && p.stage_started && done > 0 && total > done) {
      var elapsed = Date.now() / 1000 - p.stage_started;
      var rate = done / Math.max(elapsed, 0.001);
      var eta = (total - done) / rate;
      line += "  " + rate.toFixed(1) + "/s, about " + (eta < 90 ? Math.round(eta) + " s" : Math.round(eta / 60) + " min") + " left";
    } else if (p.running) {
      line += "  (running)";
    }
    return line;
  }

  function pollProgress() {
    stopProgressPoll();
    state.progressTimer = setInterval(function () {
      api("/api/progress").then(function (p) {
        progressEl.textContent = formatProgress(p);
        if (!p.running) {
          stopProgressPoll();
          if (p.stage === "error") {
            setStatus("indexing failed: " + (p.error || "unknown error"), true);
          } else {
            setStatus("indexing finished");
          }
          // the index changed under us: refresh everything that shows it
          loadStats();
          loadPeople();
          runSearch();
        }
      }).catch(function () { stopProgressPoll(); });
    }, 800);
  }

  $("#start-index").addEventListener("click", function () {
    var faces = $("#faces").checked;
    api("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ faces: faces, retry_errors: $("#retry-errors").checked }),
    }).then(function () {
      setStatus("indexing started…", true);
      progressEl.textContent = "starting…";
      pollProgress();
      loadStats();
    }).catch(function (err) {
      if (err.status === 409) {
        setStatus("already indexing");
        pollProgress();
      } else {
        setStatus("could not start indexing: " + err.message);
      }
    });
  });

  // index bundles: pack the index into one zip, or install one picked from disk
  // A POST to a picker endpoint: null on 204 (cancelled), the JSON body otherwise, an Error on failure.
  function pickerPost(path, body) {
    var opts = { method: "POST" };
    if (body) { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(body); }
    return fetch(path, opts).then(function (r) {
      if (r.status === 204) return null;
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (b) {
          var err = new Error((b && b.detail) || (r.status + " " + r.statusText));
          err.status = r.status;
          throw err;
        });
      }
      return r.json();
    });
  }

  $("#bundle-export").addEventListener("click", function () {
    setStatus("packing the index…", true);
    api("/api/bundle/export", { method: "POST" }).then(function () { pollExportProgress("packing index"); })
      .catch(function (err) { setStatus("could not pack the index: " + err.message, true); });
  });

  function importBundle() {
    setStatus("waiting for the bundle picker…", true);
    pickerPost("/api/bundle/import/choose").then(function (res) {
      if (!res) { setStatus("import cancelled"); return null; }
      if (!res.needs_root) return res;
      // Made on a Mac where the disk sat under another path: ask for the folder, then install under it.
      setStatus("that bundle was made for " + res.bundle.root + ", which is not here; pick the photo folder", true);
      return pickerPost("/api/bundle/import/choose-root", { zip: res.zip }).then(function (r2) {
        if (!r2) { setStatus("import cancelled, nothing was installed"); return null; }
        return r2;
      });
    }).then(function (info) {
      if (!info) return;
      applyFolderInfo(info);
      setStatus("imported " + (info.name || info.root) + ": " + info.photos + " photos, ready", true);
      settleFolder(info);
    }).catch(function (err) {
      setStatus("could not import the bundle: " + err.message, true);
    });
  }
  $("#bundle-import").addEventListener("click", importBundle);
  $("#bundle-import-main").addEventListener("click", importBundle);

  // ---------- boot ----------
  loadExportDest();
  loadFolder().then(function (info) {
    loadRecent();
    if (!info.root) {
      showView(state.view);
      return;
    }
    if (!info.indexed) {
      showView("index");
      var btn = $("#start-index");
      if (btn) btn.focus();
      loadStats().then(function (s) { if (s.indexing) pollProgress(); });   // first index of a fresh folder, page reloaded mid-run
      return;
    }
    loadStats().then(function (s) { if (s.indexing) pollProgress(); });
    loadPeople();
    runSearch();
    showView(state.view);
  });
})();
