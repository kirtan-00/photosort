(function () {
  "use strict";

  var state = {
    view: "search",
    results: [],
    selected: new Set(),
    people: [],
    progressTimer: null,
  };

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  var statusEl = $("#status");
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
    $$("main > section").forEach(function (s) {
      s.hidden = s.id !== "view-" + name;
    });
    if (name === "people" && state.people.length === 0) loadPeople();
  }
  $$("header nav button").forEach(function (b) {
    b.addEventListener("click", function () { showView(b.dataset.view); });
  });

  // ---------- stats ----------
  function loadStats() {
    return api("/api/stats").then(function (s) {
      var bits = [s.photos + " photos", s.faces + " faces", s.people + " people"];
      if (s.last_index) bits.push("indexed " + s.last_index);
      if (s.indexing) bits.push("indexing…");
      $("#stats").textContent = bits.join("  ·  ");
      var root = String(s.root || "").replace(/\/+$/, "");
      var folderEl = $("#folder");
      folderEl.textContent = root.split("/").pop() || root;
      folderEl.title = root;
      return s;
    });
  }

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
    return params;
  }

  function runSearch(extra) {
    var params = currentFilters();
    Object.assign(params, extra || {});
    params.limit = params.person ? 1000 : 200;
    var qs = new URLSearchParams(params).toString();
    return api("/api/search?" + qs).then(function (data) {
      state.results = data.results || [];
      renderGrid();
    }).catch(function (err) {
      if (err.status === 404) {
        state.results = [];
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

  // n_faces is null when the photo was indexed with faces off: not "0", just unknown
  function facesLabel(n) { return n == null ? "?" : String(n); }

  var gridEl = $("#grid");
  function renderGrid() {
    gridEl.innerHTML = "";
    state.results.forEach(function (r) {
      var card = document.createElement("div");
      card.className = "card";
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
      card.appendChild(tag);

      card.addEventListener("click", function () {
        toggleSelect(r.id, card);
      });
      card.addEventListener("dblclick", function () {
        openLightbox(r);
      });
      gridEl.appendChild(card);
    });
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

  $("#export").addEventListener("click", function () {
    var ids = Array.from(state.selected);
    if (!ids.length) { setStatus("select some photos first"); return; }
    var name = $("#exportname").value.trim() || "export";
    var mode = $("#exportmode").value;
    setStatus("exporting " + ids.length + " photo(s)…", true);
    api("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: ids, name: name, mode: mode }),
    }).then(function (res) {
      setStatus("exported " + ids.length + " photo(s) to " + res.path);
    }).catch(function (err) {
      setStatus("export failed: " + err.message);
    });
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
      body: JSON.stringify({ mode: "copy" }),
    }).then(function (res) {
      setStatus("exported to " + res.path);
    }).catch(function (err) {
      setStatus("export failed: " + err.message);
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
    var line = p.stage + "  " + (p.done || 0) + "/" + (p.total || 0);
    if (p.running) line += "  (running)";
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
      body: JSON.stringify({ faces: faces }),
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

  // ---------- boot ----------
  loadStats();
  loadPeople();
  runSearch();
})();
