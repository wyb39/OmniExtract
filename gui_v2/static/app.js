/* ============================================================
   OmniExtract gui_v2 — shared client behavior (no framework)
   ============================================================ */
(function () {
  "use strict";

  // ---------- dynamic field blocks (input/output) ----------
  // Each .field-block has data attributes; cloning is done from the first
  // child of the container so templates stay server-rendered.

  function uniqueId(prefix) {
    return prefix + "-" + Math.random().toString(36).slice(2, 9);
  }

  function resetCloneInputs(clone) {
    clone.querySelectorAll("input, textarea, select").forEach(function (el) {
      if (el.type === "checkbox") { el.checked = false; return; }
      if (el.tagName === "SELECT") { el.selectedIndex = 0; return; }
      el.value = "";
    });
    var rc = clone.querySelector('[data-range-container]');
    var lc = clone.querySelector('[data-literal-container]');
    if (rc) rc.style.display = "none";
    if (lc) lc.style.display = "none";
  }

  function cloneFieldBlock(container, kind) {
    var template = container.querySelector("[data-field-template]");
    if (!template) template = container.children[0];
    if (!template) return;
    var clone = template.cloneNode(true);
    clone.removeAttribute("data-field-template");
    clone.id = (kind || "field") + "-" + uniqueId("b");
    resetCloneInputs(clone);
    container.appendChild(clone);
  }

  // delegate clicks for add / delete, and changes for range/literal toggles
  document.addEventListener("click", function (e) {
    var addBtn = e.target.closest("[data-add-field]");
    if (addBtn) {
      e.preventDefault();
      var containerId = addBtn.getAttribute("data-add-field");
      var container = document.getElementById(containerId);
      if (container) cloneFieldBlock(container, addBtn.getAttribute("data-field-kind") || "field");
      return;
    }
    var delBtn = e.target.closest(".delete-button");
    if (delBtn) {
      var block = delBtn.closest(".field-block");
      if (block) {
        var container = block.parentElement;
        if (container.children.length > 1) block.remove();
      }
    }
    // task card expand toggle
    var toggle = e.target.closest("[data-task-toggle]");
    if (toggle) {
      var body = document.getElementById(toggle.getAttribute("data-task-toggle"));
      if (body) {
        var open = body.style.display !== "none";
        body.style.display = open ? "none" : "block";
        var icon = toggle.querySelector("i");
        if (icon) icon.style.transform = open ? "" : "rotate(180deg)";
      }
    }
  });

  document.addEventListener("change", function (e) {
    var cb = e.target.closest('input[type="checkbox"]');
    if (!cb) return;
    var block = cb.closest(".field-block");
    if (!block) return;
    if (cb.matches('[data-toggle="range"]')) {
      var rc = block.querySelector('[data-range-container]');
      if (rc) rc.style.display = cb.checked ? "block" : "none";
    }
    if (cb.matches('[data-toggle="literal"]')) {
      var lc = block.querySelector('[data-literal-container]');
      if (lc) lc.style.display = cb.checked ? "block" : "none";
    }
  });

  // ---------- field collection (for submission & export) ----------
  // Reads a .field-block container into an array of plain objects.
  window.collectFields = function (containerId) {
    var container = document.getElementById(containerId);
    if (!container) return [];
    var fields = [];
    Array.prototype.forEach.call(container.children, function (block) {
      if (!block.classList || !block.classList.contains("field-block")) return;
      var nameEl = block.querySelector('[data-role="name"]');
      var typeEl = block.querySelector('[data-role="type"]');
      var descEl = block.querySelector('[data-role="description"]');
      var name = nameEl ? nameEl.value.trim() : "";
      if (!name) return;
      var f = { name: name, type: typeEl ? typeEl.value : "str", description: descEl ? descEl.value.trim() : "" };
      var rangeCb = block.querySelector('[data-toggle="range"]');
      var literalCb = block.querySelector('[data-toggle="literal"]');
      f.hasRange = !!(rangeCb && rangeCb.checked);
      f.hasLiteral = !!(literalCb && literalCb.checked);
      if (f.hasRange) {
        var minEl = block.querySelector('[data-role="range-min"]');
        var maxEl = block.querySelector('[data-role="range-max"]');
        f.rangeMin = minEl ? minEl.value : "";
        f.rangeMax = maxEl ? maxEl.value : "";
      }
      if (f.hasLiteral) {
        var listEl = block.querySelector('[data-role="literal-list"]');
        f.literalList = listEl ? listEl.value : "";
      }
      fields.push(f);
    });
    return fields;
  };

  // ---------- build payload from a <form data-task="..."> ----------
  // Collects scalar inputs by name + collects field arrays by data-container.
  window.buildPayload = function (form) {
    var payload = {};
    // scalars
    Array.prototype.forEach.call(form.querySelectorAll("[name]"), function (el) {
      if (el.closest(".field-block")) return; // handled separately
      if (el.type === "checkbox") {
        payload[el.name] = el.checked;
      } else if (el.type === "radio") {
        if (el.checked) payload[el.name] = el.value;
      } else if (el.tagName === "SELECT" && el.multiple) {
        payload[el.name] = Array.prototype.map.call(el.selectedOptions, function (o) { return o.value; });
      } else {
        payload[el.name] = el.value;
      }
    });
    // field arrays
    Array.prototype.forEach.call(form.querySelectorAll("[data-fields]"), function (container) {
      var key = container.getAttribute("data-fields");
      payload[key] = window.collectFields(container.id);
    });
    return payload;
  };

  // ---------- generic form submission ----------
  // <form data-task="doc-parsing" data-result="#result"> ... <button data-submit>
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-submit]");
    if (!btn) return;
    var form = btn.closest("form");
    if (!form) return;
    e.preventDefault();

    var moduleKey = form.getAttribute("data-task");
    var resultSel = form.getAttribute("data-result") || "#task-result";
    var resultEl = document.querySelector(resultSel);
    if (!resultEl) return;

    var payload = window.buildPayload(form);

    var orig = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
    resultEl.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Starting task...</div>';

    fetch("/api/task/run/" + encodeURIComponent(moduleKey), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) { return r.json().then(function (j) { return { ok: r.ok, json: j }; }); })
      .then(function (out) {
        var j = out.json || {};
        if (j.ok) {
          resultEl.innerHTML = renderSummary(j);
          // offer a one-click reload to see the new task card
          var refresh = document.createElement("div");
          refresh.className = "alert alert-info";
          refresh.innerHTML = '<i class="fas fa-circle-info"></i><div><h4>Task started</h4><p>Refresh the page to see it in <strong>Recent Tasks</strong>. ' +
            '<a href="javascript:location.reload()">Reload now</a></p></div>';
          resultEl.appendChild(refresh);
        } else {
          resultEl.innerHTML = '<div class="alert alert-danger"><i class="fas fa-exclamation-circle"></i><div><h4>Error</h4><p>' +
            escapeHtml(j.error || "Unknown error") + '</p></div></div>';
        }
      })
      .catch(function (err) {
        resultEl.innerHTML = '<div class="alert alert-danger"><i class="fas fa-exclamation-circle"></i><div><h4>Request failed</h4><p>' +
          escapeHtml(String(err)) + '</p></div></div>';
      })
      .finally(function () { btn.disabled = false; btn.innerHTML = orig; });
  });

  function renderSummary(j) {
    var cfg = j.config || {};
    var rows = "";
    Object.keys(cfg).forEach(function (k) {
      var v = cfg[k];
      var disp;
      if (Array.isArray(v) && v.length && typeof v[0] === "object") {
        disp = v.length + " item(s)";
      } else if (typeof v === "object" && v !== null) {
        disp = escapeHtml(JSON.stringify(v));
      } else {
        disp = escapeHtml(String(v === undefined ? "" : v));
      }
      rows += '<div class="config-item"><div class="config-item-key">' + escapeHtml(k) +
        '</div><div class="config-item-value">' + disp + '</div></div>';
    });
    return '<div class="alert alert-success"><i class="fas fa-check-circle"></i><div><h4>Task started successfully!</h4>' +
      '<p>Timestamp: <strong>' + escapeHtml(j.timestamp || "") + '</strong></p></div></div>' +
      '<div class="config-items" style="margin-top:12px;">' + rows + '</div>';
  }

  // ---------- task card: cancel ----------
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-cancel]");
    if (!btn) return;
    e.preventDefault();
    var logPath = btn.getAttribute("data-cancel");
    if (!confirm("Cancel this running task?")) return;
    btn.disabled = true; btn.textContent = "Cancelling...";
    fetch("/api/task/cancel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log_path: logPath })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.ok) {
        var pill = btn.closest(".task-card-head").querySelector(".status-pill");
        if (pill) { pill.className = "status-pill cancelled"; pill.textContent = "cancelled"; }
        btn.remove();
      } else {
        alert(j.error || "Failed to cancel task");
        btn.disabled = false; btn.textContent = "Cancel task";
      }
    }).catch(function () { btn.disabled = false; btn.textContent = "Cancel task"; });
  });

  // ---------- task card: view log ----------
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-log]");
    if (!btn) return;
    e.preventDefault();
    OmniLog.open(btn.getAttribute("data-log"), btn.getAttribute("data-log-name") || "");
  });

  // ---------- task card: use this config (writeback) ----------
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-use-config]");
    if (!btn) return;
    e.preventDefault();
    var path = btn.getAttribute("data-use-config");
    var moduleKey = btn.getAttribute("data-module");
    fetch("/api/task/config?path=" + encodeURIComponent(path) + "&module=" + encodeURIComponent(moduleKey))
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (!j.ok) { alert(j.error || "Cannot load config"); return; }
        if (window.applyWriteback) window.applyWriteback(j.values || {});
        // jump to top form
        var form = document.querySelector('form[data-task]');
        if (form) form.scrollIntoView({ behavior: "smooth", block: "start" });
        flashNotice("Config loaded into the form above.");
      })
      .catch(function (err) { alert("Failed: " + err); });
  });

  // ---------- generic writeback application ----------
  // Fills scalars by [name], and rebuilds field blocks via data-fields containers.
  window.applyWriteback = function (values) {
    Object.keys(values).forEach(function (key) {
      if (["inputFields", "outputFields", "fields"].indexOf(key) !== -1) return;
      var els = document.querySelectorAll('[name="' + cssEscape(key) + '"]');
      Array.prototype.forEach.call(els, function (el) {
        if (el.type === "checkbox") { el.checked = !!values[key]; }
        else { el.value = values[key] === null || values[key] === undefined ? "" : values[key]; }
      });
    });
    // field arrays
    [["inputFields", "inputFields"], ["outputFields", "outputFields"], ["fields", "fields"]].forEach(function (pair) {
      var cfgKey = pair[0], containerDataKey = pair[1];
      var arr = values[cfgKey];
      if (!arr || !arr.length) return;
      var container = document.querySelector('[data-fields="' + containerDataKey + '"]');
      if (!container) return;
      // remove all but the first (template) block
      while (container.children.length > 1) container.removeChild(container.lastChild);
      // ensure exactly arr.length blocks (clone from the first template)
      while (container.children.length < arr.length) {
        cloneFieldBlock(container, container.getAttribute("data-field-kind") || "field");
      }
      // if fewer fields than current blocks, trim extras
      while (container.children.length > arr.length) {
        container.removeChild(container.lastChild);
      }
      // fill each block by index
      arr.forEach(function (f, idx) {
        var block = container.children[idx];
        if (!block) return;
        setName(block, '[data-role="name"]', f.name);
        setName(block, '[data-role="description"]', f.description);
        var typeEl = block.querySelector('[data-role="type"]');
        if (typeEl && f.field_type) typeEl.value = f.field_type;
        var hasRange = (f.range_min !== undefined && f.range_max !== undefined && f.range_min !== null && f.range_max !== null);
        var lit = f.literal_list;
        var hasLiteral = !!lit && (Array.isArray(lit) ? lit.length > 0 : String(lit).length > 0);
        var rangeCb = block.querySelector('[data-toggle="range"]');
        var literalCb = block.querySelector('[data-toggle="literal"]');
        if (rangeCb) { rangeCb.checked = hasRange; }
        if (literalCb) { literalCb.checked = hasLiteral; }
        var rc = block.querySelector('[data-range-container]'); if (rc) rc.style.display = hasRange ? "block" : "none";
        var lc = block.querySelector('[data-literal-container]'); if (lc) lc.style.display = hasLiteral ? "block" : "none";
        if (hasRange) {
          setName(block, '[data-role="range-min"]', f.range_min);
          setName(block, '[data-role="range-max"]', f.range_max);
        }
        if (hasLiteral) {
          var txt = Array.isArray(lit) ? lit.join(",") : lit;
          setName(block, '[data-role="literal-list"]', txt);
        }
      });
    });
  };

  function setName(block, selector, val) {
    var el = block.querySelector(selector);
    if (el && val !== undefined && val !== null) el.value = val;
  }

  // ---------- real-time log viewer ----------
  var OmniLog = {
    timer: null,
    path: null,
    polling: true,
    open: function (path, label) {
      this.path = path;
      this.polling = true;
      document.getElementById("log-modal-label").textContent = label || path;
      document.getElementById("log-modal-content").textContent = "Loading...";
      document.getElementById("log-poll-label").textContent = "Pause";
      document.getElementById("log-modal").style.display = "flex";
      this.fetch();
      var self = this;
      if (this.timer) clearInterval(this.timer);
      this.timer = setInterval(function () { if (self.polling) self.fetch(); }, 2000);
    },
    close: function () {
      if (this.timer) { clearInterval(this.timer); this.timer = null; }
      document.getElementById("log-modal").style.display = "none";
    },
    togglePoll: function () {
      this.polling = !this.polling;
      document.getElementById("log-poll-label").textContent = this.polling ? "Pause" : "Resume";
    },
    fetch: function () {
      if (!this.path) return;
      var self = this;
      fetch("/api/task/log?path=" + encodeURIComponent(this.path))
        .then(function (r) { return r.json(); })
        .then(function (j) {
          if (!j.ok) {
            var el = document.getElementById("log-modal-content");
            if (el && !el.textContent.trim()) el.textContent = "(log not available yet)";
            return;
          }
          var el = document.getElementById("log-modal-content");
          el.textContent = j.content || "(empty)";
          if (document.getElementById("log-autoscroll").checked) {
            el.scrollTop = el.scrollHeight;
          }
        })
        .catch(function () {});
    }
  };
  window.OmniLog = OmniLog;
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") OmniLog.close(); });

  // ---------- in-page sub tabs ----------
  document.addEventListener("click", function (e) {
    var tab = e.target.closest("[data-tab]");
    if (!tab) return;
    var group = tab.getAttribute("data-tab-group");
    var target = tab.getAttribute("data-tab");
    document.querySelectorAll('[data-tab-group="' + group + '"]').forEach(function (t) { t.classList.remove("active"); });
    tab.classList.add("active");
    document.querySelectorAll('[data-tab-pane="' + group + '"]').forEach(function (p) {
      p.classList.toggle("active", p.getAttribute("data-tab-pane-id") === target);
    });
  });

  // ---------- helpers ----------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function cssEscape(s) { return String(s).replace(/"/g, '\\"'); }
  function flashNotice(msg) {
    var box = document.createElement("div");
    box.className = "alert alert-success";
    box.innerHTML = '<i class="fas fa-check-circle"></i><div><h4>' + escapeHtml(msg) + '</h4></div>';
    box.style.position = "fixed"; box.style.right = "24px"; box.style.bottom = "24px"; box.style.zIndex = "2000";
    box.style.maxWidth = "360px"; box.style.margin = "0";
    document.body.appendChild(box);
    setTimeout(function () { box.style.transition = "opacity .4s"; box.style.opacity = "0"; setTimeout(function () { box.remove(); }, 500); }, 2600);
  }

  window.OmniUtils = { escapeHtml: escapeHtml };
})();
