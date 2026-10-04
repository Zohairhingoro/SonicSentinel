/**
 * mic_status.js — Microphone Status Widget Controller (FE-8a)
 *
 * Manages one of five mic states defined by AGENTS.md §3 (Audio ingestion):
 *   "Available"        – permission granted, no active stream
 *   "Active"           – MediaStream is open (capture starts in FE-8b)
 *   "Paused"           – stream was suspended
 *   "Disconnected"     – device not found or removed mid-session
 *   "Permission denied"– getUserMedia rejected by user or browser policy
 *
 * Responsibilities of this module:
 *   • Probe the Permissions API on load to get an immediate initial state.
 *   • Call getUserMedia({audio:true}) to resolve the real permission state
 *     when the API alone is insufficient (Firefox, Safari, prompt state).
 *   • React to PermissionStatus.onchange events without a page reload.
 *   • Listen for device-change events (devicechange) to detect disconnections.
 *   • Expose window.MicStatus.set(state) so FE-8b can push "Active"/"Paused".
 *   • Render all visual changes by mutating DOM class lists — no innerHTML writes.
 *
 * This module does NOT start audio capture; that is FE-8b.
 */

(() => {
  "use strict";

  // ── State catalogue ──────────────────────────────────────────────────────────

  /** @typedef {"available"|"active"|"paused"|"disconnected"|"denied"} MicState */

  /**
   * Visual config for each state.
   * ring*   – Tailwind classes for the icon ring background + border
   * icon*   – Tailwind text-color for the SVG
   * halo*   – Tailwind background for the ping halo (empty string = hidden)
   * label*  – Human-readable label shown in the widget
   * detail* – Sub-text hint
   * border* – Widget border override
   */
  const STATE_CONFIG = {
    available: {
      ring: "bg-emerald-500/10 border border-emerald-500/30",
      icon: "text-emerald-400",
      halo: "",
      label: "Available",
      detail: "Ready — capture not started",
      border: "border-emerald-500/30",
      labelColor: "text-emerald-300",
      iconPath: MIC_PATH,
      showBtn: false,
    },
    active: {
      ring: "bg-cyan-500/10 border border-cyan-500/40",
      icon: "text-cyan-400",
      halo: "bg-cyan-500/30",
      label: "Active",
      detail: "Microphone is capturing audio",
      border: "border-cyan-500/40",
      labelColor: "text-cyan-300",
      iconPath: MIC_PATH,
      showBtn: false,
    },
    paused: {
      ring: "bg-amber-500/10 border border-amber-500/30",
      icon: "text-amber-400",
      halo: "",
      label: "Paused",
      detail: "Capture suspended",
      border: "border-amber-500/30",
      labelColor: "text-amber-300",
      iconPath: MIC_PATH,
      showBtn: false,
    },
    disconnected: {
      ring: "bg-rose-500/10 border border-rose-500/30",
      icon: "text-rose-400",
      halo: "",
      label: "Disconnected",
      detail: "No microphone device found",
      border: "border-rose-500/30",
      labelColor: "text-rose-300",
      iconPath: MIC_SLASH_PATH,
      showBtn: true,
    },
    denied: {
      ring: "bg-rose-500/10 border border-rose-500/30",
      icon: "text-rose-400",
      halo: "",
      label: "Permission denied",
      detail: "Allow microphone access in browser settings",
      border: "border-rose-500/30",
      labelColor: "text-rose-300",
      iconPath: MIC_SLASH_PATH,
      showBtn: false,
    },
  };

  // SVG path data — mic icon and mic-slash (crossed-out) icon
  const MIC_PATH =
    "M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z";
  const MIC_SLASH_PATH =
    "M5.586 5.586A9 9 0 0021 12m-9 9a9 9 0 01-9-9 " +
    "M9 9a3 3 0 015.121-2.121M15 9.34V6a3 3 0 00-5.94-.6M17 " +
    "16.48A9.955 9.955 0 0121 12M3 3l18 18";

  // ── Patch STATE_CONFIG with correct path constants ────────────────────────
  // (defined above the object literal so we can reference them cleanly)
  STATE_CONFIG.available.iconPath = MIC_PATH;
  STATE_CONFIG.active.iconPath = MIC_PATH;
  STATE_CONFIG.paused.iconPath = MIC_PATH;
  STATE_CONFIG.disconnected.iconPath = MIC_SLASH_PATH;
  STATE_CONFIG.denied.iconPath = MIC_SLASH_PATH;

  // ── DOM refs ─────────────────────────────────────────────────────────────────

  const widget = document.getElementById("mic-status-widget");
  const iconRing = document.getElementById("mic-status-icon-ring");
  const halo = document.getElementById("mic-status-halo");
  const iconSvg = document.getElementById("mic-status-icon-svg");
  const iconPath = document.getElementById("mic-status-icon-path");
  const labelEl = document.getElementById("mic-status-label");
  const detailEl = document.getElementById("mic-status-detail");
  const actionBtn = document.getElementById("mic-status-action-btn");

  // Guard: if widget is not in the current page, exit silently.
  if (!widget) return;

  // ── State ────────────────────────────────────────────────────────────────────

  /** @type {MicState} */
  let _current = "available";

  /** @type {MediaStream|null} – held only during the permission probe; released immediately */
  let _probeStream = null;

  // ── Renderer ─────────────────────────────────────────────────────────────────

  /**
   * Apply all visual changes for the given state.
   * Pure DOM mutations; no innerHTML to avoid XSS vectors.
   *
   * @param {MicState} state
   */
  function _render(state) {
    const cfg = STATE_CONFIG[state];
    if (!cfg) return;

    // Widget border
    _swapClasses(widget, /border-\S+/, cfg.border);

    // Icon ring
    iconRing.className =
      "relative flex-shrink-0 w-10 h-10 rounded-full flex items-center " +
      "justify-center transition-all duration-500 " +
      cfg.ring;

    // Halo ping (only visible when Active)
    if (cfg.halo) {
      halo.className =
        "absolute inset-0 rounded-full opacity-60 animate-ping " + cfg.halo;
    } else {
      halo.className =
        "absolute inset-0 rounded-full opacity-0 transition-opacity duration-500 animate-ping";
    }

    // Icon color
    _swapClasses(iconSvg, /text-\S+/, cfg.icon);

    // Icon path (mic vs mic-slash)
    iconPath.setAttribute("d", cfg.iconPath);

    // Label text + color
    labelEl.textContent = cfg.label;
    _swapClasses(labelEl, /text-\S+/, cfg.labelColor);

    // Detail sub-text
    detailEl.textContent = cfg.detail;

    // Action button visibility
    if (cfg.showBtn) {
      actionBtn.classList.remove("hidden");
    } else {
      actionBtn.classList.add("hidden");
    }
  }

  /**
   * Replace the first class on `el` that matches `pattern` with `replacement`.
   * If no match found, appends `replacement`.
   *
   * @param {Element} el
   * @param {RegExp}  pattern
   * @param {string}  replacement
   */
  function _swapClasses(el, pattern, replacement) {
    const classes = Array.from(el.classList);
    const idx = classes.findIndex((c) => pattern.test(c));
    if (idx !== -1) {
      el.classList.remove(classes[idx]);
    }
    el.classList.add(replacement);
  }

  // ── Public API ────────────────────────────────────────────────────────────────

  /**
   * Transition the widget to a new state.
   * Called internally and by FE-8b to push "active" / "paused".
   *
   * @param {MicState} state
   */
  function set(state) {
    if (!(state in STATE_CONFIG)) return;
    _current = state;
    _render(state);
    widget.dispatchEvent(
      new CustomEvent("micstatechange", { detail: { state }, bubbles: true })
    );
  }

  /** Return the current state string. */
  function get() {
    return _current;
  }

  window.MicStatus = { set, get };

  // ── Permission probing ────────────────────────────────────────────────────────

  /**
   * Translate a PermissionState string to our MicState.
   * "prompt" maps to available because we have not asked yet.
   *
   * @param {PermissionState} ps
   * @returns {MicState}
   */
  function _permissionStateToMicState(ps) {
    if (ps === "granted") return "available";
    if (ps === "denied") return "denied";
    return "available"; // "prompt" — show ready until user is asked
  }

  /**
   * Use getUserMedia to determine the actual permission state.
   * Immediately stops the probe stream after getting the answer so
   * no audio is captured here (capture is FE-8b's responsibility).
   *
   * Sets "disconnected" when no audio input device is present at all.
   *
   * @returns {Promise<void>}
   */
  async function _probePermission() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasAudioInput = devices.some((d) => d.kind === "audioinput");
      if (!hasAudioInput) {
        set("disconnected");
        return;
      }

      _probeStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Release immediately — actual capture starts in FE-8b
      _probeStream.getTracks().forEach((t) => t.stop());
      _probeStream = null;

      set("available");
    } catch (err) {
      _probeStream = null;

      if (
        err.name === "NotAllowedError" ||
        err.name === "PermissionDeniedError"
      ) {
        set("denied");
      } else if (
        err.name === "NotFoundError" ||
        err.name === "DevicesNotFoundError"
      ) {
        set("disconnected");
      } else {
        // Covers NotReadableError, OverconstrainedError, etc.
        set("disconnected");
      }
    }
  }

  /**
   * Subscribe to Permissions API change events where supported (Chromium / Edge).
   * Falls back to polling devicechange for Firefox / Safari.
   */
  async function _watchPermission() {
    if (!navigator.permissions) return;

    let status;
    try {
      status = await navigator.permissions.query({ name: "microphone" });
    } catch {
      return; // Browser doesn't support microphone permission query
    }

    set(_permissionStateToMicState(status.state));

    status.onchange = () => {
      const next = _permissionStateToMicState(status.state);
      // Only pull back to "available"; Active/Paused are managed by FE-8b
      if (_current !== "active" && _current !== "paused") {
        set(next);
      }
    };
  }

  // ── Device-change listener ────────────────────────────────────────────────────

  /**
   * Re-enumerate devices when a hardware change occurs (plug / unplug).
   * Transitions to "disconnected" if the mic is removed while not Active.
   */
  async function _handleDeviceChange() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const hasAudioInput = devices.some((d) => d.kind === "audioinput");

    if (!hasAudioInput && _current !== "active" && _current !== "paused") {
      set("disconnected");
    } else if (hasAudioInput && _current === "disconnected") {
      set("available");
    }
  }

  // ── Action button ─────────────────────────────────────────────────────────────

  actionBtn.addEventListener("click", () => {
    // Retry: probe permission again (used when device is reconnected)
    set("available"); // optimistic reset so UI feels responsive
    _probePermission();
  });

  // ── Bootstrap ────────────────────────────────────────────────────────────────

  (async () => {
    // Show a neutral "Checking…" render before any async resolves
    _render("available");

    // Try the Permissions API first (instant, no prompt)
    await _watchPermission();

    // Then probe via getUserMedia to get ground truth
    await _probePermission();

    // Watch for hardware changes
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener(
        "devicechange",
        _handleDeviceChange
      );
    }
  })();
})();
