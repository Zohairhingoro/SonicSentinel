/**
 * live_monitor.js — Live Monitor Dashboard Controller
 * SonicSentinel AI / AcousticX Intelligence
 *
 * Responsibilities:
 *  - Intercepts gtmPredictionCaptured events from gtm_infer.js to render GTM
 *    results instantly (client-side, no round trip).
 *  - Sends each captured audio window to the Python ML service (/predict) via
 *    multipart POST and renders the server prediction.
 *  - Polls /live/api/session-status/ every 5 s to synchronise the latest
 *    persisted event's comparison result, severity, and active alert panel.
 *  - Updates every live panel: current sound, Python result, GTM result,
 *    model agreement, severity, active alert banner.
 *
 * Architecture note: GTM result comes from gtm_infer.js client-side only.
 * Python result comes from the FastAPI ml_service. The two pipelines are
 * intentionally independent — this file never passes one model's output
 * into the other.
 */

(() => {
  "use strict";

  // ── Config ────────────────────────────────────────────────────────────────
  const ML_SERVICE_PREDICT_URL = "/ml/predict/";
  const SESSION_STATUS_API_URL = "/live/api/session-status/";
  const POLL_INTERVAL_MS = 5000;
  const WAVEFORM_BARS = 40;

  // ── Severity palette ──────────────────────────────────────────────────────
  const SEVERITY_STYLES = {
    Critical: {
      badge: "bg-rose-500/20 text-rose-300 border-rose-500/40",
      glow: "shadow-rose-500/30",
      dot: "bg-rose-500",
    },
    High: {
      badge: "bg-amber-500/20 text-amber-300 border-amber-500/40",
      glow: "shadow-amber-500/20",
      dot: "bg-amber-500",
    },
    Medium: {
      badge: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
      glow: "shadow-yellow-500/20",
      dot: "bg-yellow-500",
    },
    Low: {
      badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
      glow: "shadow-cyan-500/20",
      dot: "bg-cyan-400",
    },
    Informational: {
      badge: "bg-slate-700/60 text-slate-300 border-slate-600",
      glow: "",
      dot: "bg-slate-400",
    },
  };

  const MATCH_STYLES = {
    "Acceptable Match": {
      badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
      icon: "✓",
    },
    "Weak Match": {
      badge: "bg-amber-500/20 text-amber-300 border-amber-500/40",
      icon: "~",
    },
    "Model Disagreement": {
      badge: "bg-rose-500/20 text-rose-300 border-rose-500/40",
      icon: "✕",
    },
    "Uncertain Result": {
      badge: "bg-purple-500/20 text-purple-300 border-purple-500/40",
      icon: "?",
    },
  };

  // ── Helpers ───────────────────────────────────────────────────────────────
  function getCsrfToken() {
    for (const cookie of document.cookie.split(";")) {
      const c = cookie.trim();
      if (c.startsWith("csrftoken=")) return decodeURIComponent(c.substring(10));
    }
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  function el(id) {
    return document.getElementById(id);
  }

  function setText(id, text) {
    const node = el(id);
    if (node) node.textContent = text;
  }

  function setHtml(id, html) {
    const node = el(id);
    if (node) node.innerHTML = html;
  }

  function severityStyles(severity) {
    return SEVERITY_STYLES[severity] || SEVERITY_STYLES["Informational"];
  }

  function matchStyles(matchStatus) {
    return MATCH_STYLES[matchStatus] || MATCH_STYLES["Uncertain Result"];
  }

  // ── Waveform animator ─────────────────────────────────────────────────────
  function animateWaveformBars(active) {
    const container = el("lm-waveform-bars");
    if (!container) return;
    const bars = container.querySelectorAll(".lm-bar");
    bars.forEach((bar, i) => {
      if (active) {
        const delay = (i * 40) % 400;
        bar.style.animationDelay = `${delay}ms`;
        bar.classList.add("animate-pulse");
        const heights = [20, 60, 80, 40, 90, 50, 70, 30, 85, 45];
        bar.style.height = `${heights[i % heights.length]}%`;
      } else {
        bar.classList.remove("animate-pulse");
        bar.style.height = "20%";
      }
    });
  }

  // ── Panel updaters ────────────────────────────────────────────────────────

  function updateCurrentSoundPanel(event) {
    if (!event || !event.audio_id) {
      setText("lm-current-category", "Awaiting detection…");
      setText("lm-current-quality", "—");
      setText("lm-current-audio-id", "—");
      setText("lm-current-status", "—");
      updateSeverityPanel(null);
      return;
    }

    setText("lm-current-category", event.category || "Unknown");
    setText("lm-current-quality", event.quality || "—");
    setText("lm-current-audio-id", event.audio_id);
    setText("lm-current-status", event.status || "—");
    updateSeverityPanel(event.severity);

    const categoryIcon = el("lm-category-icon");
    if (categoryIcon) {
      categoryIcon.textContent = getCategoryEmoji(event.category);
    }
  }

  function updateSeverityPanel(severity) {
    const badge = el("lm-severity-badge");
    if (!badge) return;
    const styles = severityStyles(severity || "Informational");
    badge.textContent = severity || "—";
    badge.className = `inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold border shadow-lg ${styles.badge}`;

    const dotEl = el("lm-severity-dot");
    if (dotEl) {
      dotEl.className = `w-2.5 h-2.5 rounded-full animate-ping ${styles.dot}`;
    }

    const cardEl = el("lm-severity-card");
    if (cardEl) {
      const isCritical = severity === "Critical";
      const isHigh = severity === "High";
      if (isCritical) {
        cardEl.classList.add("border-rose-500/60", "shadow-rose-500/20");
        cardEl.classList.remove("border-slate-800", "border-amber-500/40");
      } else if (isHigh) {
        cardEl.classList.add("border-amber-500/40", "shadow-amber-500/10");
        cardEl.classList.remove("border-slate-800", "border-rose-500/60");
      } else {
        cardEl.classList.remove("border-rose-500/60", "border-amber-500/40");
        cardEl.classList.add("border-slate-800");
      }
    }
  }

  function updatePythonPanel(pyData) {
    if (!pyData || !pyData.predicted_class) {
      setText("lm-py-class", "Awaiting window…");
      setText("lm-py-confidence", "—");
      setHtml("lm-py-top3", buildEmptyTop3("py"));
      return;
    }

    setText("lm-py-class", pyData.predicted_class);
    const conf = pyData.confidence != null ? `${(pyData.confidence * 100).toFixed(1)}%` : "—";
    setText("lm-py-confidence", conf);
    setText("lm-py-version", pyData.model_version || "—");

    if (pyData.top3 && pyData.top3.length) {
      setHtml("lm-py-top3", buildTop3Html(pyData.top3, "from-cyan-500 to-blue-500"));
    }
  }

  function updateGTMPanel(gtmData) {
    if (!gtmData || !gtmData.predicted_class) {
      setText("lm-gtm-class", "Awaiting window…");
      setText("lm-gtm-confidence", "—");
      setHtml("lm-gtm-top3", buildEmptyTop3("gtm"));
      return;
    }

    setText("lm-gtm-class", gtmData.predicted_class);
    const scores = gtmData.confidence_scores || {};
    const topScore = scores[gtmData.predicted_class];
    const conf = topScore != null ? `${(topScore * 100).toFixed(1)}%` : "—";
    setText("lm-gtm-confidence", conf);
    setText("lm-gtm-version", gtmData.model_version || "TF.js");

    const top3 = buildTop3FromScores(gtmData.confidence_scores);
    if (top3.length) {
      setHtml("lm-gtm-top3", buildTop3Html(top3, "from-indigo-500 to-violet-500"));
    }
  }

  function updateAgreementPanel(compData) {
    const badge = el("lm-agreement-badge");
    if (!badge) return;

    if (!compData || !compData.match_status) {
      badge.textContent = "Awaiting comparison…";
      badge.className = "inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold border bg-slate-700/50 text-slate-300 border-slate-600";
      setText("lm-conf-diff", "—");
      setText("lm-top-two-margin", "—");
      setHtml("lm-uncertain-flags", "");
      return;
    }

    const styles = matchStyles(compData.match_status);
    badge.innerHTML = `<span>${styles.icon}</span><span>${compData.match_status}</span>`;
    badge.className = `inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold border ${styles.badge}`;

    const confDiff = compData.confidence_difference != null
      ? compData.confidence_difference.toFixed(4)
      : "—";
    setText("lm-conf-diff", confDiff);

    const margin = compData.top_two_margin != null
      ? compData.top_two_margin.toFixed(4)
      : "—";
    setText("lm-top-two-margin", margin);

    let flags = "";
    if (compData.is_uncertain) {
      flags += `<span class="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">Uncertain</span>`;
    }
    if (compData.is_overlapping) {
      flags += `<span class="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20">Overlapping</span>`;
    }
    setHtml("lm-uncertain-flags", flags);
  }

  function updateActiveAlertsBanner(alerts, count) {
    const banner = el("lm-alert-banner");
    const noAlertEl = el("lm-no-alert");
    const alertListEl = el("lm-alert-list");
    const countEl = el("lm-alert-count");

    if (countEl) countEl.textContent = count || 0;

    if (!alerts || alerts.length === 0) {
      if (banner) banner.classList.add("hidden");
      if (noAlertEl) noAlertEl.classList.remove("hidden");
      if (alertListEl) alertListEl.classList.add("hidden");
      return;
    }

    if (noAlertEl) noAlertEl.classList.add("hidden");
    if (alertListEl) alertListEl.classList.remove("hidden");
    if (banner) {
      banner.classList.remove("hidden");
      const topAlert = alerts[0];
      const isCritical = topAlert.severity === "Critical" || topAlert.severity === "High";
      banner.className = `flex items-center gap-3 px-4 py-3 rounded-2xl border text-sm font-semibold transition-all ${isCritical
          ? "bg-rose-950/80 border-rose-500/60 text-rose-200 shadow-lg shadow-rose-950/50"
          : "bg-amber-950/60 border-amber-500/40 text-amber-200"
        }`;
      const bannerText = el("lm-alert-banner-text");
      if (bannerText) {
        bannerText.textContent = `${topAlert.severity}: ${topAlert.category} detected — ${topAlert.time_ago}`;
      }
    }

    if (alertListEl) {
      alertListEl.innerHTML = alerts
        .map((a) => {
          const s = severityStyles(a.severity);
          return `
            <div class="flex items-center justify-between p-3 rounded-xl bg-slate-950/70 border border-slate-800 gap-3">
              <div class="flex items-center gap-2 min-w-0">
                <span class="w-2 h-2 rounded-full flex-shrink-0 ${s.dot}"></span>
                <span class="text-xs font-semibold text-slate-200 truncate">${a.category}</span>
                <span class="px-1.5 py-0.5 rounded-full text-[10px] font-bold border ${s.badge}">${a.severity}</span>
              </div>
              <div class="flex items-center gap-2 flex-shrink-0 text-[11px] text-slate-400">
                <span class="font-mono text-cyan-400">${a.audio_id}</span>
                <span>${a.time_ago}</span>
                <a href="/alerts/" class="px-2 py-0.5 rounded-lg text-[10px] font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 transition-all">View</a>
              </div>
            </div>
          `;
        })
        .join("");
    }
  }

  // ── Top-3 HTML builders ───────────────────────────────────────────────────

  function buildTop3Html(top3, gradientClasses) {
    return top3
      .map(
        (item) => `
        <div class="space-y-1">
          <div class="flex items-center justify-between text-xs">
            <span class="font-medium text-slate-300 truncate max-w-[160px]">${item.category}</span>
            <span class="font-mono text-slate-400 text-[10px]">${item.percentage}%</span>
          </div>
          <div class="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div class="h-full rounded-full bg-gradient-to-r ${gradientClasses} transition-all duration-500"
                 style="width:${item.percentage}%"></div>
          </div>
        </div>`
      )
      .join("");
  }

  function buildEmptyTop3(type) {
    return `<p class="text-xs text-slate-500 italic">No predictions yet.</p>`;
  }

  function buildTop3FromScores(scores) {
    if (!scores || typeof scores !== "object") return [];
    return Object.entries(scores)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([category, score]) => ({
        category,
        confidence: score,
        percentage: Math.round(score * 100),
      }));
  }

  function getCategoryEmoji(category) {
    const map = {
      "Gunshot": "🔫",
      "Glass Breaking": "💥",
      "Alarm or Siren": "🚨",
      "Panic Scream": "😱",
      "Aggression or Violent Conflict": "⚔️",
      "Person Asking for Help": "🆘",
      "Machinery Fault": "⚙️",
      "Vehicle Horn": "🚗",
      "Animal Sound": "🐾",
      "Background Noise": "🔇",
    };
    return map[category] || "🔊";
  }

  // ── Session status poll ───────────────────────────────────────────────────

  async function pollSessionStatus() {
    try {
      const resp = await fetch(SESSION_STATUS_API_URL, {
        headers: { Accept: "application/json" },
      });
      if (!resp.ok) return;
      const data = await resp.json();

      updateCurrentSoundPanel(data.event || null);
      updatePythonPanel(data.python_prediction || null);
      updateAgreementPanel(data.comparison || null);
      updateActiveAlertsBanner(data.active_alerts || [], data.active_alert_count || 0);

      updateLastPolledTime();
    } catch (err) {
      console.warn("[LiveMonitor] Session status poll failed:", err);
    }
  }

  function updateLastPolledTime() {
    const el = document.getElementById("lm-last-polled");
    if (el) {
      const now = new Date();
      el.textContent = now.toLocaleTimeString();
    }
  }

  // ── GTM real-time hook ────────────────────────────────────────────────────
  // Intercept gtmPredictionCaptured from gtm_infer.js; update GTM panel immediately
  // without waiting for the 5s poll cycle.

  function wireGTMHook() {
    window.addEventListener("gtmPredictionCaptured", (evt) => {
      const result = evt.detail;
      if (!result) return;

      const gtmPayload = {
        predicted_class: result.predicted_class,
        confidence_scores: result.confidence,
        model_version: "TF.js",
      };
      updateGTMPanel(gtmPayload);
      animateWaveformBars(true);
      setTimeout(() => animateWaveformBars(false), 1500);
    });
  }

  // ── Python ML service call ─────────────────────────────────────────────────
  // Called for each audio window blob. Posts to the proxy endpoint that
  // forwards to the FastAPI ml_service. Result updates the Python panel
  // immediately on response; full comparison is picked up by next poll.

  let lastAudioBlob = null;

  async function callPythonMLService(audioBlob) {
    if (audioBlob) lastAudioBlob = audioBlob;
    const blobToSend = audioBlob || lastAudioBlob;
    if (!blobToSend) return;

    const formData = new FormData();
    formData.append("audio", blobToSend, "window.webm");

    try {
      const resp = await fetch(ML_SERVICE_PREDICT_URL, {
        method: "POST",
        headers: { "X-CSRFToken": getCsrfToken() },
        body: formData,
      });

      if (!resp.ok) {
        showPythonErrorState("ML Service returned HTTP " + resp.status);
        return;
      }

      const data = await resp.json();
      if (data && data.predicted_class) {
        const top3 = Object.entries(data.confidence || {})
          .sort(([, a], [, b]) => b - a)
          .slice(0, 3)
          .map(([category, score]) => ({
            category,
            confidence: score,
            percentage: Math.round(score * 100),
          }));

        updatePythonPanel({
          predicted_class: data.predicted_class,
          confidence: data.confidence ? data.confidence[data.predicted_class] : null,
          confidence_scores: data.confidence || {},
          model_version: data.model_version || "—",
          top3,
        });
      }
    } catch (err) {
      console.warn("[LiveMonitor] Python ML service call failed:", err);
      showPythonErrorState("ML Service offline or unreachable");
    }
  }

  function showPythonErrorState(msg) {
    setText("lm-py-class", "Service Error");
    setText("lm-py-confidence", "—");
    setHtml("lm-py-top3", `
      <div class="p-3 rounded-2xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs space-y-2">
        <div class="flex items-center gap-2 font-medium">
          <svg class="w-4 h-4 text-rose-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <span>${msg}</span>
        </div>
        <button type="button" id="lm-retry-py-btn" class="btn btn-secondary text-[11px] py-1 px-3 w-full flex items-center justify-center gap-1.5">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          <span>Retry ML Inference</span>
        </button>
      </div>
    `);
    const retryBtn = document.getElementById("lm-retry-py-btn");
    if (retryBtn) {
      retryBtn.onclick = () => callPythonMLService(lastAudioBlob);
    }
  }

  // ── Audio window callback ─────────────────────────────────────────────────
  // Overrides the stub set by mic_capture.js to route each window to the
  // Python ML service in addition to the GTM client-side path.

  function wireAudioWindowHook() {
    const existingCallback = window.onAudioWindowCaptured;

    window.onAudioWindowCaptured = function (audioBlob) {
      if (
        typeof existingCallback === "function" &&
        existingCallback !== window.onAudioWindowCaptured
      ) {
        try {
          existingCallback(audioBlob);
        } catch (e) {
          console.warn("[LiveMonitor] Existing onAudioWindowCaptured error:", e);
        }
      }
      callPythonMLService(audioBlob);
    };
  }

  // ── Mic status change visual ──────────────────────────────────────────────

  function observeMicStatus() {
    const label = document.getElementById("mic-status-label");
    if (!label) return;

    const observer = new MutationObserver(() => {
      const status = label.textContent.trim().toLowerCase();
      const isActive = status === "active";
      animateWaveformBars(isActive);

      const indicator = document.getElementById("lm-mic-state-indicator");
      if (indicator) {
        indicator.textContent = isActive ? "LIVE" : status.toUpperCase();
        indicator.className = `text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full border ${isActive
            ? "bg-rose-500/20 text-rose-400 border-rose-500/40 animate-pulse"
            : "bg-slate-700/60 text-slate-400 border-slate-600"
          }`;
      }
    });

    observer.observe(label, { childList: true, subtree: true, characterData: true });
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  function init() {
    wireGTMHook();
    wireAudioWindowHook();
    observeMicStatus();
    pollSessionStatus();
    setInterval(pollSessionStatus, POLL_INTERVAL_MS);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
