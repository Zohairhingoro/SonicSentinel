/**
 * scroll-reveal.js — Unified Scroll Animation & Parallax Utility
 * SonicSentinel AI / AcousticX Intelligence
 *
 * Responsibilities:
 *  1. [data-reveal]: IntersectionObserver fade & slide-up animation when elements
 *     scroll into view for the first time. Supports optional [data-reveal-delay].
 *  2. [data-parallax]: Hero background subtle scroll parallax effect.
 *  3. [data-nav-shrink] / .nav-shell: Shrink & compact navbar on scroll.
 *
 * All motion values (durations, distance, easing, parallax speed) are driven by
 * tokens.css CSS custom properties.
 */

(() => {
  "use strict";

  // ── Configuration & Tokens ───────────────────────────────────────────────
  const DEFAULT_STAGGER_MS = 80;
  const NAV_SHRINK_THRESHOLD_PX = 40;

  let revealObserver = null;
  let parallaxElements = [];
  let navShell = null;
  let isTicking = false;

  // ── Helper: Read CSS Custom Property ─────────────────────────────────────
  function getCssToken(name, fallback) {
    const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return val || fallback;
  }

  // ── 1. IntersectionObserver for data-reveal ─────────────────────────────
  function initRevealObserver() {
    if (!("IntersectionObserver" in window)) {
      // Fallback for environment without IntersectionObserver
      document.querySelectorAll("[data-reveal]").forEach((el) => {
        el.classList.add("is-revealed");
      });
      return;
    }

    const options = {
      root: null,
      rootMargin: "0px 0px -50px 0px",
      threshold: 0.12,
    };

    revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const target = entry.target;
          const delayAttr = target.getAttribute("data-reveal-delay");

          if (delayAttr) {
            let delayStr = delayAttr;
            if (/^\d+$/.test(delayAttr)) {
              // Pure integer: if small (<= 10), treat as index stagger step; else ms
              const val = parseInt(delayAttr, 10);
              delayStr = val <= 10 ? `${val * DEFAULT_STAGGER_MS}ms` : `${val}ms`;
            } else if (!delayAttr.endsWith("ms") && !delayAttr.endsWith("s")) {
              delayStr = `${delayAttr}ms`;
            }
            target.style.transitionDelay = delayStr;
          }

          target.classList.add("is-revealed");
          observer.unobserve(target);
        }
      });
    }, options);

    observeRevealElements();
  }

  function observeRevealElements() {
    if (!revealObserver) return;
    const elements = document.querySelectorAll("[data-reveal]:not(.is-revealed)");
    elements.forEach((el) => revealObserver.observe(el));
  }

  // ── 2. Hero Background Parallax Effect ──────────────────────────────────
  function initParallax() {
    parallaxElements = Array.from(document.querySelectorAll("[data-parallax], .hero-parallax"));
  }

  function updateParallax() {
    if (!parallaxElements.length) return;
    const scrollY = window.scrollY || window.pageYOffset;
    const tokenSpeed = parseFloat(getCssToken("--parallax-speed", "0.25"));

    parallaxElements.forEach((el) => {
      const speed = parseFloat(el.getAttribute("data-parallax-speed")) || tokenSpeed;
      const offset = scrollY * speed;
      el.style.transform = `translate3d(0, ${offset.toFixed(2)}px, 0)`;
    });
  }

  // ── 3. Shrink-on-Scroll Nav Bar ──────────────────────────────────────────
  function initNavShrink() {
    navShell = document.querySelector(".nav-shell, [data-nav-shrink]");
  }

  function updateNavShrink() {
    if (!navShell) return;
    const scrollY = window.scrollY || window.pageYOffset;

    if (scrollY > NAV_SHRINK_THRESHOLD_PX) {
      if (!navShell.classList.contains("nav-shrunk")) {
        navShell.classList.add("nav-shrunk");
      }
    } else {
      if (navShell.classList.contains("nav-shrunk")) {
        navShell.classList.remove("nav-shrunk");
      }
    }
  }

  // ── Scroll Event RAF Loop ────────────────────────────────────────────────
  function onScroll() {
    if (!isTicking) {
      window.requestAnimationFrame(() => {
        updateParallax();
        updateNavShrink();
        isTicking = false;
      });
      isTicking = true;
    }
  }

  // ── Public Global API ────────────────────────────────────────────────────
  window.SonicSentinelReveal = {
    refresh() {
      observeRevealElements();
      initParallax();
      updateParallax();
      updateNavShrink();
    },
    init() {
      initRevealObserver();
      initParallax();
      initNavShrink();
      updateParallax();
      updateNavShrink();
      window.addEventListener("scroll", onScroll, { passive: true });
    },
  };

  // ── Auto Init on DOMReady ────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => window.SonicSentinelReveal.init());
  } else {
    window.SonicSentinelReveal.init();
  }
})();
