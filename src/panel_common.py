"""
panel_common.py – shared plumbing for the waveform and spectrogram panels
===========================================================================
Both result panels are the same kind of thing: a control/readout bar above
one dark, horizontally-scrollable card of stacked tracks, drawn in the
browser inside its own st.iframe so the View controls (track height, width
in px per second, scrollable vs fit-to-width) apply instantly, without a
Streamlit rerun.

This module holds what they share:
    - layout constants (track header / gap / label strip sizes, zoom range);
    - PANEL_CSS        – styling for the bar, the card and the tracks;
    - PANEL_COMMON_JS  – the `PVPanel` helper both panel scripts use
                         (View controls + persistence, fit-to-width zoom,
                         timestamp spacing, iframe sizing);
    - view_controls_html() / scroll_card_html() – shared HTML pieces;
    - render_panel()   – assembles a panel page and embeds it.
"""

import json

import streamlit as st

# Shared color tokens — the dark "panel" palette sidebar.py and app.py also
# draw from (and .streamlit/config.toml mirrors for the page theme), so the
# sidebar, the result panels and the main page read as one surface instead
# of drifting apart with their own hardcoded hex values.
COLOR_BG = "#0b0f14"  # page / outer surface
COLOR_SURFACE = "#111827"  # raised surface: cards, the readout bar
COLOR_BORDER = "#1f2937"
COLOR_BORDER_LIGHT = "#374151"  # hover / secondary border
COLOR_BORDER_HOVER = "#4b5563"
COLOR_TEXT = "#d1d5db"  # body text
COLOR_TEXT_BRIGHT = "#e5e7eb"  # emphasis: buttons, numeric output
COLOR_TEXT_BRIGHTEST = "#f9fafb"
COLOR_TEXT_DIM = "#9ca3af"  # labels, captions
COLOR_TEXT_FAINT = "#6b7280"  # hints, disabled-ish text
COLOR_ACCENT = "#ef4444"  # sliders / checkboxes (matches theme primaryColor)

# Track layout, shared so both panels line up the same way.
TRACK_TITLE_PX = 26  # sticky header row above each track
TRACK_GAP_PX = 14  # space between tracks
TRACK_LABEL_PX = 16  # strip below each track for the timestamp labels
PLACEHOLDER_WIDTH_PX = 100  # track width before the panel script draws it

# Horizontal zoom range of the Width control, in pixels per second of audio.
DEFAULT_PX_PER_SECOND = 120
MIN_PX_PER_SECOND = 20
MAX_PX_PER_SECOND = 600

PANEL_CSS = r"""
/* ---- Scrolling card ---------------------------------------------------- */
/* No default body margin, and an explicit, visible scrollbar — the default
   one all but disappears against the dark cards. */
body { margin: 0; }
.pv-scroll { overflow-x: auto; overflow-y: hidden; }
.pv-scroll > svg, .pv-scroll > img { display: block; }
.pv-scroll::-webkit-scrollbar { height: 10px; }
.pv-scroll::-webkit-scrollbar-thumb { background: #9ca3af; border-radius: 5px; }
.pv-scroll::-webkit-scrollbar-track { background: #e5e7eb; border-radius: 5px; }
.pv-scroll { scrollbar-color: #9ca3af #e5e7eb; }
.pv-scroll-dark::-webkit-scrollbar-thumb { background: #4b5563; }
.pv-scroll-dark::-webkit-scrollbar-track { background: #111827; }
.pv-scroll-dark { scrollbar-color: #4b5563 #111827; }

/* ---- Control / readout bar above the card ------------------------------- */
#pv-readout { font-size: 0.8rem; color: #d1d5db; background: #111827;
              border: 1px solid #1f2937; border-radius: 10px; padding: 8px 10px;
              margin-bottom: 8px; box-sizing: border-box; overflow: hidden; }
#pv-readout .pv-bar { display: flex; gap: 14px; align-items: center; margin-bottom: 6px;
                      color: #9ca3af; flex-wrap: wrap; }
#pv-readout .pv-view { padding-top: 6px; border-top: 1px solid #1f2937; }
#pv-readout .pv-view label { display: inline-flex; align-items: center; gap: 6px; }
#pv-readout .pv-view input[type=range] { width: 130px; accent-color: #ef4444; }
#pv-readout .pv-view input[type=range]:disabled { opacity: 0.4; }
#pv-readout .pv-view input[type=checkbox] { accent-color: #ef4444; }
#pv-readout .pv-out { color: #e5e7eb; min-width: 64px; font-variant-numeric: tabular-nums; }
#pv-readout .pv-cap { color: #fbbf24; }
#pv-readout .pv-pin { color: #e5e7eb; font-weight: 600; }
#pv-readout .pv-play { color: #fbbf24; font-weight: 600; font-variant-numeric: tabular-nums; }
#pv-readout button { background: #1f2937; color: #e5e7eb; border: 1px solid #374151;
                     border-radius: 6px; padding: 1px 8px; cursor: pointer; font-size: 0.75rem; }
#pv-readout table { border-collapse: collapse; width: 100%; }
#pv-readout th { text-align: left; font-weight: 600; color: #9ca3af; padding: 2px 8px 2px 0; }
#pv-readout td { padding: 2px 8px 2px 0; white-space: nowrap; font-variant-numeric: tabular-nums; }
#pv-readout .pv-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                     margin-right: 6px; vertical-align: middle; }
#pv-readout .pv-dim, #pv-readout .pv-hint { color: #6b7280; }

/* ---- Tracks -------------------------------------------------------------- */
/* block, so a track narrower than the card (fit mode, shorter clip)
   still sits below its title instead of beside it */
svg.pv-wave { cursor: crosshair; display: block; }
/* A track row is as wide as its plot, not just the card: a sticky
   element can't travel past its container, so with a card-wide row the
   header scrolled away once you scrolled past the first screen. */
.pv-track { width: max-content; min-width: 100%; }
/* Track header: (play/pause +) title (+ time). Sticky so it stays visible
   at the left edge while the tracks scroll. */
.pv-track-head { position: sticky; left: 0; display: inline-flex; align-items: center;
                 gap: 8px; font-size: 0.85rem; color: #9ca3af; }
.pv-ttl { font-weight: 600; }
.pv-ttime { color: #6b7280; font-variant-numeric: tabular-nums; }
.pv-tbtn { width: 24px; height: 22px; padding: 0; border-radius: 6px; cursor: pointer;
           background: #1f2937; color: #e5e7eb; border: 1px solid #374151; font-size: 0.7rem;
           line-height: 1; }
.pv-tbtn:hover:not(:disabled) { background: #374151; }
.pv-tbtn.pv-on { background: #fbbf24; color: #111827; border-color: #fbbf24; }
.pv-tbtn:disabled { opacity: 0.4; cursor: default; }
"""

PANEL_COMMON_JS = r"""
// Shared by the waveform and spectrogram panels. Each panel runs in its own
// st.iframe document; render_panel() inlines this before the panel's own
// script. Covers the View controls (height / width / scrollable), their
// per-viewer persistence, fit-to-width zoom, timestamp spacing and sizing
// the iframe to its content.
const PVPanel = (function () {
  const clamp = (v, lo, hi, dflt) => (isFinite(+v) ? Math.min(hi, Math.max(lo, +v)) : dflt);

  // View settings remembered per viewer in localStorage (per panel `key`),
  // clamped to the current limits in case those changed since they were saved.
  function loadView(key, defaults, limits) {
    const view = { height: defaults.height, pps: defaults.pps, scroll: true };
    try { Object.assign(view, JSON.parse(localStorage.getItem(key) || "{}")); } catch (e) {}
    view.height = clamp(view.height, limits.hMin, limits.hMax, defaults.height);
    view.pps = clamp(view.pps, limits.wMin, limits.wMax, defaults.pps);
    view.scroll = view.scroll !== false;
    return view;
  }

  function saveView(key, view) {
    try { localStorage.setItem(key, JSON.stringify(view)); } catch (e) {}
  }

  // Smallest "nice" timestamp step that keeps labels >= ~80 px apart.
  function tickStep(pps) {
    for (const s of [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60]) if (s * pps >= 80) return s;
    return 120;
  }

  // text-anchor for the k-th timestamp label at x = xv in a W-px-wide track:
  // edge labels are anchored inward so "0s" / the last label don't clip.
  function tickAnchor(k, xv, W) {
    return k === 0 ? "start" : xv > W - 16 ? "end" : "middle";
  }

  // Size this iframe to fit its content. Streamlit's own content sizing
  // (st.iframe height="content") measures max(content, viewport), so it
  // can grow the frame but never shrink it — setting the frame's height
  // here first lets it shrink too, and Streamlit then re-measures and
  // resizes the element's layout slot to match.
  function fitFrame(root) {
    try {
      const fe = window.frameElement;
      if (!fe) return;
      const h = Math.ceil(root.getBoundingClientRect().height) + 4;
      fe.style.height = h + "px";
      fe.setAttribute("height", h);
    } catch (e) {}
  }

  // Wire a panel's View controls (.pv-h / .pv-w / .pv-s and their outputs
  // inside `o.panel`) to a view state, and return { view, render }.
  //   o.key, o.defaults, o.limits  — persistence key, defaults, slider ranges
  //   o.panel, o.scroller, o.root  — controls container, scrolling card, iframe root
  //   o.maxDuration                — longest track (s), used by fit mode
  //   o.onRender(pps)              — draws the tracks at `pps` px/s; may return
  //                                  a note to show next to the controls
  function bindView(o) {
    const L = o.limits;
    const view = loadView(o.key, o.defaults, L);
    const q = sel => o.panel.querySelector(sel);
    const hIn = q(".pv-h"), hOut = q(".pv-h-out");
    const wIn = q(".pv-w"), wOut = q(".pv-w-out");
    const sIn = q(".pv-s"), capNote = q(".pv-cap");

    function render() {
      // Fit mode: the longest clip fills the card; all tracks keep one px/s.
      const avail = Math.max(100, o.scroller.clientWidth - 16);
      const pps = view.scroll ? view.pps : avail / Math.max(1e-9, o.maxDuration);
      o.scroller.style.overflowX = view.scroll ? "auto" : "hidden";
      if (!view.scroll) o.scroller.scrollLeft = 0;

      const note = o.onRender(pps) || "";

      hOut.textContent = view.height + " px";
      wIn.disabled = !view.scroll;
      wOut.textContent = (pps < 10 ? pps.toFixed(1) : Math.round(pps)) + " px/s" + (view.scroll ? "" : " (fit)");
      capNote.textContent = note;
      fitFrame(o.root);
    }

    hIn.min = L.hMin; hIn.max = L.hMax; hIn.step = L.hStep; hIn.value = view.height;
    wIn.min = L.wMin; wIn.max = L.wMax; wIn.step = L.wStep; wIn.value = view.pps;
    sIn.checked = view.scroll;
    const changed = () => { saveView(o.key, view); render(); };
    hIn.addEventListener("input", () => { view.height = +hIn.value; changed(); });
    wIn.addEventListener("input", () => { view.pps = +wIn.value; changed(); });
    sIn.addEventListener("change", () => { view.scroll = sIn.checked; changed(); });
    window.addEventListener("resize", () => { if (!view.scroll) render(); });

    return { view, render };
  }

  return { tickStep, tickAnchor, fitFrame, bindView };
})();
"""


def view_limits(height_min: int, height_max: int) -> dict:
    """Slider ranges for a panel's View controls (read by PVPanel.bindView)."""
    return {
        "hMin": height_min, "hMax": height_max, "hStep": 20,
        "wMin": MIN_PX_PER_SECOND, "wMax": MAX_PX_PER_SECOND, "wStep": 10,
    }


def view_controls_html(height_tip: str, width_tip: str) -> str:
    """The Height / Width / Scrollable controls PVPanel.bindView wires up."""
    return f"""
      <label title="{height_tip}">Height
        <input class="pv-h" type="range"> <span class="pv-out pv-h-out"></span></label>
      <label title="{width_tip}">Width
        <input class="pv-w" type="range"> <span class="pv-out pv-w-out"></span></label>
      <label title="Off = fit the whole clip to the card width (no horizontal scrolling)">
        <input class="pv-s" type="checkbox"> Scrollable</label>
    """


def scroll_card_html(rows_html: str) -> str:
    """The dark card the tracks are stacked in, with one shared scrollbar."""
    return f"""
      <div class="pv-scroll pv-scroll-dark"
           style="border:1px solid #1f2937; border-radius:10px; background:#0b0f14;
                  padding:8px 8px 0 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        {rows_html}
      </div>
    """


def render_panel(body_html: str, data: dict, panel_js: str, extra_css: str = "") -> None:
    """Assemble a panel page and embed it.

    `body_html` goes inside #pv-root; `data` is exposed to the scripts as the
    global PV_DATA; then PANEL_COMMON_JS and the panel's own `panel_js` run.

    st.iframe(height="content") — not components.html — so Streamlit keeps
    the element's layout slot in sync with the frame's content height (with
    components.html the slot stayed at its render-time height, and a taller
    track height made the frame overlap the sections below it). The iframe
    is same-origin, which the waveform panel relies on to drive the audio
    players in the parent page.
    """
    data_json = json.dumps(data).replace("</", "<\\/")
    page = f"""
    <style>{PANEL_CSS}{extra_css}</style>
    <div id="pv-root" style="font-family: -apple-system, Segoe UI, Roboto, sans-serif;">
      {body_html}
    </div>
    <script>const PV_DATA = {data_json};</script>
    <script>{PANEL_COMMON_JS}</script>
    <script>{panel_js}</script>
    """
    st.iframe(page, height="content")
