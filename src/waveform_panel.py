"""
waveform_panel.py – the Waveforms results panel
=================================================
Stacks the result clips' waveforms in one dark, scrollable card (see
panel_common.py) and makes it interactive in the browser:

    - View controls: track height, width (px per second), scrollable vs fit;
    - measurement cursor: hover/click to read each track's value at the same
      moment, and its difference from the Original;
    - playback: per-track play/pause, double-click to seek, and a playhead
      synced across all tracks. The audio itself is played by st.audio
      players rendered hidden by render_players(); the panel drives them.

Python ships each track's min/max/RMS envelope once, at the finest zoom;
the browser (WAVEFORM_JS) derives every coarser zoom and draws it.
"""

import base64
import html as _html

import numpy as np
import streamlit as st

from panel_common import (
    COLOR_TEXT,
    DEFAULT_PX_PER_SECOND,
    MAX_PX_PER_SECOND,
    PLACEHOLDER_WIDTH_PX,
    TRACK_GAP_PX,
    TRACK_LABEL_PX,
    TRACK_TITLE_PX,
    render_panel,
    scroll_card_html,
    view_controls_html,
    view_limits,
)

WAVEFORM_HEIGHT_PX = 160  # default track height
WAVEFORM_FILL = 0.92  # fraction of the half-height the shared peak reaches

# The base envelope is computed once at the finest zoom the Width control
# allows (MAX_PX_PER_SECOND). WAVEFORM_MAX_COLS caps it so the page stays a
# sane size on long clips (a 4:45 clip at 600 px/s would be 171k columns);
# past the cap, zooming in further adds no detail.
WAVEFORM_MAX_COLS = 40000

WAVEFORM_READOUT_ROW_PX = 24  # one row of the cursor readout table

# st.container key holding the hidden st.audio players the panel drives.
PLAYERS_KEY = "pv_players"


def render_legend(items: list[tuple[str, str]]) -> None:
    """A row of colored swatches + labels, e.g. above the waveform panel."""
    swatches = "".join(
        f'<span style="display:inline-flex; align-items:center; margin-right:18px;">'
        f'<span style="width:12px; height:12px; border-radius:3px; background:{color}; '
        f'display:inline-block; margin-right:6px;"></span>'
        f'<span style="font-size:0.85rem; color:{COLOR_TEXT};">{_html.escape(label)}</span></span>'
        for label, color in items
    )
    st.markdown(f'<div style="margin:0.1rem 0 0.9rem 0;">{swatches}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=16)
def _waveform_envelope(samples: np.ndarray, sample_rate: int) -> dict:
    """Per-pixel-column summary of a clip at the FINEST zoom the View
    controls allow (MAX_PX_PER_SECOND), capped at WAVEFORM_MAX_COLS columns.
    The browser derives every coarser zoom from this by merging columns, so
    changing the view never needs Python.

    One column covers many samples (e.g. ~74 at 44.1 kHz / 600 px/s), so
    each column stores:
      - col_max / col_min: highest / lowest sample (drawn as the envelope bar;
        the cursor's signed peak is whichever has the larger magnitude)
      - col_rms: RMS level over the column (the column's average loudness)

    Cached so reruns don't recompute it for the same clip.
    """
    n = len(samples)
    if n == 0 or sample_rate <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"duration": 0.0, "col_max": empty, "col_min": empty, "col_rms": empty}

    duration = n / sample_rate
    cols = int(max(1, min(WAVEFORM_MAX_COLS, round(duration * MAX_PX_PER_SECOND), n)))

    starts = np.minimum(np.linspace(0, n, num=cols, endpoint=False).astype(np.int64), n - 1)
    col_max = np.maximum.reduceat(samples, starts)
    col_min = np.minimum.reduceat(samples, starts)

    counts = np.maximum(np.diff(np.append(starts, n)), 1)
    sq_sums = np.add.reduceat(np.square(samples, dtype=np.float32), starts, dtype=np.float64)
    col_rms = np.sqrt(sq_sums / counts).astype(np.float32)

    return {"duration": duration, "col_max": col_max, "col_min": col_min, "col_rms": col_rms}


def _b64_int16(values: np.ndarray, full_scale: float) -> str:
    """Quantize to little-endian int16 relative to `full_scale` and base64 it.
    ~3x smaller than JSON floats, and 1/32767 of full scale is far finer
    than the 4 decimals the readout shows."""
    q = np.clip(np.round(values / full_scale * 32767.0), -32767, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


# Runs in the panel's iframe after panel_common.PANEL_COMMON_JS.
WAVEFORM_JS = r"""
// Waveform panel — runs in the panel's own st.iframe document, after
// PANEL_COMMON_JS. PV_DATA (built by render_waveform_group, defined just
// before this script) holds each track's base envelope and the panel's settings.
//   - draws each track from the base envelope at the current view
//     (height / px-per-second / scroll-or-fit — see PVPanel.bindView);
//   - measurement cursor (compare tracks at the same moment);
//   - playback: drives the hidden st.audio players in the parent page
//     (per-track play/pause, double-click to seek) and draws a synced
//     playhead on every track.
(function () {
  const D = PV_DATA;
  const tracks = D.tracks;
  const svgs = Array.from(document.querySelectorAll("svg.pv-wave"));
  const readout = document.getElementById("pv-readout");
  const scroller = document.querySelector(".pv-scroll");
  const root = document.getElementById("pv-root");
  const bodyEl = readout.querySelector(".pv-body");
  const NS = "http://www.w3.org/2000/svg";

  // ---- Base data ----------------------------------------------------------
  // Per-column max/min/RMS at the finest zoom, int16 relative to D.fullScale
  // (the loudest peak across all tracks, so every track shares one scale).
  function decode(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let k = 0; k < bin.length; k++) bytes[k] = bin.charCodeAt(k);
    const q = new Int16Array(bytes.buffer);
    const out = new Float32Array(q.length);
    for (let k = 0; k < q.length; k++) out[k] = (q[k] / 32767) * D.fullScale;
    return out;
  }
  tracks.forEach(tr => {
    tr.bMax = decode(tr.b64max);
    tr.bMin = decode(tr.b64min);
    tr.bRms = decode(tr.b64rms);
  });

  let plotHeight = 0, plotMid = 0, scale = 1;

  // Default to relative alignment when the clips differ in length (speed != 1):
  // then the same clock time is NOT the same moment in the music.
  const lengthsDiffer = tracks.some(t => Math.abs(t.duration - tracks[0].duration) > 0.01);
  let mode = lengthsDiffer ? "relative" : "clock";
  let pinned = false;
  let last = null;  // {ref: track index, t: seconds on that track}

  // One marker (vertical line + dot + value label) per track SVG.
  const markers = svgs.map((svg, i) => {
    const g = document.createElementNS(NS, "g");
    g.style.display = "none";
    g.style.pointerEvents = "none";
    const line = document.createElementNS(NS, "line");
    line.setAttribute("y1", 0);
    line.setAttribute("stroke", "#e5e7eb");
    line.setAttribute("stroke-width", 1);
    line.setAttribute("stroke-dasharray", "3,3");
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("r", 4.5);
    dot.setAttribute("fill", tracks[i].color);
    dot.setAttribute("stroke", "#ffffff");
    dot.setAttribute("stroke-width", 1.5);
    const label = document.createElementNS(NS, "text");
    label.setAttribute("font-size", 11);
    label.setAttribute("font-weight", 600);
    label.setAttribute("fill", "#f9fafb");
    label.setAttribute("paint-order", "stroke");
    label.setAttribute("stroke", "#0b0f14");
    label.setAttribute("stroke-width", 3);
    g.append(line, dot, label);
    svg.appendChild(g);
    return { g, line, dot, label };
  });

  // Playback playhead, one per track (drawn above the cursor marker).
  const heads = svgs.map(svg => {
    const g = document.createElementNS(NS, "g");
    g.style.display = "none";
    g.style.pointerEvents = "none";
    const line = document.createElementNS(NS, "line");
    line.setAttribute("y1", 0);
    line.setAttribute("stroke", "#fbbf24");
    line.setAttribute("stroke-width", 2);
    const tri = document.createElementNS(NS, "path");
    tri.setAttribute("fill", "#fbbf24");
    g.append(line, tri);
    svg.appendChild(g);
    return { g, line, tri };
  });

  // ---- Drawing ---------------------------------------------------------------
  // Merge base columns down to W display columns: max of maxes, min of
  // mins, RMS of RMSes (base columns cover ~equal sample counts).
  function aggregate(tr, W) {
    const B = tr.bMax.length;
    const mx = new Float32Array(W), mn = new Float32Array(W);
    const pk = new Float32Array(W), rm = new Float32Array(W);
    if (B > 0) {
      for (let k = 0; k < W; k++) {
        const s = Math.floor((k * B) / W);
        const e = Math.max(s + 1, Math.floor(((k + 1) * B) / W));
        let a = -Infinity, b = Infinity, sq = 0;
        for (let j = s; j < e; j++) {
          if (tr.bMax[j] > a) a = tr.bMax[j];
          if (tr.bMin[j] < b) b = tr.bMin[j];
          sq += tr.bRms[j] * tr.bRms[j];
        }
        mx[k] = a; mn[k] = b;
        pk[k] = a >= -b ? a : b;  // signed sample with the largest magnitude
        rm[k] = Math.sqrt(sq / (e - s));
      }
    }
    tr.cMax = mx; tr.cMin = mn; tr.peak = pk; tr.rms = rm; tr.width = W;
  }

  function drawTrack(i) {
    const tr = tracks[i], svg = svgs[i], W = tr.width, H = view.height;
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);

    // Min/max envelope: one vertical bar per pixel column.
    const parts = new Array(W);
    for (let x = 0; x < W; x++) {
      const top = plotMid - tr.cMax[x] * scale;
      const bot = Math.max(plotMid - tr.cMin[x] * scale, top + 1);  // silence still draws 1px
      parts[x] = "M" + (x + 0.5) + " " + top.toFixed(1) + "V" + bot.toFixed(1);
    }
    svg.querySelector(".pv-bars").setAttribute("d", parts.join(""));

    // Grid (quarter heights + timestamp lines) and timestamp labels.
    let g = "";
    for (const f of [0.25, 0.5, 0.75]) {
      const y = (plotHeight * f).toFixed(1);
      g += '<line x1="0" y1="' + y + '" x2="' + W + '" y2="' + y + '" stroke="#1f2a37" stroke-width="1"/>';
    }
    if (tr.duration > 0) {
      const step = PVPanel.tickStep(W / tr.duration);
      for (let k = 0; k * step <= tr.duration + 1e-9; k++) {
        const t = Math.round(k * step * 1000) / 1000;  // k * step, not +=, so no float drift
        const xv = (t / tr.duration) * W;
        const x = xv.toFixed(1);
        g += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="' + plotHeight +
             '" stroke="#1f2a37" stroke-width="1"/>';
        g += '<text x="' + x + '" y="' + (H - 4) + '" font-size="10" fill="#6b7280" text-anchor="' +
             PVPanel.tickAnchor(k, xv, W) + '">' + t + "s</text>";
      }
    }
    svg.querySelector(".pv-bg").innerHTML = g;

    markers[i].line.setAttribute("y2", plotHeight);
    heads[i].line.setAttribute("y2", plotHeight);
  }

  const msNote = readout.querySelector(".pv-ms");

  // Redraw every track at `pps` px/s (called by PVPanel on any view change);
  // returns the note to show next to the View controls.
  function drawAll(pps) {
    plotHeight = view.height - D.labelPx;
    plotMid = plotHeight / 2;
    scale = (plotMid * D.fill) / D.fullScale;

    let capped = false;
    tracks.forEach((tr, i) => {
      const want = Math.max(1, Math.round(tr.duration * pps));
      const W = Math.max(1, Math.min(want, tr.bMax.length || 1));
      if (want > W && tr.bMax.length) capped = true;
      aggregate(tr, W);
      drawTrack(i);
    });

    const ms = tracks[0] && tracks[0].duration > 0 ? (tracks[0].duration / tracks[0].width) * 1000 : 0;
    msNote.textContent = "\u24d8 value = largest-magnitude sample in the ~" + ms.toFixed(1) + " ms under the cursor";
    msNote.title = "Each pixel column covers ~" + ms.toFixed(1) + " ms of audio. Value = the sample with the " +
                   "largest magnitude in that span (sign kept); RMS = average level over the same span.";

    update();
    drawPlayhead(false);
    return capped ? "max detail reached for the longest clip" : "";
  }

  const panelView = PVPanel.bindView({
    key: "pv-waveform-view", defaults: D.defaults, limits: D.limits,
    panel: readout, scroller, root,
    maxDuration: Math.max(...tracks.map(t => t.duration)),
    onRender: drawAll,
  });
  const view = panelView.view;

  // ---- Measurement cursor ----------------------------------------------------
  const fmtV = v => (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(4);
  const db = v => (Math.abs(v) < 1e-9 ? -Infinity : 20 * Math.log10(Math.abs(v)));
  const fmtDb = d => (d === -Infinity ? "\u2212\u221e dB" : d.toFixed(1) + " dB");
  const fmtDelta = (d, unit, digits) => {
    if (!isFinite(d)) return "n/a";
    return (d >= 0 ? "+" : "\u2212") + Math.abs(d).toFixed(digits) + unit;
  };

  // Time on track j that corresponds to time t on track `ref`, per the Align setting.
  function mapTime(ref, t, j) {
    if (mode === "clock") return t;
    const d = tracks[ref].duration;
    return d > 0 ? (t / d) * tracks[j].duration : 0;
  }

  function sample(i, t) {
    const tr = tracks[i];
    if (tr.duration <= 0 || t < 0 || t > tr.duration || !tr.peak) return null;
    const x = (t / tr.duration) * tr.width;
    const col = Math.min(tr.width - 1, Math.floor(x));
    return { t, x, v: tr.peak[col], rms: tr.rms[col] };
  }

  function hide() {
    markers.forEach(m => (m.g.style.display = "none"));
    bodyEl.classList.remove("pv-active");
    bodyEl.innerHTML =
      '<div class="pv-hint">Hover a waveform to measure \u2014 click to pin the cursor \u2014 ' +
      'double-click to move playback there. Play/pause each track with its \u25b6 button.</div>';
  }

  function update() {
    if (!last) return hide();
    const vals = tracks.map((tr, i) => sample(i, mapTime(last.ref, last.t, i)));

    vals.forEach((s, i) => {
      const m = markers[i];
      if (!s) { m.g.style.display = "none"; return; }
      const y = plotMid - s.v * scale;
      m.g.style.display = "";
      m.line.setAttribute("x1", s.x);
      m.line.setAttribute("x2", s.x);
      m.dot.setAttribute("cx", s.x);
      m.dot.setAttribute("cy", y);
      const nearRight = s.x > tracks[i].width - 70;
      m.label.setAttribute("x", nearRight ? s.x - 8 : s.x + 8);
      m.label.setAttribute("text-anchor", nearRight ? "end" : "start");
      m.label.setAttribute("y", Math.max(12, Math.min(plotHeight - 4, y - 8)));
      m.label.textContent = fmtV(s.v);
    });

    const base = vals[0];
    const rows = vals.map((s, i) => {
      const tr = tracks[i];
      const sw = '<span class="pv-sw" style="background:' + tr.color + '"></span>';
      if (!s) {
        return '<tr><td>' + sw + tr.title + '</td><td colspan="6" class="pv-dim">beyond end of clip</td></tr>';
      }
      let dV = "\u2014", dRms = "\u2014";
      if (i > 0 && base) {
        dV = fmtDelta(s.v - base.v, "", 4);
        dRms = fmtDelta(db(s.rms) - db(base.rms), " dB", 1);
      }
      return "<tr><td>" + sw + tr.title + "</td>" +
        "<td>" + s.t.toFixed(3) + " s</td>" +
        "<td><b>" + fmtV(s.v) + "</b></td>" +
        "<td>" + fmtDb(db(s.v)) + "</td>" +
        "<td>" + fmtDb(db(s.rms)) + "</td>" +
        "<td>" + dV + "</td>" +
        "<td>" + dRms + "</td></tr>";
    });

    bodyEl.classList.add("pv-active");
    bodyEl.innerHTML =
      '<table><thead><tr><th>Track</th><th>Time</th><th>Value</th><th>Value (dB)</th>' +
      '<th>RMS</th><th>\u0394 value vs Original</th><th>\u0394 RMS vs Original</th></tr></thead>' +
      "<tbody>" + rows.join("") + "</tbody></table>";
  }

  let hovering = false;
  svgs.forEach((svg, i) => {
    // Cursor position is kept as a TIME, so it survives view changes.
    const tOf = e => ((e.clientX - svg.getBoundingClientRect().left) / tracks[i].width) * tracks[i].duration;
    svg.addEventListener("mousemove", e => {
      hovering = true;
      if (pinned) return;
      last = { ref: i, t: tOf(e) };
      update();
    });
    svg.addEventListener("mouseleave", () => {
      hovering = false;
      if (!pinned) { last = null; hide(); }
    });
    svg.addEventListener("click", e => {
      pinned = true;
      last = { ref: i, t: tOf(e) };
      readout.querySelector(".pv-pin").textContent = "Pinned";
      update();
    });
  });

  readout.querySelector(".pv-clear").addEventListener("click", () => {
    pinned = false;
    last = null;
    readout.querySelector(".pv-pin").textContent = "";
    hide();
  });

  readout.querySelectorAll('input[name="pv-align"]').forEach(r => {
    r.checked = r.value === mode;
    r.addEventListener("change", () => { mode = r.value; update(); drawPlayhead(false); });
  });

  // ---- Playback playhead -----------------------------------------------------
  // The st.audio players live in the parent Streamlit page (in a hidden
  // container — this panel is their UI), not in this iframe. The iframe is
  // same-origin, so we can drive and read them directly. Players are
  // matched to tracks by DOM order (Original, Processed, Naive) — the same
  // order the tracks are passed in. Whichever player last moved (played or
  // was seeked) drives a playhead on every track, mapped via Align.
  const playStatus = readout.querySelector(".pv-play");
  let audios = [];
  let lastTimes = [];
  let active = -1;

  function findAudios() {
    let doc;
    try { doc = window.parent.document; } catch (e) { return; }  // not same-origin
    // Prefer the (hidden) players container; fall back to the main area.
    const main = doc.querySelector(".st-key-" + D.playersKey) ||
                 doc.querySelector('[data-testid="stMain"]') || doc.querySelector("section.main") || doc;
    const found = Array.from(main.querySelectorAll("audio")).slice(0, tracks.length);
    if (found.length !== tracks.length) return;  // players not rendered yet
    audios = found;
    lastTimes = found.map(a => a.currentTime);
  }

  const fmtClock = t => Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0");

  function drawPlayhead(follow) {
    if (active < 0 || !audios[active]) return;
    const a = audios[active];
    const t = a.currentTime;
    heads.forEach((h, j) => {
      const tj = mapTime(active, t, j);
      const tr = tracks[j];
      if (tr.duration <= 0 || tj < 0 || tj > tr.duration) { h.g.style.display = "none"; return; }
      const x = (tj / tr.duration) * tr.width;
      h.g.style.display = "";
      h.line.setAttribute("x1", x);
      h.line.setAttribute("x2", x);
      h.tri.setAttribute("d", "M" + (x - 5) + " 0 L" + (x + 5) + " 0 L" + x + " 7 Z");
    });

    playStatus.textContent = (a.paused ? "\u275a\u275a " : "\u25b6 ") +
      tracks[active].title.replace(/&[^;]+;/g, "") + "  " + fmtClock(t);

    // Keep the playing track's playhead in view while it plays.
    if (follow && !a.paused && view.scroll) {
      const svg = svgs[active];
      const x = (t / tracks[active].duration) * tracks[active].width;
      const left = svg.getBoundingClientRect().left - scroller.getBoundingClientRect().left
                   + scroller.scrollLeft + x;
      const vw = scroller.clientWidth;
      if (left < scroller.scrollLeft + 40 || left > scroller.scrollLeft + vw - 80) {
        scroller.scrollLeft = Math.max(0, left - vw * 0.25);
      }
    }

    // With no hover/pin in the way, the measurement readout follows the playhead.
    if (!pinned && !hovering) {
      last = { ref: active, t: t };
      update();
    }
  }

  // (Re)discover players on a timer — they render after this iframe, and
  // Streamlit may re-render them.
  setInterval(() => {
    if (audios.length === 0 || !audios.every(a => a.isConnected)) {
      audios = [];
      findAudios();
    }
  }, 500);

  // ---- Transport: per-track play/pause + time, double-click to seek -------
  const tbtns = Array.from(document.querySelectorAll(".pv-tbtn"));
  const ttimes = Array.from(document.querySelectorAll(".pv-ttime"));
  const uiCache = [];  // last-rendered state per track, so the DOM only changes when needed

  tbtns.forEach((btn, i) => {
    btn.addEventListener("click", () => {
      if (audios.length === 0) findAudios();
      const a = audios[i];
      if (!a) return;
      if (a.paused) {
        // One track at a time — the point is to compare them back to back.
        audios.forEach((o, j) => { if (j !== i) o.pause(); });
        const p = a.play();
        if (p && p.catch) p.catch(() => {});
      } else {
        a.pause();
      }
      active = i;
      drawPlayhead(false);
    });
  });

  svgs.forEach((svg, i) => {
    svg.addEventListener("dblclick", e => {
      if (audios.length === 0) findAudios();
      const a = audios[i];
      if (!a) return;
      const t = ((e.clientX - svg.getBoundingClientRect().left) / tracks[i].width) * tracks[i].duration;
      a.currentTime = Math.max(0, Math.min(tracks[i].duration, t));
      active = i;
    });
  });

  function updateTransport() {
    tbtns.forEach((btn, i) => {
      const a = audios[i];
      const playing = !!a && !a.paused;
      const label = a ? fmtClock(a.currentTime) + " / " + fmtClock(tracks[i].duration) : "";
      const state = (a ? 1 : 0) + "|" + playing + "|" + label;
      if (uiCache[i] === state) return;
      uiCache[i] = state;
      btn.disabled = !a;
      btn.textContent = playing ? "\u275a\u275a" : "\u25b6";
      btn.classList.toggle("pv-on", playing);
      btn.title = !a ? "Play / pause (players still loading)" : playing ? "Pause" : "Play";
      ttimes[i].textContent = label;
    });
  }

  function loop() {
    let changed = false;
    audios.forEach((a, i) => {
      if (a.currentTime !== lastTimes[i]) {
        lastTimes[i] = a.currentTime;
        active = i;
        changed = true;
      }
    });
    if (changed) drawPlayhead(true);
    updateTransport();
    // .pv-body grows/shrinks on hover/pin (see hide()/update()) outside of
    // any View-control render, so the host <iframe> — sized once up front —
    // goes stale and the panel gets its own separate vertical scrollbar.
    // Piggyback on this already-running poll to keep it in sync too.
    PVPanel.fitFrame(root);
  }

  // Reset: stop and rewind every player, drop any pinned cursor, and put
  // the playheads + scroll position back at the start. Align and the View
  // controls are settings, not positions, so they're left as is.
  readout.querySelector(".pv-reset").addEventListener("click", () => {
    if (audios.length === 0) findAudios();
    audios.forEach(a => { a.pause(); a.currentTime = 0; });
    lastTimes = audios.map(() => 0);
    pinned = false;
    last = null;
    readout.querySelector(".pv-pin").textContent = "";
    hide();
    scroller.scrollLeft = 0;
    if (audios.length) {
      active = 0;
      drawPlayhead(false);
    } else {
      heads.forEach(h => (h.g.style.display = "none"));
      playStatus.textContent = "";
    }
  });

  panelView.render();
  findAudios();
  setInterval(loop, 33);  // ~30 fps playhead — plenty smooth, and cheap
})();
"""


def render_waveform_group(tracks: list[tuple[str, np.ndarray, int, str]]) -> None:
    """Stack several waveforms in ONE dark card with a single horizontal
    scrollbar, so they all scroll together on the shared time scale.

    `tracks` is a list of (title, samples, sample_rate, color), in the same
    order as the players passed to render_players(). Clips of different
    lengths keep the same px-per-second scale — a shorter clip simply ends
    earlier.

    All tracks share one amplitude scale (the loudest peak across the group)
    so heights — and the measurement cursor's values — are comparable.
    """
    envelopes = [_waveform_envelope(samples, sr) for _, samples, sr, _ in tracks]
    shared_peak = max(
        [1e-6]
        + [float(max(np.max(e["col_max"]), -np.min(e["col_min"])))
           for e in envelopes if len(e["col_max"])]
    )

    rows = []
    track_data = []
    for (title, _samples, _sr, color), env in zip(tracks, envelopes):
        track_data.append({
            "title": _html.escape(title),
            "color": color,
            "duration": env["duration"],
            "b64max": _b64_int16(env["col_max"], shared_peak),
            "b64min": _b64_int16(env["col_min"], shared_peak),
            "b64rms": _b64_int16(env["col_rms"], shared_peak),
        })
        rows.append(
            f"""
            <div class="pv-track" style="margin-bottom:{TRACK_GAP_PX}px;">
              <div class="pv-track-head" style="height:{TRACK_TITLE_PX}px;">
                <button class="pv-tbtn" type="button" disabled
                        title="Play / pause (players still loading)">&#9654;</button>
                <span class="pv-ttl">{_html.escape(title)}</span>
                <span class="pv-ttime"></span>
              </div>
              <svg class="pv-wave" width="{PLACEHOLDER_WIDTH_PX}" height="{WAVEFORM_HEIGHT_PX}"
                   xmlns="http://www.w3.org/2000/svg">
                <g class="pv-bg"></g>
                <path class="pv-bars" fill="none" stroke="{color}" stroke-width="1"
                      shape-rendering="crispEdges"/>
              </svg>
            </div>
            """
        )

    data = {
        "tracks": track_data,
        "fullScale": shared_peak,
        "labelPx": TRACK_LABEL_PX,
        "fill": WAVEFORM_FILL,
        "playersKey": PLAYERS_KEY,
        "defaults": {"height": WAVEFORM_HEIGHT_PX, "pps": DEFAULT_PX_PER_SECOND},
        "limits": view_limits(80, 480),
    }

    # The table area only reserves its full height (header + one row per
    # track) once the cursor readout actually has something to show — the
    # idle hint just needs a couple of lines. .pv-active (toggled by
    # WAVEFORM_JS on hover/pin) grows it, with a CSS transition so the panel
    # eases into that space instead of jumping.
    body_px = (len(tracks) + 1) * WAVEFORM_READOUT_ROW_PX
    body_idle_px = 2 * WAVEFORM_READOUT_ROW_PX
    body_css = f"""
      #pv-readout .pv-body {{ max-height: {body_idle_px}px; overflow: hidden;
                              transition: max-height 0.15s ease; }}
      #pv-readout .pv-body.pv-active {{ max-height: {body_px}px; }}
    """

    controls = view_controls_html(
        "Vertical size of each waveform track",
        "Horizontal zoom. Higher = more detail per second (the cursor's values get more "
        "precise) but a longer scroll.",
    )
    body = f"""
      <div id="pv-readout">
        <div class="pv-bar">
          <span>Align:
            <label><input type="radio" name="pv-align" value="clock"> same clock time</label>
            <label><input type="radio" name="pv-align" value="relative"> same relative position</label>
          </span>
          <button class="pv-clear" type="button">Clear</button>
          <button class="pv-reset" type="button" title="Stop all players, rewind to 0:00, clear the cursor and scroll back to the start">&#8634; Reset</button>
          <span class="pv-pin"></span>
          <span class="pv-play"></span>
        </div>
        <div class="pv-bar pv-view">
          {controls}
          <span class="pv-cap"></span>
          <span class="pv-ms"></span>
        </div>
        <div class="pv-body"></div>
      </div>
      {scroll_card_html(''.join(rows))}
    """
    render_panel(body, data, WAVEFORM_JS, extra_css=body_css)


def render_players(clips: list[np.ndarray], sample_rate: int) -> None:
    """Render the st.audio players the waveform panel drives, hidden.

    Playback lives in the waveform panel (per-track play/pause, double-click
    to seek), but these players are still what serves and plays the audio,
    so they must be rendered — just not shown. `clips` must be in the same
    order as the tracks given to render_waveform_group(); the panel matches
    players to tracks by that order.
    """
    st.html(f"<style>.st-key-{PLAYERS_KEY} {{ display: none; }}</style>")
    with st.container(key=PLAYERS_KEY):
        for clip in clips:
            st.audio(clip, format="audio/wav", sample_rate=sample_rate)
