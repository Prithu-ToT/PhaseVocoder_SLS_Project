"""
spectrogram_panel.py – the Spectrograms results panel
=======================================================
Magnitude spectrograms (dB) of the result clips, stacked in one dark,
scrollable card with the same View controls as the waveform panel (see
panel_common.py).

Python computes each spectrogram once (STFT -> dB -> max-pool) and sends
it as a bare, axis-less image on a shared color scale; the browser
(SPECTROGRAM_JS) stretches it to the current view and draws the time and
frequency axes over it.
"""

import base64
import html as _html
import io

import numpy as np
import streamlit as st
from matplotlib import colormaps
from PIL import Image

from panel_common import (
    COLOR_SURFACE,
    DEFAULT_PX_PER_SECOND,
    PLACEHOLDER_WIDTH_PX,
    TRACK_GAP_PX,
    TRACK_LABEL_PX,
    TRACK_TITLE_PX,
    render_panel,
    scroll_card_html,
    view_controls_html,
    view_limits,
)
from stft_processor import STFTProcessor

# The (freq x time) dB array is max-pooled down to these caps before it is
# colored: a long clip has tens of thousands of STFT frames, and colorizing
# the full array once tried to allocate multiple GB (a 2.5 GiB float64
# (1025, 81895, 4) RGBA array crashed the old matplotlib version).
SPECTROGRAM_MAX_TIME_BINS = 4000
SPECTROGRAM_MAX_FREQ_BINS = 512
SPECTROGRAM_DYNAMIC_RANGE_DB = 80  # color scale spans [loudest - 80 dB, loudest]
SPECTROGRAM_JPEG_QUALITY = 88
SPECTROGRAM_HEIGHT_PX = 260  # default track height


def _max_pool(a: np.ndarray, axis: int, target: int) -> np.ndarray:
    """Shrink `a` along `axis` to at most `target` bins by taking the max of
    each block. Max (not striding) keeps short, loud events — transients,
    onsets — visible after downsampling, the same reason the waveform uses
    a min/max envelope."""
    n = a.shape[axis]
    factor = -(-n // target)  # ceil
    if factor <= 1:
        return a
    pad = factor * (-(-n // factor)) - n
    if pad:
        widths = [(0, 0)] * a.ndim
        widths[axis] = (0, pad)
        a = np.pad(a, widths, mode="edge")
    shape = list(a.shape)
    shape[axis : axis + 1] = [shape[axis] // factor, factor]
    return a.reshape(shape).max(axis=axis + 1)


@st.cache_data(show_spinner=False, max_entries=8)
def _compute_spectrogram_db(
    samples: np.ndarray, sample_rate: int, spec_frame_size: int, spec_hop_size: int
) -> tuple[np.ndarray, float, bool]:
    """Magnitude spectrogram in dB, max-pooled down to at most
    SPECTROGRAM_MAX_FREQ_BINS x SPECTROGRAM_MAX_TIME_BINS.
    Returns (mag_db [freq, time], duration covered in seconds, whether the
    time axis had to be pooled — i.e. the cap cost time detail).

    Cached so re-running the Streamlit script (any widget interaction reruns
    the whole file top to bottom) doesn't redo the STFT — and only the small
    pooled array is kept, so the large complex STFT matrix is freed right
    after this returns instead of lingering in memory across reruns.
    """
    spec_processor = STFTProcessor(frame_size=spec_frame_size, hop_size=spec_hop_size)
    stft_matrix = spec_processor.stft(samples)  # rfft: rows = 0 Hz .. Nyquist

    # float32: dB magnitudes don't need float64 precision to look right.
    mag_db = (20 * np.log10(np.abs(stft_matrix) + 1e-10)).astype(np.float32)
    extent_duration = stft_matrix.shape[1] * spec_hop_size / sample_rate
    del stft_matrix

    time_pooled = mag_db.shape[1] > SPECTROGRAM_MAX_TIME_BINS
    mag_db = _max_pool(mag_db, axis=1, target=SPECTROGRAM_MAX_TIME_BINS)
    mag_db = _max_pool(mag_db, axis=0, target=SPECTROGRAM_MAX_FREQ_BINS)
    return mag_db, extent_duration, time_pooled


@st.cache_data(show_spinner=False, max_entries=8)
def _spectrogram_jpeg(mag_db: np.ndarray, vmin: float, vmax: float) -> bytes:
    """Color `mag_db` with magma over [vmin, vmax] dB into a bare JPEG — no
    axes or margins; the browser draws those (see SPECTROGRAM_JS) so the
    image can be resized freely. Low frequencies at the bottom.

    The colormap lookup is done straight to uint8 (bytes=True) — colorizing
    to float64 RGBA first is what used to blow up memory on long clips.
    JPEG rather than PNG: spectrogram texture is noise-like and barely
    compresses losslessly, and JPEG artifacts are invisible at this scale.
    """
    norm = np.clip((mag_db - vmin) / max(1e-6, vmax - vmin), 0.0, 1.0)
    rgba = colormaps["magma"](np.flipud(norm), bytes=True)
    buf = io.BytesIO()
    Image.fromarray(rgba[..., :3]).save(buf, format="JPEG", quality=SPECTROGRAM_JPEG_QUALITY)
    return buf.getvalue()


SPECTROGRAM_CSS = r"""
.sg-plot { position: relative; }
.sg-img { display: block; }
.sg-ov { position: absolute; left: 0; top: 0; pointer-events: none; }
.sg-freq { position: absolute; left: 0; top: 0; pointer-events: none; }
.sg-freq span { position: absolute; left: 4px; font-size: 10px; line-height: 14px; color: #e5e7eb;
                background: rgba(11, 15, 20, 0.6); padding: 0 4px; border-radius: 3px;
                white-space: nowrap; }
.sg-cbar { display: inline-flex; align-items: center; gap: 6px; }
.sg-cbar i { display: inline-block; width: 120px; height: 10px; border-radius: 3px; }
"""

# The shared scroll_card_html() card defaults to the page's near-black
# COLOR_BG — fine behind the waveform bars, but it makes this panel's own
# dark-magma spectrogram images blend into their surroundings. Lift just
# this panel's card to the lighter "raised surface" shade instead (an
# override, since scroll_card_html() sets its background inline).
SPECTROGRAM_CARD_CSS = f"""
.pv-scroll.pv-scroll-dark {{ background: {COLOR_SURFACE} !important; }}
"""

# Runs in the panel's iframe after panel_common.PANEL_COMMON_JS.
SPECTROGRAM_JS = r"""
// Spectrogram panel — runs in the panel's own st.iframe document, after
// PANEL_COMMON_JS. PV_DATA (built by render_spectrogram_group, defined just
// before this script) describes each track's image. Sizes each track's bare
// spectrogram image to the current view (height / px-per-second /
// scroll-or-fit — see PVPanel.bindView) and draws the time + frequency
// axes over it.
(function () {
  const D = PV_DATA;
  const tracks = D.tracks;
  const root = document.getElementById("pv-root");
  const panel = document.getElementById("pv-readout");
  const scroller = document.querySelector(".pv-scroll");
  const plots = Array.from(document.querySelectorAll(".sg-plot"));

  // Smallest "nice" frequency step that keeps labels >= ~28 px apart.
  function freqStep(nyq, h) {
    for (const s of [100, 200, 500, 1000, 2000, 2500, 5000, 10000]) if ((s / nyq) * h >= 28) return s;
    return 20000;
  }
  const fmtHz = f => (f >= 1000 ? (f / 1000) + " kHz" : f + " Hz");

  // Draw track i at `pps` px/s; returns true if the column cap (not the
  // STFT hop itself) limits detail at this zoom.
  function drawTrack(i, pps) {
    const tr = tracks[i], plot = plots[i], H = view.height;
    const W = Math.max(1, Math.round(tr.duration * pps));
    plot.style.width = W + "px";
    const img = plot.querySelector(".sg-img");
    img.style.width = W + "px";
    img.style.height = H + "px";

    const svg = plot.querySelector(".sg-ov");
    svg.setAttribute("width", W);
    svg.setAttribute("height", H + D.labelPx);

    let g = "";
    // Frequency grid (horizontal); labels go in the pinned .sg-freq column.
    const fs = freqStep(tr.nyquist, H);
    let labels = "";
    for (let f = fs; f < tr.nyquist; f += fs) {
      const y = H - (f / tr.nyquist) * H;
      g += '<line x1="0" y1="' + y.toFixed(1) + '" x2="' + W + '" y2="' + y.toFixed(1) +
           '" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>';
      if (y > 10) labels += '<span style="top:' + (y - 7).toFixed(1) + 'px">' + fmtHz(f) + "</span>";
    }
    plot.querySelector(".sg-freq").innerHTML = labels;

    // Time grid (vertical) + timestamp labels below the image.
    if (tr.duration > 0) {
      const step = PVPanel.tickStep(W / tr.duration);
      for (let k = 0; k * step <= tr.duration + 1e-9; k++) {
        const t = Math.round(k * step * 1000) / 1000;  // k * step, not +=, so no float drift
        const xv = (t / tr.duration) * W;
        const x = xv.toFixed(1);
        g += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="' + H +
             '" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>';
        g += '<text x="' + x + '" y="' + (H + D.labelPx - 4) + '" font-size="10" fill="#6b7280" text-anchor="' +
             PVPanel.tickAnchor(k, xv, W) + '">' + t + "s</text>";
      }
    }
    svg.innerHTML = g;
    return tr.pooled && W > tr.cols * 1.5;
  }

  // Keep each track's frequency labels pinned to the visible left edge.
  function pinFreqLabels() {
    const x = scroller.scrollLeft;
    plots.forEach(p => { p.querySelector(".sg-freq").style.transform = "translateX(" + x + "px)"; });
  }

  function drawAll(pps) {
    let stretched = false;
    tracks.forEach((_, i) => { if (drawTrack(i, pps)) stretched = true; });
    pinFreqLabels();
    return stretched ? "long clip: zooming in further adds no time detail" : "";
  }

  const panelView = PVPanel.bindView({
    key: "pv-spectrogram-view", defaults: D.defaults, limits: D.limits,
    panel, scroller, root,
    maxDuration: Math.max(...tracks.map(t => t.duration)),
    onRender: drawAll,
  });
  const view = panelView.view;
  scroller.addEventListener("scroll", pinFreqLabels);

  panelView.render();
})();
"""


def render_spectrogram_group(
    tracks: list[tuple[str, np.ndarray, int]], spec_frame_size: int, spec_hop_size: int
) -> None:
    """Magnitude spectrograms (dB) of several clips, stacked in ONE dark card
    with a single horizontal scrollbar — the same layout, theme and View
    controls as the waveform panel. Computed at the given frame/hop sizes.

    `tracks` is a list of (title, samples, sample_rate). All tracks share
    one color scale — the loudest bin across the group maps to the top of
    the colormap, SPECTROGRAM_DYNAMIC_RANGE_DB below it to the bottom — so
    colors are comparable between tracks.
    """
    computed = []
    for title, samples, sample_rate in tracks:
        if len(samples) < spec_frame_size:
            st.caption(f"{title}: clip too short to compute a spectrogram at this frame size.")
            continue
        mag_db, duration, pooled = _compute_spectrogram_db(
            samples, sample_rate, spec_frame_size, spec_hop_size
        )
        computed.append((title, sample_rate, mag_db, duration, pooled))
    if not computed:
        return

    vmax = max(float(np.max(c[2])) for c in computed)
    vmin = vmax - SPECTROGRAM_DYNAMIC_RANGE_DB

    rows = []
    track_data = []
    for title, sample_rate, mag_db, duration, pooled in computed:
        jpeg = _spectrogram_jpeg(mag_db, vmin, vmax)
        track_data.append({
            "title": _html.escape(title),
            "duration": duration,
            "nyquist": sample_rate / 2,
            "cols": int(mag_db.shape[1]),
            "pooled": pooled,
        })
        rows.append(
            f"""
            <div class="pv-track" style="margin-bottom:{TRACK_GAP_PX}px;">
              <div class="pv-track-head" style="height:{TRACK_TITLE_PX}px;">
                <span class="pv-ttl">{_html.escape(title)}</span>
              </div>
              <div class="sg-plot">
                <img class="sg-img" alt=""
                     src="data:image/jpeg;base64,{base64.b64encode(jpeg).decode('ascii')}"
                     style="width:{PLACEHOLDER_WIDTH_PX}px; height:{SPECTROGRAM_HEIGHT_PX}px;">
                <svg class="sg-ov" xmlns="http://www.w3.org/2000/svg"></svg>
                <div class="sg-freq"></div>
              </div>
              <div style="height:{TRACK_LABEL_PX}px;"></div>
            </div>
            """
        )

    # Color bar legend: magma sampled into a CSS gradient.
    stops = ", ".join(
        "rgb({},{},{})".format(*(int(c * 255) for c in colormaps["magma"](x)[:3]))
        for x in np.linspace(0.0, 1.0, 9)
    )

    data = {
        "tracks": track_data,
        "labelPx": TRACK_LABEL_PX,
        "defaults": {"height": SPECTROGRAM_HEIGHT_PX, "pps": DEFAULT_PX_PER_SECOND},
        "limits": view_limits(120, 600),
    }

    controls = view_controls_html(
        "Vertical size of each spectrogram track",
        "Horizontal zoom in pixels per second of audio",
    )
    body = f"""
      <div id="pv-readout">
        <div class="pv-bar pv-view" style="border-top:none; padding-top:0; margin-bottom:0;">
          {controls}
          <span class="sg-cbar" title="Shared color scale across all tracks, in dB relative to the loudest time-frequency bin">
            &minus;{SPECTROGRAM_DYNAMIC_RANGE_DB} dB
            <i style="background: linear-gradient(to right, {stops});"></i> 0 dB (loudest)
          </span>
          <span class="pv-cap"></span>
        </div>
      </div>
      {scroll_card_html(''.join(rows))}
    """
    render_panel(body, data, SPECTROGRAM_JS, extra_css=SPECTROGRAM_CSS + SPECTROGRAM_CARD_CSS)
