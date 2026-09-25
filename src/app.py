"""
app.py – Phase Vocoder Studio (Streamlit front end)
=====================================================
Time-stretch and pitch-shift audio through a custom STFT phase vocoder.

This file only composes the page; each responsibility lives in its own
module (see MODULES.md for the full map):
    - `sidebar.py`            – every control; returns a `Settings`
    - `processing.py`         – runs one Process request -> `ProcessingResult`
    - `waveform_panel.py`     – Waveforms panel (cursor, playback, view controls)
    - `spectrogram_panel.py`  – Spectrograms panel (same view controls)
    - `panel_common.py`       – what the two panels share

Processing itself goes through `VocoderOrchestrator` (vocoder_orchestrator.py),
which picks/coordinates the fast phase-manipulation vs. accurate
resample+stretch pipeline. See architecture.md for the full rationale.
"""

import json

import streamlit as st
import streamlit.components.v1 as components

from panel_common import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_SURFACE,
    COLOR_TEXT_BRIGHT,
    COLOR_TEXT_BRIGHTEST,
    COLOR_TEXT_DIM,
)
from processing import run_processing
from sidebar import RESULTS_KEY, SHOW_SPECTROGRAMS_KEY, render_sidebar
from spectrogram_panel import render_spectrogram_group
from waveform_panel import render_legend, render_players, render_waveform_group

# Track colors, shared by the legend and the waveform panel.
COLOR_ORIGINAL = "#2563eb"
COLOR_PROCESSED = "#059669"
COLOR_NAIVE = "#d97706"

st.set_page_config(page_title="Phase Vocoder Studio", page_icon="🎚️", layout="wide")

# Session-state keys for the "scroll results into view" trick below — kept
# separate from RESULTS_KEY/SHOW_SPECTROGRAMS_KEY since they track *when* a
# result last changed, not the result itself.
_RESULT_RUN_ID_KEY = "_pv_result_run_id"
_SCROLLED_RUN_ID_KEY = "_pv_scrolled_run_id"

# --------------------------------------------------------------------------
# Light visual polish — this is meant to read as a small product, not a
# lab notebook, so we tone down the emoji/step-by-step "assignment" framing.
# No hero banner in the page body: the title lives in Streamlit's own top
# header bar (the strip holding the hamburger menu / Deploy button) instead,
# via a CSS ::before pseudo-element — one less dark card competing for
# attention above the results. The header is also given the app's
# "raised surface" color so it reads as a panel rather than blending into
# the page background.
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      header[data-testid="stHeader"] {{
          background: {COLOR_SURFACE};
          border-bottom: 1px solid {COLOR_BORDER};
      }}
      header[data-testid="stHeader"]::before {{
          content: "Phase Vocoder";
          position: absolute;
          left: 1rem;
          top: 50%;
          transform: translateY(-50%);
          font-size: 1.05rem;
          font-weight: 700;
          color: {COLOR_TEXT_BRIGHTEST};
          pointer-events: none;
      }}
      .pv-section-title {{
          font-size: 1.05rem;
          font-weight: 600;
          margin: 1.4rem 0 0.5rem 0;
          color: inherit;
      }}
      .st-key-pv_spectro_placeholder {{
          background: {COLOR_BG};
          border: 1px solid {COLOR_BORDER};
          border-radius: 10px;
          padding: 14px 18px;
          min-height: 90px;
          display: flex;
          align-items: center;
      }}
      .st-key-pv_spectro_placeholder p {{
          color: {COLOR_TEXT_DIM};
          margin: 0;
          font-size: 0.85rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

settings = render_sidebar()

# --------------------------------------------------------------------------
# Processing — the result goes into session_state so it survives the rerun
# triggered by any later click (e.g. Generate Spectrograms).
# --------------------------------------------------------------------------
if settings.process_clicked and settings.audio_path:
    with st.spinner("Running the phase vocoder..."):
        try:
            result = run_processing(
                settings.audio_path,
                settings.speed_factor,
                settings.semitone_shift,
                fast=settings.is_fast,
                mode_label=settings.mode,
                frame_size=settings.frame_size,
                hop_size=settings.hop_size,
                include_naive=settings.show_naive,
            )
        except Exception as e:
            st.error(f"An error occurred during audio processing: {str(e)}")
            if "regular file" in str(e) or "possibly a pipe" in str(e):
                st.caption(
                    "This usually means the file isn't actually the format its extension "
                    "claims — e.g. some \"video to mp3\" downloaders save a raw MP4/AAC "
                    "stream with a `.mp3` name. Try a real WAV/FLAC/OGG/MP3 file instead."
                )
            with st.expander("Technical details"):
                import traceback
                st.code(traceback.format_exc())
        else:
            st.session_state[RESULTS_KEY] = result
            # New results — spectrograms have to be requested again.
            st.session_state[SHOW_SPECTROGRAMS_KEY] = False
            # Bump the run id so the results section scrolls into view once,
            # on the render right after this rerun — and not again on an
            # unrelated rerun (e.g. the Generate Spectrograms click) where
            # the run id stays the same.
            st.session_state[_RESULT_RUN_ID_KEY] = st.session_state.get(_RESULT_RUN_ID_KEY, 0) + 1
            # Rerun so the sidebar's Generate Spectrograms button (already
            # drawn, disabled) picks up the new results.
            st.rerun()

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
results = st.session_state.get(RESULTS_KEY)

if results is not None:
    sr = results.sample_rate

    # Persistent results footer — docked to the bottom of the *browser*
    # viewport (not just this Streamlit block), so it stays on screen while
    # scrolling. That means its markup can't just live in the normal
    # Streamlit element tree: the script below reaches into the parent page
    # (window.parent.document, same trick waveform_panel.py's findAudios()
    # uses) and creates/updates a fixed-position bar there directly, the
    # first time this runs, then just refreshes its summary text on later
    # reruns. Play/Pause and Mute drive the same hidden <audio> elements the
    # Waveforms panel's own transport controls use — "Play" resumes
    # whichever track isn't playing yet (or the first one, if none is);
    # "Pause" stops whatever's currently playing.
    #
    # This is also where the Results metrics live now — moved here from
    # the main panel so they're visible without scrolling back up, same
    # numbers as before (Sample Rate / Original & Output Duration / Pitch
    # Shift / identity-check SNR), just laid out as one line.
    _footer_out_duration = len(results.output) / sr if sr else 0.0
    _footer_stats = [
        f"{sr} Hz",
        f"Original {results.original_duration:.2f} s",
        f"Output {_footer_out_duration:.2f} s",
        f"{results.mode.split(' ')[0]} pitch shift {results.semitone_shift:+.1f} st",
    ]
    if results.snr is not None:
        _footer_stats.append(f"SNR {results.snr}")
    components.html(
        f"""
        <script>
          (function () {{
            const doc = window.parent.document;

            if (!doc.getElementById("pv-footer-style")) {{
              const style = doc.createElement("style");
              style.id = "pv-footer-style";
              style.textContent = `
                #pv-footer {{
                  position: fixed; right: 0; bottom: 0; z-index: 999999;
                  display: flex; align-items: center; gap: 0.9rem;
                  background: {COLOR_SURFACE}; border-top: 1px solid {COLOR_BORDER};
                  color: {COLOR_TEXT_DIM}; font-size: 0.8rem;
                  padding: 0.5rem 1.2rem; font-variant-numeric: tabular-nums;
                  box-shadow: 0 -2px 6px rgba(0,0,0,0.35);
                }}
                #pv-footer .pv-footer-summary {{ flex: 1 1 auto; white-space: nowrap; overflow-x: auto; }}
                #pv-footer .pv-footer-chip {{
                  display: inline-block; background: {COLOR_BORDER}; color: {COLOR_TEXT_BRIGHT};
                  border: 1px solid {COLOR_BORDER_LIGHT}; border-radius: 5px;
                  padding: 2px 9px; margin-right: 6px; font-weight: 600;
                }}
                #pv-footer button {{
                  background: {COLOR_BORDER}; color: {COLOR_TEXT_BRIGHT}; border: 1px solid {COLOR_BORDER_LIGHT};
                  border-radius: 6px; padding: 4px 12px; cursor: pointer; font-size: 0.8rem;
                }}
                #pv-footer button:hover {{ background: {COLOR_BORDER_LIGHT}; }}
                #pv-footer button.pv-on {{ background: {COLOR_ACCENT}; color: #111827; border-color: {COLOR_ACCENT}; }}
              `;
              doc.head.appendChild(style);
            }}

            let footer = doc.getElementById("pv-footer");
            const isNewFooter = !footer;
            if (isNewFooter) {{
              footer = doc.createElement("div");
              footer.id = "pv-footer";
              footer.innerHTML =
                '<span class="pv-footer-summary" id="pv-footer-summary"></span>' +
                '<button id="pv-footer-wave" type="button">Waveforms</button>' +
                '<button id="pv-footer-spec" type="button">Spectrograms</button>' +
                '<button id="pv-footer-play" type="button">▶ Play</button>' +
                '<button id="pv-footer-mute" type="button">🔊 Mute</button>';
              doc.body.appendChild(footer);

              const main = doc.querySelector('[data-testid="stMain"]') || doc.querySelector("section.main");
              if (main) main.style.paddingBottom = "54px";
            }}

            // The footer only spans the main content area — not the sidebar
            // (Process Audio / Generate Spectrograms sit at its bottom, and
            // a full-width fixed bar was drawing over them). Re-measured on
            // every rerun and every poll tick since the sidebar can be
            // resized or collapsed without either firing a resize event.
            function positionFooter() {{
              const sidebar = doc.querySelector('[data-testid="stSidebar"]');
              const left = sidebar ? sidebar.getBoundingClientRect().right : 0;
              footer.style.left = Math.max(0, left) + "px";
            }}
            positionFooter();

            if (isNewFooter) {{
              doc.defaultView.addEventListener("resize", positionFooter);

              const audios = () => Array.from(doc.querySelectorAll(".st-key-pv_players audio"));
              const waveBtn = footer.querySelector("#pv-footer-wave");
              const specBtn = footer.querySelector("#pv-footer-spec");
              const playBtn = footer.querySelector("#pv-footer-play");
              const muteBtn = footer.querySelector("#pv-footer-mute");

              const scrollToSection = id => {{
                const el = doc.getElementById(id);
                if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
              }};
              waveBtn.addEventListener("click", () => scrollToSection("waveforms"));
              specBtn.addEventListener("click", () => scrollToSection("spectrograms"));

              playBtn.addEventListener("click", () => {{
                const a = audios();
                if (!a.length) return;
                const playing = a.find(x => !x.paused);
                if (playing) {{
                  playing.pause();
                }} else {{
                  const next = a.find(x => x.currentTime > 0) || a[0];
                  const p = next.play();
                  if (p && p.catch) p.catch(() => {{}});
                }}
              }});

              muteBtn.addEventListener("click", () => {{
                const a = audios();
                if (!a.length) return;
                const nowMuted = !a[0].muted;
                a.forEach(x => {{ x.muted = nowMuted; }});
                muteBtn.classList.toggle("pv-on", nowMuted);
                muteBtn.textContent = nowMuted ? "🔇 Muted" : "🔊 Mute";
              }});

              // Playback can also start/stop from the Waveforms panel's own
              // per-track buttons, or a clip simply finishing — poll so this
              // button's label stays honest regardless of what drove it.
              setInterval(() => {{
                positionFooter();
                const playing = audios().some(x => !x.paused);
                playBtn.classList.toggle("pv-on", playing);
                playBtn.textContent = playing ? "❚❚ Pause" : "▶ Play";
              }}, 300);
            }}

            const stats = {json.dumps(_footer_stats)};
            const summary = doc.getElementById("pv-footer-summary");
            if (summary) {{
              summary.innerHTML = stats.map(s => '<span class="pv-footer-chip">' + s + '</span>').join("");
            }}
          }})();
        </script>
        """,
        height=0,
    )

    st.markdown('<div id="pv-results-anchor"></div>', unsafe_allow_html=True)

    # Scroll the results section into view, but only once per new result —
    # a later rerun that doesn't touch _RESULT_RUN_ID_KEY (e.g. the Generate
    # Spectrograms click) leaves run_id unchanged, so it's a no-op.
    _run_id = st.session_state.get(_RESULT_RUN_ID_KEY, 0)
    if _run_id and st.session_state.get(_SCROLLED_RUN_ID_KEY) != _run_id:
        st.session_state[_SCROLLED_RUN_ID_KEY] = _run_id
        components.html(
            """
            <script>
              const doc = window.parent.document;
              const el = doc.getElementById("pv-results-anchor");
              if (el) {
                el.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )

    # Clips in display order — (title, waveform title, samples, color). The
    # waveform panel, its hidden players and the spectrograms all use this
    # same order.
    processed_title = f"Processed — {results.mode}"
    clips = [
        ("Original", "Original", results.original, COLOR_ORIGINAL),
        (processed_title, processed_title, results.output, COLOR_PROCESSED),
    ]
    if results.naive is not None:
        clips.append((
            "Naive resample",
            "Naive resample (pitch shifts too — no phase correction)",
            results.naive,
            COLOR_NAIVE,
        ))

    st.markdown('<div class="pv-section-title" id="waveforms">Waveforms</div>', unsafe_allow_html=True)
    render_legend([(title, color) for title, _, _, color in clips])
    render_waveform_group([(wave_title, clip, sr, color) for _, wave_title, clip, color in clips])

    st.markdown('<div class="pv-section-title" id="spectrograms">Spectrograms</div>', unsafe_allow_html=True)
    if st.session_state.get(SHOW_SPECTROGRAMS_KEY):
        with st.spinner("Computing spectrograms..."):
            render_spectrogram_group(
                [(title, clip, sr) for title, _, clip, _ in clips],
                settings.frame_size,
                settings.hop_size,
            )
    else:
        with st.container(key="pv_spectro_placeholder"):
            st.markdown("Click **Generate Spectrograms** in the sidebar to compute them.")

    render_players([clip for _, _, clip, _ in clips], sr)

else:
    if settings.audio_path is None:
        st.info("Choose a sample track, upload a file, or record from your microphone to get started.")
    else:
        st.info("Click **Process Audio** in the sidebar to run the phase vocoder.")

    st.markdown('<div class="pv-section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
1. **Pick an audio source** — the built-in sample track, a file upload, or a microphone recording.
2. **Set the vocoder parameters** — speed factor and/or pitch shift (in semitones), and a mode:
   *Fast* (cheap, capped at a few semitones) or *Accurate* (resamples then time-stretches, wider range).
3. **Click Process Audio.** The result appears here as waveforms you can play, plus metrics like
   output duration and pitch shift.
4. **Optionally click Generate Spectrograms** to see a magnitude spectrogram (dB, magma colormap)
   of each clip, computed at the current engine settings.

Turn on **naive-resample comparison** in the sidebar to see what happens *without* the phase
vocoder — plain resampling changes pitch along with speed, which the vocoder avoids.
        """
    )
