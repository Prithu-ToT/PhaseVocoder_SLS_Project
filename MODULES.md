# Phase Vocoder Studio — Module & Class Reference

What each file in `src/` is responsible for, and what its classes and main
functions do. For *why* the layers are split the way they are, see
`architecture.md`; for the DSP math, see the docstrings in the DSP modules
and `MILESTONE_1_GUIDE.md`.

Run the app from the project root: `streamlit run src/app.py`.

---

## How a run flows

```
app.py
 ├─ sidebar.render_sidebar()                  -> Settings
 ├─ processing.run_processing(...)            -> ProcessingResult  (kept in session_state)
 │     └─ VocoderOrchestrator ─┬─ vocoder_processor ── STFTProcessor
 │                             └─ Resampler
 └─ results
     ├─ waveform_panel.render_waveform_group()      ┐ drawn in the browser,
     ├─ spectrogram_panel.render_spectrogram_group() ┘ via panel_common.render_panel()
     └─ waveform_panel.render_players()   (hidden audio players the waveform panel drives)
```

---

## UI layer

### `app.py` — page composition (entry point)

Composes the page and nothing else: page config and header, then
sidebar → processing → results. Holds the track colors
(`COLOR_ORIGINAL`, `COLOR_PROCESSED`, `COLOR_NAIVE`) and the order the
result clips are shown in, which the waveform panel, its players and the
spectrograms all share.

### `sidebar.py` — every control

| Name | What it does |
|---|---|
| `Settings` (dataclass) | Everything the controls decide for one script run: `audio_path`, `speed_factor`, `mode`, `semitone_shift`, `show_naive`, `frame_size`, `hop_size`, `process_clicked`; `is_fast` property. |
| `render_sidebar()` | Draws all sidebar controls (audio source, vocoder parameters, engine settings, Process / Generate Spectrograms buttons) and returns a `Settings`. Keeps the semitone value inside the current mode's range, and records a Generate Spectrograms click in session state. |
| `_audio_source()` | Sample track / upload / microphone picker; returns a readable file path or `None`. |
| `_save_temp()` | Writes uploaded or recorded audio to the temp folder so `AudioLoader` can read it. |
| Constants | `FAST_MODE_MAX_SEMITONES`, `ACCURATE_MODE_MAX_SEMITONES`, `SPEED_MIN`/`SPEED_MAX`, `MODE_FAST`/`MODE_ACCURATE`, `SAMPLE_TRACK_PATH`, and the session-state keys `RESULTS_KEY` / `SHOW_SPECTROGRAMS_KEY`. |

### `processing.py` — one Process Audio request (no UI)

| Name | What it does |
|---|---|
| `ProcessingResult` (dataclass) | One processed clip set: `sample_rate`, `original`, `original_duration`, `output` (peak-normalized), `naive` (or `None`), `mode`, `semitone_shift`, `snr` (identity check, or `None`). Stored in session state so results survive reruns. |
| `run_processing(...)` | Loads the audio, runs the Fast path (`orchestrator.vocoder_process`) or Accurate path (`orchestrator.process_and_write`) per the mode, optionally builds the naive-resample clip, runs the identity SNR check, and returns a `ProcessingResult`. Raises on error; the caller shows it. |
| `_identity_snr()` | Reconstruction SNR for a speed 1.0 / shift 0 run, as display text. |

### `panel_common.py` — shared by both result panels

Both panels are a control/readout bar above one dark, horizontally
scrollable card of stacked tracks, drawn in the browser inside its own
`st.iframe` so the View controls respond without a Streamlit rerun.

| Name | What it does |
|---|---|
| `render_panel(body_html, data, panel_js, extra_css)` | Assembles a panel page — CSS, body, `PV_DATA` (the `data` dict as JSON), `PANEL_COMMON_JS`, then the panel's own script — and embeds it with `st.iframe(height="content")`, so the page layout grows and shrinks with the panel. |
| `view_controls_html()` | The Height / Width / Scrollable controls. |
| `view_limits()` | Slider ranges for those controls. |
| `scroll_card_html()` | The dark card the tracks are stacked in, with one shared scrollbar. |
| `PANEL_CSS` | Styling for the bar, card, tracks and sticky track headers. |
| `PANEL_COMMON_JS` | The `PVPanel` browser helper both panel scripts use: `bindView()` (wires the View controls, remembers them per viewer in localStorage, fit-to-width zoom), `tickStep()` / `tickAnchor()` (timestamp labels), `fitFrame()` (sizes the iframe to its content). |
| Constants | Track layout sizes (`TRACK_TITLE_PX`, `TRACK_GAP_PX`, `TRACK_LABEL_PX`, `PLACEHOLDER_WIDTH_PX`) and the zoom range (`DEFAULT_/MIN_/MAX_PX_PER_SECOND`). |

### `waveform_panel.py` — Waveforms panel

| Name | What it does |
|---|---|
| `render_waveform_group(tracks)` | Renders the waveforms panel for `(title, samples, sample_rate, color)` tracks: all tracks on one amplitude scale and one time scale. |
| `render_players(clips, sample_rate)` | Renders the `st.audio` players the panel drives, hidden. They must be in the same order as the tracks. |
| `render_legend(items)` | Colored swatch + label row shown above the panel. |
| `_waveform_envelope()` | Per-column max/min/RMS of a clip at the finest zoom (cached). The browser derives every coarser zoom from it. |
| `_b64_int16()` | Packs envelope values as base64 int16 for the page (~3× smaller than JSON). |
| `WAVEFORM_JS` | Browser code: draws the tracks at the current view; measurement cursor (hover / click to pin; each track's value, RMS and Δ vs Original; Align by clock time or relative position); playback (per-track play/pause, double-click to seek, synced playhead, Reset). |
| Constants | `WAVEFORM_HEIGHT_PX`, `WAVEFORM_FILL`, `WAVEFORM_MAX_COLS` (envelope size cap), `PLAYERS_KEY` (container key of the hidden players). |

### `spectrogram_panel.py` — Spectrograms panel

| Name | What it does |
|---|---|
| `render_spectrogram_group(tracks, frame_size, hop_size)` | Renders the spectrograms panel for `(title, samples, sample_rate)` tracks, on one shared color scale (−80 dB … loudest bin). |
| `_compute_spectrogram_db()` | STFT → magnitude in dB → max-pooled to at most 512 × 4000 (cached). Also reports whether the time axis had to be pooled. |
| `_spectrogram_jpeg()` | Colors the dB array with magma into a bare JPEG (no axes); the browser stretches it to the view. |
| `_max_pool()` | Block-max downsampling along one axis, so short loud events stay visible. |
| `SPECTROGRAM_JS` / `SPECTROGRAM_CSS` | Browser code and styling: sizes each image to the current view and draws the time and frequency axes, with frequency labels pinned to the left edge while scrolling. |
| Constants | `SPECTROGRAM_MAX_TIME_BINS`, `SPECTROGRAM_MAX_FREQ_BINS`, `SPECTROGRAM_DYNAMIC_RANGE_DB`, `SPECTROGRAM_JPEG_QUALITY`, `SPECTROGRAM_HEIGHT_PX`. |

---

## Orchestration layer

### `vocoder_orchestrator.py` — `VocoderOrchestrator`

Decides *which* transformations run and in what order; the single API the
UI uses for processing.

| Method | What it does |
|---|---|
| `vocoder_process(speed_factor, pitch_factor)` | Fast path: phase-vocoder time-stretch with direct phase-rotation pitch shift. Never writes to disk. |
| `process_and_write(speed_factor, semitone_shift, output_path=None)` | Accurate path: resample to shift pitch, then time-stretch back with the phase vocoder; writes the result to disk via `AudioLoader.unload()`. |
| `get_resampled_audioloader(duration_factor)` | Naive-resample comparison clip (pitch changes with speed). Never writes to disk. |

---

## Processing layer

### `vocoder_processor.py` — `vocoder_processor`

The phase-vocoder transformation itself: STFT → phase tracking → ISTFT.
Main methods: `vocoder_process()`, `get_resampled_audioloader()`; helpers
`run_stft()`, `get_synthesis_hop()`, `calculate_wk_hat()`,
`calculate_shi_out()`.

### `stft_processor.py` — `STFTProcessor`

Hanning-windowed STFT (`stft()`) and overlap-add ISTFT (`istft()`), with
`frame_size` / `hop_size` properties and a cached `window`.

### `lanczos_resampler.py` — `Resampler`

Lanczos-windowed sinc resampling with anti-aliasing when compressing.
Main method: `get_resampled_audio()`; helpers
`calculate_output_sample_nums()`, `get_resampled_src_indx()`,
`calculate_La()`.

---

## Audio / I/O layer

### `audio_loader.py` — `AudioLoader`, `normalize_audio()`

| Name | What it does |
|---|---|
| `AudioLoader` | Loads audio (`load()`; mono down-mix, float32), holds `audio_data` / `sample_rate`, with `num_samples` and `duration_seconds` properties. `unload(output_path=None)` is the only place that writes a WAV file; `plot()` draws the waveform with matplotlib. |
| `normalize_audio(audio)` | Peak-normalizes to ±0.99. |

### `milestone_1.py`

The Milestone 1 command-line demo (see `MILESTONE_1_GUIDE.md` and
`REFACTORING.md`). Not used by the app.
