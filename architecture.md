# Phase Vocoder Studio — Architecture

This document describes the project's module layout, with emphasis on the
orchestration refactor that introduced `VocoderOrchestrator`. It does not
change, and is not meant to describe any change to, the underlying
DSP/math — see `settings_tradeoffs.md` for tuning guidance and the
docstrings in `vocoder_processor.py` / `stft_processor.py` /
`lanczos_resampler.py` for the algorithms themselves.

---

## 1. Layers

```
┌──────────────────────────────────────────────────────────┐
│                      UI Layer                             │
│   app.py            page composition                      │
│   sidebar.py        controls -> Settings                  │
│   processing.py     one Process request -> ProcessingResult│
│   waveform_panel.py / spectrogram_panel.py / panel_common.py│
│                     result panels (browser-drawn)          │
│   Processing goes ONLY through VocoderOrchestrator +       │
│   AudioLoader (processing.py) — never Resampler or         │
│   vocoder_processor directly. The one exception:           │
│   spectrogram_panel.py runs its own display-only STFT      │
│   via STFTProcessor to draw spectrograms.                  │
└───────────────────────────┬────────────────────────────────┘
                              │
┌───────────────────────────▼────────────────────────────────┐
│         Application / Orchestration Layer                   │
│         VocoderOrchestrator (vocoder_orchestrator.py)        │
│                                                                │
│   "Which transformations should run, in what order,          │
│    and should the result be persisted?"                      │
│                                                                │
│   - vocoder_process(speed_factor, pitch_factor)               │
│       -> fast path, thin passthrough                          │
│   - get_resampled_audioloader(duration_factor)                │
│       -> naive-resample comparison, thin passthrough          │
│   - process_and_write(speed_factor, semitone_shift,           │
│                        output_path=None)                      │
│       -> accurate pitch-shift recipe + disk write              │
└───────────┬───────────────────────────────────┬──────────────┘
            │                                     │
┌───────────▼───────────┐             ┌───────────▼────────────┐
│      Resampler          │             │    vocoder_processor    │
│ (lanczos_resampler.py)  │             │ (vocoder_processor.py)  │
│                          │             │                          │
│ Lanczos-windowed sinc    │             │ STFT -> phase-vocoder    │
│ interpolation, with      │             │ phase tracking -> ISTFT  │
│ anti-aliasing when       │             │                          │
│ compressing.              │             │ Delegates windowing/     │
│                          │             │ STFT/ISTFT to            │
│                          │             │ STFTProcessor.           │
└──────────────────────────┘             └─────────────┬────────────┘
                                                          │
                                              ┌───────────▼────────────┐
                                              │     STFTProcessor       │
                                              │ (stft_processor.py)     │
                                              │                          │
                                              │ Hanning-windowed STFT /  │
                                              │ WOLA ISTFT, frame/hop    │
                                              │ size + window caching.  │
                                              └──────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Audio / I/O Layer                          │
│                    AudioLoader (audio_loader.py)               │
│                                                                  │
│   File I/O (soundfile), mono downmix, normalize_audio,          │
│   waveform plotting, and unload() — the ONLY place that         │
│   writes a WAV file to disk.                                    │
└──────────────────────────────────────────────────────────────┘
```

## 2. Responsibility table

| Component               | Responsibility                                              | Should NOT decide                          |
|--------------------------|---------------------------------------------------------------|-----------------------------------------------|
| `AudioLoader`            | Load/hold audio; write WAV via `unload()`                     | How audio should be transformed               |
| `Resampler`               | Lanczos-interpolation resampling                              | Why resampling is needed                      |
| `STFTProcessor`           | STFT / ISTFT (windowing, overlap-add)                          | Which pipeline runs, phase-vocoder policy      |
| `vocoder_processor`       | STFT + phase-vocoder phase tracking -> reconstructed audio     | Which pipeline to run; disk output             |
| `VocoderOrchestrator`     | Choose/compose the pipeline (fast vs. accurate vs. naive); persist accurate-mode results to disk | Low-level interpolation/STFT/phase math |
| `app.py`                  | Compose the page: header, sidebar -> processing -> results layout | Any DSP; any file writing; widget details   |
| `sidebar.py`              | Draw every control; return one `Settings`; save uploads/recordings to temp files | Processing; results display     |
| `processing.py`           | Run one Process request through the orchestrator; naive clip; identity SNR -> `ProcessingResult` | Any Streamlit UI; DSP math |
| `waveform_panel.py`       | Waveforms panel: envelope data, cursor, playback (drives hidden `st.audio` players), view controls | Processing   |
| `spectrogram_panel.py`    | Spectrograms panel: display STFT -> dB -> pooled JPEG, axes, view controls | Processing                      |
| `panel_common.py`         | What both panels share: layout constants, CSS, `PVPanel` JS, `render_panel()` | Panel-specific content          |

## 3. Why the refactor: separating processing from orchestration

Before this refactor, `vocoder_processor.vocoder_accurate_process()` mixed
two different kinds of responsibility in one class:

- **Signal processing** — STFT, phase-vocoder phase tracking, ISTFT
  (`vocoder_process`). A genuine, single DSP operation.
- **Workflow** — "should I resample first? by how much? can the
  phase-vocoder stage be skipped? should a temporary `AudioLoader` be
  created?" (`vocoder_accurate_process`). This is coordination logic, not
  DSP — it composes other operations rather than performing one itself.

`VocoderOrchestrator` now owns the second kind. `vocoder_processor` keeps
only the first: it is unaware of pitch-shift recipes, semitones, or file
output. This mirrors the "processing classes answer *how*; the
orchestrator answers *what/when*" principle laid out in
`orchastration.md`.

### What actually moved

- `vocoder_processor.vocoder_accurate_process()` was **removed** from
  `vocoder_processor.py` and re-implemented, math untouched, as
  `VocoderOrchestrator.process_and_write()`. The recipe (convert
  semitones to a resample ratio, resample, compute the compensating
  `true_speed_factor`, run the phase vocoder unless a trivial case
  applies) is byte-for-byte the same logic, just relocated.
- The two trivial-case optimizations (`semitone_shift == 0`, and
  `true_speed_factor ≈ 1`) are preserved exactly, in the orchestrator,
  since deciding whether a stage can be skipped is itself a workflow
  decision.
- `vocoder_processor.vocoder_process()` and
  `vocoder_processor.get_resampled_audioloader()` are untouched, single
  operations — the orchestrator exposes thin passthroughs to both so
  `app.py` has one API surface (the orchestrator) for every path.

### Writing output to disk

`AudioLoader.unload()` gained an optional `output_path` argument:

```python
def unload(self, output_path: str | None = None) -> str: ...
```

When `output_path` is given, it's written there. When omitted, `unload()`
falls back to its original behavior — deriving `<input_stem>_output.wav`
next to the loaded input file. This is the only change to `AudioLoader`;
its write path (`soundfile`, `PCM_16`) is unchanged.

`VocoderOrchestrator.process_and_write()` is the only place in the app
that persists a processed result: after computing the accurate-mode
output array, it wraps that array in a throwaway `AudioLoader` (reusing
the original source's `sample_rate` and `file_path` so the default
derived path still makes sense) and calls `unload()` on it. This keeps
"how a WAV gets written" in exactly one place (`AudioLoader.unload`)
rather than duplicating `soundfile.write` calls elsewhere.

Per-run behavior:

- **Fast mode** (`vocoder_process`) — never writes to disk. It's a cheap,
  direct operation; nothing here asked for a persisted file.
- **Accurate mode** (`process_and_write`) — always writes to disk as a
  silent internal side effect, next to the original input file by
  default. The UI shows no new element for this — no download button, no
  changed player source — it's purely an internal persistence step of the
  orchestration workflow.
- **Naive-resample comparison** (`get_resampled_audioloader`) — never
  writes to disk. It's a diagnostic/demo clip, not the orchestration
  recipe from `orchastration.md`.

## 4. UI-side dispatch (`processing.py`)

The UI no longer imports or instantiates `vocoder_processor` directly.
`processing.run_processing()` (called by `app.py`; originally inline in
`app.py`) constructs one `VocoderOrchestrator(loader, stft_processor)` per
run and dispatches on the existing Fast/Accurate toggle exactly as before:

```python
orchestrator = VocoderOrchestrator(loader, stft_processor)

if is_fast:
    pitch_factor = 2.0 ** (semitone_shift / 12.0)
    reconstructed = orchestrator.vocoder_process(speed_factor, pitch_factor=pitch_factor)
else:
    reconstructed = orchestrator.process_and_write(speed_factor, semitone_shift)
...
naive_loader = orchestrator.get_resampled_audioloader(duration_factor)
```

No behavior described in `UI_Requirement.md` changed: the dispatch rule
(mode toggle alone decides the path, regardless of semitone value), the
fast/accurate semitone caps, the speed-factor range, the naive-resample
checkbox semantics, the identity sanity check, waveform/spectrogram
rendering, and section ordering are all untouched — this refactor is
confined to *which object* the UI calls into for processing, not what the
UI shows or how the math works.

## 5. UI layer split

`app.py` used to hold the whole UI (~1,500 lines). It is now split by
responsibility, with no change in behavior:

```
app.py
 ├─ sidebar.render_sidebar()            -> Settings
 ├─ processing.run_processing(...)      -> ProcessingResult   (stored in session_state)
 └─ results layout
     ├─ waveform_panel.render_legend()
     ├─ waveform_panel.render_waveform_group()   ┐
     ├─ spectrogram_panel.render_spectrogram_group() ┤ both via panel_common.render_panel()
     └─ waveform_panel.render_players()   (hidden st.audio players the waveform panel drives)
```

The two result panels are drawn in the browser (each in its own
`st.iframe`) so their View controls, cursor and playback respond without a
Streamlit rerun. Their JavaScript and CSS live as raw strings in the panel
modules (`WAVEFORM_JS`, `SPECTROGRAM_JS`, `SPECTROGRAM_CSS`) and in
`panel_common.py` (`PANEL_CSS`, and `PANEL_COMMON_JS` — the `PVPanel`
helper both panel scripts share). See `MODULES.md` for what each module,
class and function does.

## 6. File structure

```
audio_loader.py          AudioLoader        — audio data + audio I/O (incl. unload(output_path=None))
lanczos_resampler.py     Resampler          — Lanczos interpolation / resampling
stft_processor.py        STFTProcessor      — STFT / ISTFT
vocoder_processor.py     vocoder_processor  — phase-vocoder transformation (fast path + resample passthrough)
vocoder_orchestrator.py  VocoderOrchestrator — workflow: choose/compose pipeline, persist accurate-mode output
app.py                   Streamlit entry point — composes the page
sidebar.py               Settings, render_sidebar() — all controls
processing.py            ProcessingResult, run_processing() — dispatches to VocoderOrchestrator
panel_common.py          shared panel CSS / PVPanel JS / render_panel()
waveform_panel.py        Waveforms panel + hidden audio players
spectrogram_panel.py     Spectrograms panel
```
