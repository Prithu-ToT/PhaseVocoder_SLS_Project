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
│                      UI Layer (app.py)                   │
│   Streamlit widgets, plotting, audio players.             │
│   Talks ONLY to VocoderOrchestrator + AudioLoader for     │
│   metrics/display — never to Resampler or STFTProcessor   │
│   directly.                                                │
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
| `app.py`                  | Streamlit UI: inputs, dispatch to the orchestrator, rendering  | Any DSP; any file writing                      |

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

## 4. `app.py` changes

`app.py` no longer imports or instantiates `vocoder_processor` directly.
It constructs one `VocoderOrchestrator(loader, stft_processor)` per run
and dispatches on the existing Fast/Accurate toggle exactly as before:

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

## 5. File structure

```
audio_loader.py          AudioLoader        — audio data + audio I/O (incl. unload(output_path=None))
lanczos_resampler.py     Resampler          — Lanczos interpolation / resampling
stft_processor.py        STFTProcessor      — STFT / ISTFT
vocoder_processor.py     vocoder_processor  — phase-vocoder transformation (fast path + resample passthrough)
vocoder_orchestrator.py  VocoderOrchestrator — workflow: choose/compose pipeline, persist accurate-mode output
app.py                   Streamlit UI — dispatches to VocoderOrchestrator
```
