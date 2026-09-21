# UI Requirements — Phase Vocoder Studio (`app.py`)

This file records UI/UX decisions that were deliberately made for this project.
Anyone (human or agent) repolishing or restyling `app.py` later should treat
the items under **Major Decisions** as constraints, not suggestions — visual
style, colors, copy, and layout details can change, but the underlying
capability/behavior must not be silently dropped or altered.

---

## Major Decisions (must not be overridden)

### 1. Microphone recording must be retained
- The audio-source selector must offer **three** ways to get audio in:
  sample track, file upload, **and** live microphone recording.
- Recording is implemented with Streamlit's built-in `st.audio_input`
  (confirmed available in this project's environment, Streamlit ≥1.31).
  Do not silently drop this in favor of only sample/upload, and do not
  swap in a third-party recorder component without checking first —
  `st.audio_input` was explicitly chosen to avoid extra dependencies.
- Recorded audio is saved to a temp WAV file the same way an uploaded
  file would be, then loaded through the normal `AudioLoader` path.

### 2. Waveform graphs must be human-readable and scrollable
- Waveforms must **not** be squeezed into fixed-width side-by-side
  columns (this was an earlier iteration that was explicitly rejected).
- Each waveform (Original / Processed / Naive resample) is its own
  full-width card, stacked **vertically**.
- Each waveform renders at a width proportional to clip duration
  (see "Shared time scale" below) inside a horizontally scrollable
  container, so long clips can be scrolled through and zoomed into
  visually rather than being crushed to fit the screen.

### 3. Spectrograms must be kept
- A magnitude spectrogram (dB scale, `20*log10(|STFT|+eps)`, `magma`
  colormap) must be shown for every signal that gets a waveform card:
  Original, Processed, and Naive resample (when enabled).
- Spectrograms are computed fresh from the actual output audio (not
  reused from internal pipeline state) using the sidebar's current
  frame size / hop size, via a throwaway `STFTProcessor` instance —
  this keeps them accurate even for the naive-resample clip, which
  never otherwise passes through `STFTProcessor`.
- Spectrograms must also be horizontally scrollable, not squashed to
  container width — rendered as a PNG at a duration-proportional
  width and embedded in the same scrollable-card style as the
  waveforms (not a plain `st.pyplot` call, which sizes to the
  container and can't scroll).

### 4. Technical vs. user-facing controls must stay separated
- **Sidebar ("Engine Settings")**: STFT internals only —
  frame size, overlap ratio (→ hop size). This is technical/DSP
  configuration, not something the audience needs to touch.
- **Main page ("Vocoder Parameters")**: the controls a viewer/instructor
  actually cares about — speed factor, pitch-shift mode, semitone
  shift, and the naive-resample comparison checkbox.
- If controls are reorganized later (e.g. into a dedicated "Advanced
  Settings" panel/expander instead of the sidebar), the *separation*
  itself — technical/DSP config kept out of the primary flow — must be
  preserved, even if the container changes.

### 5. Fixed-interval timestamp markers, shared across waveform and spectrogram
- Both waveform **and spectrogram** cards must use the **same** tick
  rule: **10s markers for clips ≤90s, 15s markers for longer clips.**
  This is not an arbitrary/cosmetic choice — it was chosen deliberately
  over "N evenly-spaced ticks" schemes (rejected in an earlier
  iteration) specifically so tick spacing is predictable and readable
  regardless of clip length.
- Waveform and spectrogram cards must also share a **common
  pixels-per-second scale** (`PX_PER_SECOND = 120`, clamped between
  `MIN_PLOT_WIDTH = 900` and `MAX_PLOT_WIDTH = 12000` px) so that a
  given timestamp lines up at the **same horizontal position across
  all six graphs** (Original/Processed/Naive × waveform/spectrogram)
  for a given clip. A future restyle must not give spectrograms a
  different width formula or tick rule than the waveforms — the two
  graph types are meant to be visually comparable at a glance, not
  just individually readable.

---

## Other locked-in behavior (from this conversation's decisions)

- **Dispatch rule**: the Pitch-Shift Mode toggle alone decides the
  code path — **Fast → `vocoder_process`**, **Accurate →
  `vocoder_accurate_process`** — always, regardless of whether the
  semitone shift is 0. Do not reintroduce a special case that routes
  0-semitone requests differently.
- **Fast mode is capped at ±3 semitones** (`FAST_MODE_MAX_SEMITONES`).
  This is a correctness constraint, not a UI nitpick: direct STFT
  phase manipulation smears energy across bins past a few semitones.
  Accurate mode is allowed up to ±12 semitones
  (`ACCURATE_MODE_MAX_SEMITONES`).
- **Speed factor range**: slider is bounded to `0.25–2.4`
  (`SPEED_MIN` / `SPEED_MAX`), staying safely inside
  `vocoder_processor.get_synthesis_hop`'s valid `0.20 < sf < 2.5` range.
- **Naive-resample comparison checkbox**: exists specifically to
  demonstrate, side by side, that the phase vocoder preserves pitch
  while plain resampling does not. It calls
  `vp.get_resampled_audioloader(1 / speed_factor)` — duration change
  only, via `np.interp`, no phase correction — and must **ignore the
  pitch-shift slider entirely** (that's the point of the comparison).
- **Identity sanity check**: when speed = 1.0 and semitone shift = 0,
  an SNR check against the original signal is shown as a correctness
  indicator (round-trip should be near-transparent). This should stay
  gated to that exact condition — it isn't meaningful once speed or
  pitch actually changes the output length/content.
- **Color coding + legend**: Original = `#2563eb` (blue),
  Processed = `#059669` (green), Naive resample = `#d97706` (orange).
  A color legend must be shown above the waveform section so the
  color coding is explained, not just implied.
- **Shared time scale & timestamp markers**: see Major Decision #5 —
  non-negotiable, not just a styling note.
- **Section order** on the results page: Results metrics → Waveforms
  (with legend) → Spectrograms → Listen (audio players). Waveforms
  come before spectrograms, not interleaved per-signal.
- **Visual tone**: styled to read as a small product/demo (dark hero
  header, card-style sections) rather than a bare "Step 1 / Step 2"
  assignment-style layout — this was an explicit ask when this was
  being prepared to show an instructor.

---
