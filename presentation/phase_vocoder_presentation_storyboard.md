# Phase Vocoder Voice Changer — Presentation Storyboard
### CSE220 · Prithu (2305152) & Rafi (2305153)
*Editable Markdown draft — 6–8 min talk incl. 4 min live demo. Not Beamer/PPT (yet).*

> **Accuracy pass note (read once, then ignore):** I checked this storyboard's claims
> against `vocoder_orchestrator.py`, `vocoder_processor.py`, `sidebar.py`, and
> `settings_tradeoffs.md`. A handful of numbers/claims in the original master prompt
> didn't match the actual code, so I corrected them and flagged each with **⚠ corrected**.
> Biggest one: the pipeline is **not** one fixed `STFT → Vocoder → Transient → Resample`
> chain — see Slide 4.

---

## Slide 1 — The Problem

**Goal:** In the first 20 seconds, make the audience feel *why this is hard*: changing
speed changes pitch too, unless you do something clever.

**Visual Suggestion:** `IMG_001` — waveform of a spoken sentence, three lanes:
"Original", "Naive fast‑forward (chipmunk)", "Our output (fast, same pitch)".

**High-level "how":** Just a teaser — no math yet. Play the naive clip vs. ours.

**Speaker Notes (conversational):**
> "Ever sped up a voice memo and it turned into a chipmunk? That's because on a
> computer, speed and pitch are physically glued together — stretch the timeline,
> and you stretch every sound wave in it too. Today we're showing a tool that
> unglues them."

---

## Slide 2 — What Our Software Does

**Goal:** Land the feature list before any theory. This is the "what" slide.

**Visual Suggestion:** `IMG_003` — simple 3-icon feature row (speedometer / musical
note / waveform-with-sharp-edge) for the three core capabilities below.

**High-level "how":** Bullet list only — no pipeline yet:
- 🎚️ **Independent playback speed** — 0.25× to 2.4× (UI slider), slower or faster.
- 🎵 **Independent pitch shift** — in semitones, up or down, without changing speed.
- 🗣️ **Transient preservation for speech** — consonants (`t`, `k`, `p`, `s`) stay sharp
  instead of smearing when time is stretched.

**Speaker Notes:**
> "Our app is a Phase Vocoder Voice Changer. You can slow a speech down for
> transcription without dropping the pitch, pitch a voice up or down without
> changing how fast it talks, or do both at once — and it still sounds like
> speech, not mush."

---

## Slide 3 — Spotting the Change: How to Read a Spectrogram

**Goal:** Give the audience the visual vocabulary they'll need for the rest of the
talk — this is the single most important "aha" slide (per the manuscript, §1.3).

**Visual Suggestion:** `IMG_002` — side-by-side spectrograms:
1. Original clip.
2. **Naive resample** (pitch band shifts vertically *by the same proportion* the
   clip stretches horizontally — bands and duration move together).
3. **Our output** (pitch band moves independently of the horizontal/time axis).

**High-level "how":**
- Horizontal axis = time, vertical axis = frequency, color = loudness.
- Horizontal bands = sustained pitch (vowels); vertical smears = transients (consonants/drums).
- **The tell:** in a naive resample, the bands slide diagonally with the stretch.
  In ours, the bands move independently of the time axis — that decoupling *is* the project.

**Speaker Notes:**
> "Look at where the horizontal stripes sit. In the naive version, stretching the
> clip drags the stripes up or down with it — pitch and speed are chained together.
> In our output, we can move the stripes without touching the timeline at all."

---

## Slide 4 — Solution Overview: The Pipeline

**Goal:** One low-complexity slide. The audience should walk away knowing the
*order* of processing, not the math.

**Visual Suggestion:** `Beamer_Diagram_1` — pipeline diagram (flag: draw natively
in Beamer with TikZ/Beamer blocks, not an image).

```
Audio (mono, float32)
        │
        ▼
Pitch change needed? ──yes──► Lanczos Resample (shifts pitch, changes duration)
        │no                                │
        ▼                                  ▼
        └──────────────► Phase Vocoder (STFT → phase tracking
                          + transient-safe phase reset → ISTFT)
                                            │
                                            ▼
                                  Peak Normalize → Output
```

**⚠ corrected vs. master prompt draft:** The original brief proposed a single
straight chain `STFT → Phase Vocoder → Transient Correction → Resampling → Output`.
That doesn't match `vocoder_orchestrator.py`:
- **Resampling happens *before* the phase vocoder**, not after — it's how pitch
  gets shifted; the phase vocoder afterward only cleans up the *leftover* duration
  change the resample introduced.
- **Transient detection + phase reset isn't a separate stage** — it's computed
  *inside* the phase-vocoder step, before the inverse STFT, not as a post-process.
- There are actually **two distinct code paths** (Fast / Accurate), not one fixed
  chain — the diagram above is the mode-agnostic simplification; the real
  branching lives in the QnA/appendix slides.

**Speaker Notes:**
> "Three moving parts: if you asked for a pitch shift, we resample first — that's
> a well-known trick that changes pitch and duration together, on purpose. Then
> the phase vocoder steps in just to fix whatever duration is left over, while
> being careful not to smear consonants. Normalize, and you're done."

---

# Processing Pipeline — Deep-Dive Slides
*(Per the brief: these are QnA-reference slides, not for live narration.)*

## Slide 5 — Stage 1: Audio Loading

**Input:** Any file `soundfile`/FFmpeg can decode — WAV, MP3, FLAC, OGG (mic
recording also accepted via `st.audio_input`).

**Output:** 1-D mono `float32` array in `[-1.0, 1.0]`, plus its native sample rate
(nothing is resampled to a fixed rate).

**Algorithm:** Read with `soundfile` → down-mix multi-channel by averaging channels.

**Key Equation:**
$$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^{C} x_c[n]$$

**Why this step exists:** Every later stage assumes a single 1-D real signal;
this normalizes away format/channel differences up front.

**Visual suggestion:** simple block diagram, file icon → mono waveform.

**Viva note:** Stereo width/imaging is **permanently lost** here — this is a
known, stated limitation (see Slide 18 correction), not an oversight.

---

## Slide 6 — Stage 2: STFT (Short-Time Fourier Transform)

**Input:** Mono `float32` waveform.

**Output:** Complex matrix `X[k, m]` — frequency bin `k` × time frame `m`
(`frame_size//2 + 1` bins, real-signal spectrum only).

**Algorithm:** Slice into overlapping `frame_size`-sample frames spaced
`hop_size` apart, apply a Hanning window, FFT each frame (`np.fft.rfft`).

**Key Equation:**
$$X[k,m] = \sum_{n=0}^{N-1} x[mR+n]\, w[n]\, e^{-j\frac{2\pi}{N}kn}$$

**Why this step exists:** Converts a 1-D time signal into a time–frequency
representation so pitch (frequency) and timing (frame position) can be
manipulated **separately**.

**Visual suggestion:** overlapping-frames-over-waveform illustration
(`Beamer_Diagram_2` candidate).

**Viva note:** Window is a hand-implemented raised-cosine Hanning window
(no `np.hanning` call) — see Appendix slide "Why Hann Window."

---

## Slide 7 — Stage 3: Phase Vocoder (Time-Scale Modification)

**Input:** STFT matrix `X[k, m]`, target speed factor, optional pitch factor.

**Output:** A new phase-corrected STFT matrix, sampled at a different frame
spacing (`Hs`, the synthesis hop) than it was analyzed at (`Ha`).

**Algorithm:** Estimate each bin's true instantaneous frequency from the phase
drift between consecutive analysis frames, then re-accumulate phase at the new
synthesis hop so magnitude and (corrected) phase stay coherent.

**Key Equation:**
$$\hat\omega_k = \omega_k + \frac{\Delta\phi - \omega_k H_a}{H_a}, \qquad
\phi_{\text{out}}[m] = \phi_{\text{out}}[m-1] + \hat\omega_k \cdot H_s$$

**Why this step exists:** Naively replaying frames at a new spacing keeps
magnitudes right but breaks phase continuity — this is what causes the
metallic "phasiness" artifact if skipped.

**Visual suggestion:** two frame-timelines (analysis hop `Ha` vs. synthesis hop
`Hs`) with arrows showing phase re-accumulation.

**Viva note:** For a pure pitch shift, `wk_hat` is additionally scaled by
`pitch_factor` before accumulation — this is the "Fast mode" trick (direct
phase rotation), capped at **±3 semitones** because it smears energy across
bins for larger shifts.

---

## Slide 8 — Stage 4: Transient Detection & Phase Reset

**Input:** STFT magnitude matrix `|X[k,m]|`.

**Output:** A per-frame boolean transient flag; on a transient frame, synthesis
phase is **reset** to the analysis phase instead of accumulated.

**Algorithm:** Spectral flux (frame-to-frame magnitude *rise*, rectified and
summed across bins) compared against an adaptive threshold from recent history.

**Key Equation:**
$$F_t = \sum_k \max\!\big(0,\ |X_t(k)| - |X_{t-1}(k)|\big), \qquad
\text{transient if } F_t > \mu_F + 2\sigma_F$$

**Why this step exists:** Phase accumulation smears energy across a hard onset
(a consonant "attack") because it forces the attack's phase to interpolate with
its neighbors. Resetting phase at the onset keeps the attack crisp.

**Visual suggestion:** flux curve over time with a threshold line and a
reset marker at each detected spike (`Beamer_Diagram_2` / transient timeline).

**Viva note (⚠ corrected):** μ, σ are computed over a trailing **~250 ms**
window of frames (not a fixed count). The refractory period is **~20 ms**,
which — depending on hop size — works out to roughly **1–2 frames** at
default settings, not a fixed "3-frame" refractory as sometimes assumed.

---

## Slide 9 — Stage 5: Lanczos Resampling

**Input:** Mono audio, a duration-stretch factor `s` (`s < 1` = compress/pitch up,
`s > 1` = expand/pitch down).

**Output:** A new signal of length `N·s`, at fractional sample positions.

**Algorithm:** Windowed-sinc (Lanczos) interpolation; when compressing
(`s < 1`), the kernel's cutoff is lowered to the new Nyquist to act as an
anti-aliasing filter, and its support widens to match.

**Key Equation:**
$$r=\min(1,s), \qquad L_a(x) = r\cdot\operatorname{sinc}(rx)\cdot\operatorname{sinc}\!\left(\frac{rx}{a}\right)$$

**Why this step exists:** This is *how* pitch actually gets shifted — reading
the waveform out at new sample positions changes both pitch and duration; the
phase vocoder later fixes the duration side-effect.

**Visual suggestion:** sinc kernel plotted over a few neighboring samples,
before/after compression.

**Viva note:** `a = 8` taps by default. This is the **same operation** used for
the "naive resample" comparison clip in the UI — the only difference is that
here it's one deliberate stage in a larger recipe, and the naive clip skips
the phase-vocoder correction entirely (that omission is the whole point of
the naive-vs-corrected demo).

---

## Slide 10 — Stage 6: Final Output

**Input:** Reconstructed time-domain audio from the ISTFT / overlap-add stage.

**Output:** Peak-normalized `float32` audio in `[-0.99, 0.99]`, playable in
the browser; Accurate mode additionally writes a 16-bit PCM WAV to disk.

**Algorithm:** Overlap-add with squared-window normalization (WOLA), then
scale by the loudest sample.

**Key Equation:**
$$y[n] = \frac{\sum_m y_m[n-mR]\,w[n-mR]}{\sum_m w^2[n-mR]}, \qquad
x_{\text{norm}} = \frac{x}{\max|x|}\times 0.99$$

**Why this step exists:** WOLA undoes the amplitude ripple windowing
introduces; normalization prevents clipping on playback.

**Visual suggestion:** waveform before/after normalization, peak line at ±1.0.

**Viva note:** With identity settings (speed 1.0, shift 0), the app reports a
round-trip SNR — a sanity check, not a general audio-quality metric.

---

## Slide 11 — Trade-offs: Window Size, Overlap & Limits *(high priority)*

**Goal:** The audience understands the resolution trade-off and *why* the
sliders exist, without reading the full settings table live.

**Visual Suggestion:** `IMG_004` — two-axis diagram: frequency resolution ↔
time resolution, window size as the dial between them.

**High-level "how":**
- **Frequency vs. time resolution:** a bigger analysis window (up to 8192
  samples) resolves pitch better but blurs fast transients; a smaller window
  (down to 512) does the reverse.
- **Overlap vs. capability:** higher overlap (up to 90%) gives denser phase
  tracking, which is needed as the *effective stretch* moves away from 1.0.
- **Recommended defaults** (from `settings_tradeoffs.md`):
  - Speech: **1024–2048** samples (smaller for large pitch shifts, to protect
    consonants).
  - Music/sustained tone: **2048–4096** samples, default 4096 at no shift.
- **Hard limits** (from the actual code):
  - Compound (effective) time-stretch: **0.25× – 3.0×** — this is the value
    that actually gets clamped in `vocoder_processor.get_synthesis_hop`.
  - Independent pitch shift: **±3 semitones** in Fast mode, **±12** in
    Accurate mode; quality noticeably degrades past ~±9 in either mode.
  - Independent playback speed (UI slider): **0.25× – 2.4×**.

**⚠ corrected vs. master prompt draft:** the draft listed "independent time
scaling from 0.5× to 2×" and didn't distinguish Fast (±3 st) from Accurate
(±12 st) pitch caps — the sidebar's actual slider range is **0.25×–2.4×**, and
the two modes genuinely have different pitch ranges.

**Speaker Notes:**
> "Two sliders, one underlying idea: the harder we're stretching or shifting
> the audio, the more the phase vocoder has to work to keep up — so we widen
> the overlap and shrink the window as things get more extreme."

---

## Slide 12 — Transient Preservation *(high priority)*

**Goal:** Make the "why doesn't speech sound mushy" answer land as a single
clear idea, ahead of Q&A.

**Problem:** Naively accumulating phase through a hard consonant onset smears
it across neighboring frames — speech sounds soft/blurred.

**Detection:** Spectral flux, $F_t = \sum_k \max(0, |X_t(k)|-|X_{t-1}(k)|)$.

**Decision:** Flag frame `t` as a transient when it exceeds an adaptive
threshold from its own recent history.

**Action:** Reset the synthesis phase at that frame to the *analysis* phase,
instead of accumulating from the previous frame — then resume normal
accumulation right after.

**Mention:** A short refractory window (~20 ms, ≈1–2 frames at default
settings) stops one long attack from re-triggering every frame it spans.

**Visual:** `IMG_005` — smeared consonant spectrogram vs. corrected consonant
spectrogram, same clip, side by side.

**Speaker Notes:**
> "Every time the sound has a sudden attack — a 't' or a 'k' — instead of
> letting the phase vocoder blend it into its neighbors, we just tell it:
> reset here, start fresh. That single trick is most of why the speech still
> sounds like speech."

---

## Slide 13 — Resampling: Why Not Just Interpolate Linearly?

**Goal:** Close the loop on "why Lanczos" before moving to results.

**High-level "how":**
- Plain linear interpolation, when compressing a signal, has **no
  anti-aliasing** — high-frequency content folds back into the audible range
  as metallic, comb-like artifacts.
- Lanczos resampling is a **windowed-sinc** method with a built-in low-pass
  cutoff when compressing, so it acts as its own anti-aliasing filter.
- Trade-off: more computation per sample (wider kernel, more taps) for a
  cleaner result — this project defaults to `a = 8` taps.

**Completion check:** the audience should be able to say "linear resampling
aliases; Lanczos filters as it resamples" in one sentence.

**Visual suggestion:** spectrogram strip showing comb-like striping under
naive compression vs. clean output under Lanczos.

**Speaker Notes:**
> "You could resample with straight-line interpolation — it's simpler — but
> compress a signal that way and you get aliasing: high frequencies folding
> down into places they don't belong. Lanczos filters as it resamples, so
> that never happens."

---

# Results *(shown live, during the 4-minute demo)*

## Slide 14 — Result: Speed-Only Change

**Goal:** Demonstrate pitch stays fixed while duration changes.

**Setup:** Same clip, speed factor ≠ 1.0, semitone shift = 0.

**Expected observation:** Waveform/spectrogram compress or stretch
horizontally; harmonic bands stay at the **same vertical position**.

**Visual:** live waveform + spectrogram panel (already built into the app).

---

## Slide 15 — Result: Pitch-Only Change

**Goal:** Demonstrate duration stays fixed while pitch changes.

**Setup:** Speed factor = 1.0, semitone shift ≠ 0 (try Accurate mode, ±7 st).

**Expected observation:** Clip duration is unchanged; harmonic bands shift
**vertically**, and the demo audibly sounds pitched up/down but same tempo.

---

## Slide 16 — Result: Combined Speed + Pitch

**Goal:** Show the two controls are independent even used together.

**Setup:** e.g. speed 1.5×, pitch +7 semitones (pick and lock in this exact
pair before the talk — see open question below).

**Expected observation:** Bands move vertically **and** the clip compresses
horizontally, on their own separate terms — not tied to one ratio.

**Viva note:** Effective stretch for this combo = `2^(-7/12) × 1.5 ≈ 1.06` —
well inside the valid 0.25–3.0 range, so the phase-vocoder leftover
correction stays small and clean. Comment on this if asked "why this
example."

---

## Slide 17 — Spectrogram Comparison: Naive vs. Corrected

**Goal:** The payoff visual, revisited with the actual demo clip.

**Visual:** `IMG_002` (live version) — original / naive-resample / phase-vocoder
output, same duration change, side by side, shared color scale.

**Expected observation:** naive clip's bands move diagonally with the
stretch; corrected clip's bands move independently of the time axis.

---

## Slide 18 — Evaluation Checklist

**Goal:** Give a grader a table they can tick against the rubric in seconds.

| Feature | Implemented |
|---|:---:|
| Speed Control | ✔ |
| Pitch Shift | ✔ |
| Independent Speed/Pitch Control | ✔ (within 0.25×–3.0× effective stretch) |
| Transient Preservation (speech) | ✔ |
| Naive-resample comparison mode | ✔ |
| Formant Preservation | ✘ |
| Stereo Support | ✘ (mono only, down-mixed on load) |

**⚠ corrected vs. master prompt example:** the draft table marked *Formant
Preservation* and *Stereo Support* as ✔. Neither is actually true of this
implementation — `AudioLoader` down-mixes to mono unconditionally, and
neither the Fast (phase-rotation) nor Accurate (resample-then-stretch) pitch
path includes a separate formant-envelope step, so formants shift along with
pitch. Marking these ✔ would misrepresent the project to a grader; I've
listed them honestly as known, stated limitations instead (they're already
called out in `PRESENTATION_MANUSCRIPT.md` §5).

**Speaker Notes:**
> "To be upfront about scope: this is mono-only, and we don't do formant
> correction — pitch shifts move the whole voice character, not just the
> fundamental. Everything else on this list is fully implemented."

---

## Slide 19 — Conclusion

**Goal:** Recap in one breath, invite questions.

**Visual Suggestion:** Return to Slide 1's before/after clip, one more time.

**Speaker Notes:**
> "So — speed and pitch, unglued, with an extra step to keep consonants
> sharp. Two black-box algorithms doing the heavy lifting: the phase
> vocoder for time, Lanczos resampling for pitch. Happy to take questions,
> including on the math we skipped over."

---

# Appendix — Viva-Only Slides
*(One topic per slide, kept concise — pulled up only if asked.)*

## Slide 20 — Appendix: STFT Overlap-Add / WOLA Normalization

Overlap-adding windowed frames re-introduces amplitude ripple from the
window shape; dividing by the running sum of **squared** windows at each
sample position removes it exactly (for a well-chosen hop/window pair,
this is near-constant, so the correction is small). Equation on Slide 10.

## Slide 21 — Appendix: Phase Accumulation

Why can't we just IFFT the analysis phase directly at the new hop? Because
frame `m`'s phase was measured relative to hop `Ha`; replaying it at hop
`Hs` without correction produces phase discontinuities between frames,
heard as a warbling/robotic artifact. Re-accumulating from the estimated
*instantaneous* frequency instead keeps phase continuous. Equation on
Slide 7.

## Slide 22 — Appendix: Why Hann (Hanning) Window

Tapers to exactly zero at both endpoints, suppressing the spectral leakage
a hard (rectangular) frame edge would otherwise introduce. Paired with 75%
overlap, its squared-sum is nearly constant — the standard, well-behaved
choice for WOLA reconstruction. Implemented by hand
(`0.5·(1−cos(2πn/(N−1)))`), not via `np.hanning`.

## Slide 23 — Appendix: Why Lanczos (not simpler kernels)

A truncated sinc alone still has significant sidelobes; Lanczos windows the
sinc with *another* sinc, giving a sharper frequency-domain rolloff — closer
to an ideal brick-wall filter — for a similar number of taps. That rolloff
is what supplies the anti-aliasing when compressing.

## Slide 24 — Appendix: Spectral-Flux Threshold Choice

Threshold = mean + 2·standard-deviation of flux over the **trailing ~250 ms**
of frames (excluding the current frame, so a spike can't inflate its own
threshold). Two standard deviations is a common "clearly above noise floor"
heuristic for onset detection — not derived from a formal false-positive
budget in this implementation.

## Slide 25 — Appendix: Adaptive Threshold, Not a Fixed Gaussian σ
*(⚠ note: replaces the master prompt's "Gaussian sigma" appendix topic — the
code doesn't fit a Gaussian; it computes a running mean/σ of the flux signal
itself.)* The mean and σ above are recomputed every frame from a **sliding
window** of recent flux values, so the threshold adapts to the current
loudness/energy level of the clip rather than using one fixed number for
the whole track.

## Slide 26 — Appendix: Phase Reset Mechanics

On a flagged transient frame, `shi_out[:, frame] = angle(STFT[:, frame])` —
the synthesis phase is simply set to that frame's own analysis phase,
bypassing the accumulation recurrence for one frame only; accumulation
resumes normally on the next frame.

## Slide 27 — Appendix: Computational Complexity

Dominant cost is per-frame FFT/IFFT: `O(N log N)` per frame, times
`num_frames ≈ signal_length / hop_size`. Smaller hop (higher overlap) means
more frames, so cost scales roughly linearly with overlap percentage for a
fixed window size; larger windows cost more per frame but need fewer frames
for a fixed hop ratio. The Lanczos resampler is chunked (`chunk_size =
250,000` samples) to keep memory bounded on long clips.

## Slide 28 — Appendix: Valid Range & Why It's a Hard Wall

`vocoder_processor.get_synthesis_hop` raises `ValueError` outside
**0.25×–3.0×** effective stretch — this is a deliberate refuse-rather-than-
degrade choice, not a soft warning, because phase tracking quality falls
off in a way that's hard to bound gracefully near the edges.

---

# Visual Asset Log

| ID | Description | Notes |
|---|---|---|
| IMG_001 | Waveform: original / naive fast-forward / our output | Slide 1 |
| IMG_002 | Spectrogram: original / naive / corrected | Slides 3 & 17, reuse the same asset live vs. pre-rendered |
| IMG_003 | 3-icon feature row | Slide 2 |
| IMG_004 | Frequency-vs-time resolution dial | Slide 11 |
| IMG_005 | Smeared vs. corrected consonant spectrogram | Slide 12 |
| Beamer_Diagram_1 | Pipeline flowchart | Slide 4 — build natively in Beamer (TikZ blocks), don't screenshot |
| Beamer_Diagram_2 | Overlapping frames + phase-reset timeline | Slides 6/8 — also better as native Beamer diagram |

---

# Timing Targets (unchanged from brief)

| Section | Time |
|---|---|
| Problem (Slides 1–3) | 40 sec |
| Product Overview (Slides 1–3, feature framing) | 1 min 20 sec |
| Pipeline (Slide 4) | 1 min |
| Trade-offs (Slides 11–13) | 1 min |
| Core Processing (Slides 5–10) | QnA only |
| Results (Slides 14–17) | during live demo |
| Conclusion (Slide 19) | 30 sec |
| Appendix (Slides 20–28) | backup |

---

# Open Questions for You

1. **Live demo vs. screenshots for IMG_002?** Recommend live — the app already
   renders exactly this comparison.
2. **Lock in the Slide 16 combo example** (currently proposed: +7 semitones,
   1.5× speed, Accurate mode) so Slides 16/17 and the live demo all reference
   the same numbers.
3. **Beamer conversion:** flagged two diagrams (`Beamer_Diagram_1/2`) as
   better built natively in Beamer/TikZ rather than as static images — let me
   know when you want this converted and I'll build those as real Beamer code
   rather than screenshots.
