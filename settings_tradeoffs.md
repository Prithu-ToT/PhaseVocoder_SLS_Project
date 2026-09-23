# Phase Vocoder Settings Guide

## Choosing Frame Length & Hop Ratio

The phase vocoder has two important tuning parameters:

- **Frame Length** — window size used for the STFT.
- **Hop Ratio** — overlap between consecutive frames.

These are not chosen independently from pitch shift and speed change. Instead, the real quantity that determines how difficult the processing becomes is the **effective stretch factor**.

---

## Effective Stretch

After the resampler changes pitch, the phase vocoder only performs the remaining time-scaling.

<math value="\\text{effective_stretch}=2^{-\\frac{\\text{semitone_shift}}{12}}\\times\\text{speed_change}"/>

This is `true_speed_factor` inside `vocoder_accurate_process`.

The farther `effective_stretch` is from **1.0**, the harder phase tracking becomes.

Examples:

- `+12 semitones + 2× speed` → `0.5 × 2 = 1.0` → easy.
- `−12 semitones + 2× speed` → `2 × 2 = 4.0` → invalid.

A large pitch shift and speed change can either **cancel each other** or **stack together** depending on their directions.

---

## Valid Operating Range

The implementation accepts approximately:

<math value="0.2<\\text{effective_stretch}\\le3.0"/>

Outside this range, the code raises `ValueError`.

Changing frame length or overlap cannot fix those cases.

---

# Audio Type Matters

The recommendations differ depending on the source material.

## Music / Vowel-Rich Audio

Best for:

- singing
- piano
- guitar
- strings
- synth pads
- sustained vowels

These signals benefit from larger windows because harmonic accuracy matters more than preserving tiny transients.

The settings table below is primarily tuned for this case.

## Human Speech (≈30 s recordings)

Speech contains many short consonant transients:

- `t`
- `k`
- `p`
- `s`
- `ch`
- `f`

Large windows blur these sounds, making speech feel muffled.

For speech, **only the frame length changes**.

| Pitch Shift | Music | Speech |
|---:|:---:|:---:|
| 0 | 4096 | **2048** |
| ±1–8 | 2048 | **1024–2048** |
| ±9–12 | 2048 | **1024** |

The **hop ratio stays the same**, since it depends on `effective_stretch`, not on whether the source is speech or music.

---

# General Trends

- **Hop ratio increases as `|effective_stretch−1|` grows.**

  Larger stretch means greater phase movement between frames, so denser overlap helps phase unwrapping remain correct.

- **Frame length generally decreases as `|semitone_shift|` grows.**

  Larger windows improve frequency resolution but also increase transient smearing.

- **Positive pitch shifts** stress the resampler's anti-aliasing filter more, so percussion artifacts usually appear earlier than with equivalent downward shifts.

- **Near the `0.2–3.0` limits**, artifacts become noticeable regardless of settings.

---

# Settings Table (Music)

| Semitone | Speed | Effective Stretch | Recommended Settings | Notes |
|---:|---:|---:|:---:|---|
| 0 | 0.25 | 0.25 | (4096, 85%) | extreme slowdown |
| 0 | 0.5 | 0.50 | (4096, 85%) | moderate slowdown |
| 0 | 1.0 | 1.00 | (4096, 75%) | default |
| 0 | 1.5 | 1.50 | (4096, 85%) | moderate speed-up |
| 0 | 2.0 | 2.00 | (4096, 90%) | large speed-up |
| 0 | 3.0 | 3.00 | (4096, 90%) | quality limit |
| −6 | 0.25 | 0.35 | (2048, 85%) | heavy pitch-down + slowdown |
| −6 | 0.5 | 0.71 | (2048, 80%) | moderate stretch |
| −6 | 1.0 | 1.41 | (2048, 80%) | pitch shift only |
| −6 | 1.5 | 2.12 | (2048, 90%) | dense overlap |
| −6 | 2.0 | 2.83 | (2048, 90%) | near upper limit |
| −6 | 3.0 | 4.24 | **invalid** | exceeds 3.0 |
| +6 | 0.25 | 0.18 | **invalid** | below 0.2 |
| +6 | 0.5 | 0.35 | (2048, 85%) | strong compression |
| +6 | 1.0 | 0.71 | (2048, 80%) | moderate pitch-up |
| +6 | 1.5 | 1.06 | (2048, 75%) | nearly cancels |
| +6 | 2.0 | 1.41 | (2048, 80%) | moderate stretch |
| +6 | 3.0 | 2.12 | (2048, 90%) | near upper limit |
| −12 | 0.25 | 0.50 | (2048, 85%) | extreme pitch-down |
| −12 | 0.5 | 1.00 | (2048, 75%) | perfect cancellation |
| −12 | 1.0 | 2.00 | (2048, 90%) | quality limit approaching |
| −12 | 1.5 | 3.00 | (2048, 90%) | maximum valid |
| −12 | 2.0 | 4.00 | **invalid** | exceeds limit |
| −12 | 3.0 | 6.00 | **invalid** | exceeds limit |
| +12 | 0.25 | 0.125 | **invalid** | below limit |
| +12 | 0.5 | 0.25 | (2048, 85%) | aggressive anti-aliasing |
| +12 | 1.0 | 0.50 | (2048, 85%) | pitch-up only |
| +12 | 1.5 | 0.75 | (2048, 80%) | partial cancellation |
| +12 | 2.0 | 1.00 | (2048, 75%) | perfect cancellation |
| +12 | 3.0 | 1.50 | (2048, 85%) | moderate stretch |

---

# Quick Lookup

## Step 1 — Choose Hop Ratio

Compute:

<math value="\\left|\\text{effective_stretch}-1\\right|"/>

Then use:

| Distance from 1 | Hop Ratio |
|---:|:---:|
| 0 | **75%** |
| 0.1–0.4 | **80%** |
| 0.4–0.9 | **85%** |
| ≥0.9 | **90%** |

---

## Step 2 — Choose Frame Length

### Music

| Pitch Shift | Frame Length |
|---:|:---:|
| 0 | **4096** |
| ±1–8 | **2048** |
| ±9–12 | **1024–2048** |

Use **1024** for percussion-heavy material where transient preservation matters more than harmonic resolution.

### Speech

| Pitch Shift | Frame Length |
|---:|:---:|
| 0 | **2048** |
| ±1–8 | **1024–2048** |
| ±9–12 | **1024** |

This preserves consonant clarity much better than the music-oriented defaults.

---

# Practical Recommendations

### Music

- Default song processing → **4096 / 75%**
- Moderate pitch shift → **2048 / 80–85%**
- Large pitch or speed changes → **2048 / 90%**

### Human Speech

- Voice recordings → **2048 / 75–80%**
- Large pitch shifts → **1024 / 80–90%**
- Fast playback (2–3×) → keep **2048**, increase overlap rather than increasing window size.