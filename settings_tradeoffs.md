# Frame Length & Hop Ratio vs. Pitch Shift & Speed Change

`semitone_shift` and `speed_change` don't independently determine good
settings — they combine into a single number that actually drives the
phase vocoder's behavior:

```
effective_stretch = 2^(-semitone_shift / 12) * speed_change
```

This is `true_speed_factor` in `vocoder_accurate_process` — the actual
amount of time-scaling the phase vocoder has to perform after the
resampler has already changed the pitch. **How far this value sits from
1.0 is what governs how much overlap you need** — not the semitone shift
or speed change individually. A large pitch shift and a large speed
change in *opposite* directions can partially cancel out into a mild
`effective_stretch`, and settings can stay close to default; the same
shift and change in the *same* direction stack into a large deviation
that needs much denser overlap.

`frame_length`, by contrast, is driven mainly by `|semitone_shift|` on
its own — it's the resampler + phase-vocoder chain's phase coherence
across frequency (not the raw stretch amount) that determines how much
transient smearing you're already risking, so how much you can "spend"
on frequency resolution.

There's also a hard boundary, not just a quality one:
`get_synthesis_hop` requires `0.2 < effective_stretch < 2.5`. Outside
that range the code raises `ValueError` — no frame/hop tuning fixes it,
only reducing the shift or speed magnitude does. Those cells are marked
**invalid** below.

## General trend

- **Hop ratio (overlap) rises with `|effective_stretch − 1|`.** The
  further the true stretch factor is from 1, the more phase advances
  between frames, and the more likely phase unwrapping breaks down
  (`calculate_wk_hat`'s `delta` wraps into the wrong bin). Denser
  overlap keeps the phase estimate valid at the cost of more computation.
- **Frame length falls (or is capped) as `|semitone_shift|` rises.**
  Bigger frames buy frequency resolution but smear transients over a
  longer window — fine when there's no pitch shift, risky once the
  resampler + phase-locking-free phase propagation are already
  decorrelating transient energy across bins. Past about ±9 semitones,
  frame length stops helping and can start hurting.
- **Direction matters differently for each knob.** Positive
  `semitone_shift` (pitch up) drives the resampler's anti-aliasing
  filter into its widened, more aggressive regime — comb artifacts on
  percussive material show up there before anywhere else. Negative
  shifts combined with `speed_change > 1` are more likely to hit the
  *upper* `effective_stretch` ceiling (`> 2.5`) instead.
- **Near the `(0.2, 2.5)` boundary, quality degrades regardless of
  settings.** Overlap can only compensate so much — corners close to
  (but inside) the valid range should be treated as "expect audible
  artifacts on transient material" even with maxed-out overlap.

## Settings table

| Semitone Shift | Speed Change | Effective Stretch (\|·−1\|) | (Frame Length, Hop Ratio) | Trade-off |
|---:|---:|---:|:---:|---|
| 0   | 0.25 | 0.75 | (4096, 85%) | high-res safe; needs extra overlap for the slowdown |
| 0   | 0.5  | 0.50 | (4096, 85%) | moderate slowdown; extra overlap keeps phase coherent |
| 0   | 1.0  | 0.00 | (4096, 75%) | no change — defaults are transparent |
| 0   | 1.5  | 0.50 | (4096, 85%) | moderate speed-up; extra overlap keeps phase coherent |
| 0   | 2.0  | 1.00 | (4096, 90%) | large speed-up; dense overlap needed, near phase-tracking limit |
| −6  | 0.25 | 0.65 | (2048, 85%) | large combined stretch; resolution capped for transient safety |
| −6  | 0.5  | 0.29 | (2048, 80%) | resolution reduced; mild extra overlap |
| −6  | 1.0  | 0.41 | (2048, 80%) | pitch shift alone already needs above-default overlap |
| −6  | 1.5  | 1.12 | (2048, 90%) | stacked pitch+speed stretch; dense overlap required |
| −6  | 2.0  | —    | **invalid** | effective stretch 2.83 > 2.5 — raises `ValueError` |
| +6  | 0.25 | —    | **invalid** | effective stretch 0.18 < 0.2 — raises `ValueError` |
| +6  | 0.5  | 0.65 | (2048, 85%) | resolution reduced; extra overlap for combined compression |
| +6  | 1.0  | 0.29 | (2048, 80%) | moderate pitch-up; anti-aliasing filter + above-default overlap both active |
| +6  | 1.5  | 0.06 | (2048, 75%) | opposing pitch/speed changes mostly cancel — near-default is fine |
| +6  | 2.0  | 0.41 | (2048, 80%) | pitch-up and speed-up compound; keep overlap above default |
| −12 | 0.25 | 0.50 | (2048, 85%) | extreme combined stretch; resolution capped, dense overlap |
| −12 | 0.5  | 0.00 | (2048, 75%) | opposing changes cancel to near-unity — safest cell in this row |
| −12 | 1.0  | 1.00 | (2048, 90%) | extreme shift alone; near-max overlap, transients still at risk |
| −12 | 1.5  | —    | **invalid** | effective stretch 3.0 > 2.5 — raises `ValueError` |
| −12 | 2.0  | —    | **invalid** | effective stretch 4.0 > 2.5 — raises `ValueError` |
| +12 | 0.25 | —    | **invalid** | effective stretch 0.125 < 0.2 — raises `ValueError` |
| +12 | 0.5  | 0.75 | (2048, 85%) | extreme pitch-up + slowdown stack; comb artifacts likely on cymbals/drums |
| +12 | 1.0  | 0.50 | (2048, 85%) | extreme pitch-up alone; anti-aliasing filter near its widest — expect comb artifacts on percussion regardless of hop |
| +12 | 1.5  | 0.25 | (2048, 80%) | opposing changes partly cancel; still above-default overlap |
| +12 | 2.0  | 0.00 | (2048, 75%) | opposing changes fully cancel back toward unity — safest cell in this row |

`Hop Ratio` above is overlap percentage (matches the app's slider: hop
size = `frame_size * (1 - overlap/100)`). Rows in between these sampled
semitone/speed values interpolate the same way — compute
`effective_stretch` for your actual pair and read off the nearest bucket
above (75% for ~0, 80% for ~0.1–0.4, 85% for ~0.4–0.9, 90% for ≥~0.9).

For frame length, treat `|semitone_shift|` as the deciding factor
regardless of speed: 0 → 4096 is safe, 1–8 → 2048 is the sensible
ceiling, and past ±9 semitones consider dropping to 1024 specifically
for percussion-heavy or otherwise transient-rich source material — the
comb artifacts at that point come from phase decorrelation across bins
(see `resampler_theory.md` and the earlier phasiness discussion), which
smaller frames only partially mitigate.
