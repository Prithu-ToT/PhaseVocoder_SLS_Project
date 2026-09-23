# Resampler Theory: `s`, `r`, and Sinc-Based Resampling

This explains the math behind `lanczos_resampler.py` — what `s` and `r`
mean, why the resampler is built out of a sinc kernel, and how it plugs
into the phase vocoder's accurate pitch-shift path.

## 1. The problem: reading a signal at non-integer positions

`self.src.audio_data` is a sequence of samples `x[0], x[1], x[2], ...`
taken at fixed spacing. To time-stretch or compress the signal, we need
values like `x[2.37]` — a point *between* samples. The signal doesn't
actually have a value there; we have to reconstruct what the original
continuous-time waveform would have been and read it off at that point.

Classic sampling theory says: if `x[n]` was sampled at (or above) the
Nyquist rate, the *entire* original continuous waveform can be recovered
exactly as

```
x(t) = sum over n of  x[n] * sinc(t - n)
```

where `sinc(t) = sin(pi*t) / (pi*t)`. This is an infinite sum — every
sample contributes to every point of the reconstructed waveform. That's
the "ideal kernel is sinc(t) to infinity" comment in the code.

## 2. `s` — the duration stretch factor

`s = duration_stretch_factor` is simply the ratio

```
s = output_duration / input_duration
```

Since the output is stored at the *same* sample rate as the input,
`output_num_samples = input_num_samples * s`. The resampler then asks,
for every output sample `i`, "what source position does this correspond
to?":

```
idx[i] = i / s
```

- `s > 1` → output has *more* samples than input → the same content is
  spread over a longer stretch → each step of `i` moves less than one
  original sample forward → **stretching / upsampling**.
- `s < 1` → output has *fewer* samples than input → the same content is
  squeezed into fewer samples → each step of `i` skips more than one
  original sample → **compressing / decimating**.

Note `s` already *is* the compression ratio — not its reciprocal. This
matters in the next section.

## 3. Truncating the ideal kernel: Lanczos

Summing infinitely many samples per output point isn't practical, so we
truncate the sinc kernel with a window. The Lanczos window multiplies the
ideal kernel by a second, wider sinc:

```
L_a(x) = sinc(x) * sinc(x / a)
```

`a` is the number of side-lobes kept on each side (`a = 8` here, so 17
taps total: the center tap plus 8 on either side). In the frequency
domain this looks like a rounded-off rectangle — much closer to an ideal
brick-wall low-pass filter than a bare truncated sinc, which is why
Lanczos resampling sounds cleaner than simpler kernels (linear, cubic,
etc.) for audio.

`interpolate_src_x` uses this directly: to estimate `x[t]` for
non-integer `t`, take a weighted sum of the nearby samples, where the
weight of sample `n` is `L_a(t - n)`.

## 4. `r` — the anti-aliasing cutoff

Here's the piece that's easy to get backwards.

When `s < 1` (compressing), reading the signal out at the new, sparser
spacing is exactly a decimation operation. If the original signal has
energy above the *new*, lower effective Nyquist frequency, that energy
folds back down into the audible band as **aliasing** — you'll hear
metallic, comb-like artifacts (this is exactly what shows up as the
vertical striping in a spectrogram).

To prevent it, the source has to be low-pass filtered *before* being
read out sparsely, with the cutoff set to the new effective Nyquist. A
low-pass filter with cutoff at a fraction `r` of the original Nyquist has
impulse response `r * sinc(r*x)` (the `r` out front keeps the DC gain at
1; scaling `x` by `r` compresses the passband to `r` of the original
band). Folding that into the Lanczos-windowed kernel:

```
r = min(1, s)
L_a(x) = r * sinc(r*x) * sinc(r*x / a)
```

- `s >= 1` (stretching) → `r = 1` → no extra filtering needed, this
  reduces to the plain Lanczos kernel from §3. Makes sense: interpolating
  *more* points out of the same bandwidth never causes aliasing.
- `s < 1` (compressing) → `r = s < 1` → cutoff pulled down proportionally
  to how much you're compressing. Compress by half (`s = 0.5`) → cutoff
  at half the original Nyquist (`r = 0.5`).

The reason `r = s` and not `r = 1/s`: `s` is *already* the ratio of new
sample spacing to old (equivalently, new-Nyquist / old-Nyquist), so it
plugs straight into the filter's cutoff. Using `1/s` instead flips the
compressing and stretching cases, disabling the filter exactly when it's
needed and (harmlessly, since it saturates at 1 anyway) enabling it when
it isn't.

## 5. Support width — how many neighbors to sum

Shrinking `r` widens the kernel's main lobe: `sinc(r*x)` has its first
zero at `r*x = 1`, i.e. `x = 1/r`, so a smaller `r` pushes that zero
further out. The window term `sinc(r*x/a)` zeroes out at `x = a/r`. So
the kernel's total reach (its "support") in source samples is

```
support = ceil(a / r)
```

- `r = 1` → `support = a` (8) — the plain Lanczos-8 reach.
- `r = 0.5` (e.g. shifting pitch up an octave) → `support = 16` — twice
  as many neighboring samples are folded into each output point, because
  the filter's impulse response is twice as wide.

Getting this backwards (e.g. computing `support = ceil(r * a)`, which
*shrinks* as `r` shrinks) truncates the filter exactly when it needs to
be widest, which reintroduces artifacts even if `r` itself is correct.

## 6. Where this fits in the vocoder

`vocoder_processor.vocoder_accurate_process` uses this resampler for
pitch shifting:

```
pitch_time_stretch = 2 ** (-semitone_shift / 12)
```

- Shifting pitch **up** (`semitone_shift > 0`) needs
  `pitch_time_stretch < 1` → the resampler compresses → `r < 1` kicks in
  → anti-aliasing filtering is required and matters a lot.
- Shifting pitch **down** (`semitone_shift < 0`) needs
  `pitch_time_stretch > 1` → the resampler stretches → `r = 1` always →
  no filtering needed, which is why downward shifts never showed the
  artifact.

After resampling changes the pitch (and, as a side effect, the
duration), the phase vocoder's STFT time-stretch (`vocoder_process`)
brings the duration back to the target `speed_factor` without touching
pitch again — the resampler and the phase vocoder each solve one half of
the "change pitch without changing duration" problem.
