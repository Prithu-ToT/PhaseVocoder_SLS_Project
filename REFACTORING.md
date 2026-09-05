# Milestone 1 Refactor Notes

This document explains what changed when `milestone_1.py` (originally a flat
script of five loose functions) was reorganized into classes, and why. It's
meant to sit alongside `MILESTONE_1_GUIDE.md`, which still describes the DSP
math — this file is about the *code structure*, not the signal processing.

---

## 1. New directory layout

Both `app.py` and `milestone_1.py` moved into `src/`. `target_io/` (sample
audio, demo output) stays at the project root, so it is now a **sibling of
`src/`**, not a sibling of the scripts themselves:

```
project_root/
├── target_io/
│   ├── mohiner_ghoraguli_sample.mp3
│   └── milestone_1_output.wav
├── MILESTONE_1_GUIDE.md
├── REFACTORING.md
└── src/
    ├── audio_loader.py       # AudioLoader
    ├── stft_processor.py     # STFTProcessor
    ├── milestone_1.py        # normalize_audio() + CLI demo
    └── app.py                # Streamlit UI
```

Every path that used to be built off `os.path.dirname(__file__)` and assume
`target_io/` was right there now goes one level further up first:

```python
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
TARGET_IO_DIR = os.path.join(PROJECT_ROOT, "target_io")
```

This is done independently in both `milestone_1.py` and `app.py`, so each
file resolves `target_io/` correctly regardless of the current working
directory the script is launched from.

---

## 2. From functions to classes

### `AudioLoader` (`src/audio_loader.py`)

Replaces the old `load_and_prep_audio()` function.

| Old | New |
|---|---|
| `sr, audio = load_and_prep_audio(path)` | `loader = AudioLoader(path)` → `loader.sample_rate`, `loader.audio_data` |
| *(no plotting existed)* | `loader.plot()` — matplotlib waveform plot |
| *(recomputed every call)* | `loader.duration_seconds`, `loader.num_samples` — convenience properties |

The loading logic (down-mix to mono, cast to `float32`, etc.) is unchanged —
it's the exact same math, just living on `self` instead of being returned
as a bare tuple. `plot()` is new: it renders the loaded waveform with
matplotlib, downsampling for long files and returning the `Axes` so it can
be composed into a larger figure (e.g. several `AudioLoader`s plotted side
by side) if needed later.

### `STFTProcessor` (`src/stft_processor.py`)

Replaces `create_hanning_window()`, `perform_stft()`, and `perform_istft()`.

| Old | New |
|---|---|
| `win = create_hanning_window(frame_size)` | `processor.window` (auto-generated, cached) |
| `perform_stft(audio, frame_size, hop_size, win)` | `processor.stft(audio)` |
| `perform_istft(stft, frame_size, hop_size, win, expected_length)` | `processor.istft(stft, expected_length)` |

`frame_size` and `hop_size` are properties — settable in the constructor
(`STFTProcessor(frame_size=2048, hop_size=512)`) or reassigned any time
afterward (`processor.frame_size = 4096`). `stft()` / `istft()` also accept
one-off `frame_size=` / `hop_size=` overrides per call without touching the
instance's stored settings.

**Window caching:** the Hanning window is expensive-ish to build (a `cos()`
over the whole frame) and was previously recomputed by hand wherever it was
needed. Now it's built lazily and cached — `_get_window()` only rebuilds it
when the requested frame size differs from whichever one is currently
cached; otherwise it just returns the array it already has:

```python
def _get_window(self, frame_size):
    if self._cached_frame_size != frame_size or self._cached_window is None:
        self._cached_window = self._create_hanning_window(frame_size)
        self._cached_frame_size = frame_size
    return self._cached_window
```

Only the *last* frame size is cached (not a dict of every size ever used),
matching how the processor is actually driven — one frame size at a time,
occasionally changed via the sidebar slider in `app.py`.

### `normalize_audio()` — stayed a function

This one didn't fit either class (it's not about loading audio or
transforming spectra), so it stayed as a plain function in `milestone_1.py`,
which now also keeps the file's original module docstring and its
`__main__` CLI demo, rewritten to use the two classes above.

---

## 3. Fixing the Pylance / `None`-type warnings

Two attributes are genuinely `Optional` by design: `AudioLoader.audio_data`
and `AudioLoader.sample_rate` start as `None` because "no file loaded yet"
is a real, distinct state (e.g. `AudioLoader()` with no path). That
optionality is correct — the fix wasn't to remove it, but to make sure the
code around it actually proves to the type checker when it's safe to treat
those values as non-`None`. Three specific issues were cleaned up:

**a) `AudioLoader.plot()`'s `ax` parameter had no type annotation.**
An un-annotated `ax=None` default is inferred as the literal type `None`,
so every later call like `ax.plot(...)` was flagged (`None` has no
`.plot()` method). Fixed by giving it a real type behind a
`TYPE_CHECKING` guard (so `matplotlib` isn't required just to *import* the
module):

```python
if TYPE_CHECKING:
    from matplotlib.axes import Axes
...
def plot(self, ax: Axes | None = None, ...) -> Axes:
```

**b) Repeated `self.attr` access after a `None` check doesn't reliably
narrow.** Pylance can lose track of `self.audio_data`'s narrowed type
across intervening calls. Fixed by binding to local variables right after
the check, once, and using the locals from then on (see
`duration_seconds` and `plot()` in `audio_loader.py`):

```python
audio_data = self.audio_data
sample_rate = self.sample_rate
if audio_data is None or sample_rate is None:
    raise RuntimeError(...)
# audio_data / sample_rate are now known non-None for the rest of the function
```

**c) `STFTProcessor._get_window()` was declared to return `np.ndarray` but
its cache attribute is typed `np.ndarray | None`.** Even though the logic
guarantees the cache is populated before the `return`, the type checker
can't see that across two separate attribute reads. Fixed with an explicit
`assert` right before the return, which both documents the invariant and
narrows the type:

```python
if self._cached_frame_size != frame_size or self._cached_window is None:
    self._cached_window = self._create_hanning_window(frame_size)
    self._cached_frame_size = frame_size
assert self._cached_window is not None
return self._cached_window
```

**d) Call sites (`milestone_1.py`, `app.py`) treated `loader.sample_rate` /
`loader.audio_data` as non-`None` without saying so.** Both scripts only
ever construct `AudioLoader(some_path)`, and a successful construction
means `load()` already ran and populated both fields (any failure would
have raised instead) — but that guarantee lives in a different function
than the one reading the attributes, so the type checker can't infer it on
its own. A single `assert` states the guarantee explicitly:

```python
loader = AudioLoader(audio_path)
assert loader.audio_data is not None and loader.sample_rate is not None
sr, audio = loader.sample_rate, loader.audio_data
```

After this, `sr` and `audio` are plain `int` / `np.ndarray` for the rest of
the script, so passing them into `processor.stft(audio)`, `sf.write(...,
sr)`, etc. no longer trips "possibly `None`" warnings either.

Verified with `pyright` directly against these files: 0 errors / 0
warnings on `audio_loader.py`, `stft_processor.py`, and `milestone_1.py`;
`app.py` comes back clean too once the (expected, unrelated) "streamlit
isn't installed in this environment" stub notice is set aside.

---

## 4. `app.py`: separate Original / Reconstructed charts

The waveform section used to render one combined `st.line_chart` with all
three series (original, reconstructed, magnified error) overlaid on the
same axes. That's now split into:

- **Two side-by-side columns** (`st.columns(2)`), each with its own
  `st.line_chart` — Original Signal on the left, Reconstructed Signal on
  the right, each on its own scale.
- **One full-width chart below** for the magnified reconstruction error,
  kept separate since it's a diagnostic view rather than something to
  compare directly against the other two.

This only touches the plotting block inside `app.py`'s processing branch;
the STFT/ISTFT pipeline itself, the metrics row, and the audio playback
section are unchanged.

Note: this app-level visualization still uses Streamlit's native
`st.line_chart` (interactive, renders in the browser) rather than
`AudioLoader.plot()` (matplotlib). The matplotlib method remains available
for non-Streamlit use — e.g. `milestone_1.py` run from the command line, or
a notebook — where an interactive browser chart isn't the target.

---

## 5. Running things after the move

```bash
# from the project root
python src/milestone_1.py                       # uses target_io/mohiner_ghoraguli_sample.mp3
python src/milestone_1.py path/to/other.wav      # or an explicit file

streamlit run src/app.py                         # Streamlit UI
```

Both resolve `target_io/` correctly regardless of which directory they're
launched from, since the lookup is anchored to each script's own file
location rather than the process's current working directory.
