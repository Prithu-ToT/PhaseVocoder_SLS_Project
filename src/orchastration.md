# Refactoring the Vocoder Pipeline: Separating Processing from Orchestration

## 1. Current situation

The current `VocoderProcessor` contains two different kinds of responsibility.

### Signal-processing responsibility

`VocoderProcessor` should be responsible for operations such as:

* STFT
* Phase processing
* Phase-vocoder time adjustment
* ISTFT
* Producing transformed audio

For example:

```python
vocoder_process(
    speed_factor,
    pitch_factor
)
```

is a genuine signal-processing operation.

### Workflow / orchestration responsibility

`vocoder_accurate_process()` is different.

It performs a multi-stage recipe:

```text
Requested pitch shift
        |
        v
Calculate resampling factor
        |
        v
Resample audio
        |
        v
Calculate compensating time stretch
        |
        v
Run phase vocoder
        |
        v
Return final result
```

This is not one algorithm. It is coordinating several algorithms.

It also knows:

* when a temporary `AudioLoader` must be created
* when a phase-vocoder object must be created
* which processing path should be used
* when an optimization can skip a processing stage
* when temporary resources should be cleaned up
* eventually, where the final output should be written

This makes it a better fit for an **orchestrator**.

---

# 2. Proposed architecture

Split the responsibilities into four components:

```text
                    ┌──────────────────────┐
                    │  VocoderOrchestrator │
                    │                      │
                    │  Workflow / Policy   │
                    └──────────┬───────────┘
                               |
                 ┌─────────────┴─────────────┐
                 |                           |
                 v                           v
          ┌─────────────┐             ┌──────────────┐
          │  Resampler  │             │   Vocoder    │
          │             │             │  Processor   │
          │ Resampling  │             │              │
          └─────────────┘             │ STFT / PV /  │
                                      │ ISTFT         │
                                      └──────────────┘
                 |
                 v
          ┌─────────────┐
          │ AudioLoader │
          │             │
          │ Audio data  │
          │ + file I/O  │
          └─────────────┘
```

The important idea is:

> The processing classes perform transformations. The orchestrator decides which transformations to perform and in what order.

---

# 3. `Resampler`

The `Resampler` should remain responsible only for resampling.

Its job is essentially:

```text
Input audio
    |
    v
Calculate output positions
    |
    v
Lanczos interpolation
    |
    v
Resampled audio
```

For example:

```python
resampler = Resampler(audio_loader, duration_factor)
result_loader = resampler.get_resampled_audio()
```

It should not know why the caller wants to resample.

It does not need to know:

* whether resampling is being used for pitch shifting
* whether a phase vocoder will run afterward
* where the final file will be written

Its responsibility is simply:

> Given an input signal and a duration factor, produce the resampled signal.

---

# 4. `VocoderProcessor`

`VocoderProcessor` should contain the actual phase-vocoder processing.

The important public operation is:

```python
vocoder_process(
    speed_factor,
    pitch_factor=1.0
)
```

Conceptually:

```text
AudioLoader
    |
    v
STFT
    |
    v
Modify magnitude / phase
    |
    v
ISTFT
    |
    v
numpy audio array
```

This class should not need to know about the larger pitch-shifting recipe.

In particular, `VocoderProcessor` should not be responsible for deciding:

```text
"Should I resample first?"
"How many semitones did the user request?"
"Should I create a temporary AudioLoader?"
"Should I write the result to disk?"
```

Those are workflow decisions.

The processor should answer:

> Given this audio and these phase-vocoder parameters, how do I perform the transformation?

---

# 5. `VocoderOrchestrator`

The new class coordinates the complete operation.

For example:

```python
class VocoderOrchestrator:

    def __init__(self, audio_loader, stft_processor):
        self.audio_loader = audio_loader
        self.stft_processor = stft_processor
```

Its accurate pitch-processing method can contain the existing recipe:

```text
Requested semitone shift
          |
          v
Calculate pitch_time_stretch
          |
          v
      Resampler
          |
          v
Temporary AudioLoader
          |
          v
Calculate true_speed_factor
          |
          +---------------------------+
          |                           |
          | true_speed ≈ 1            | otherwise
          v                           v
Return resampled               VocoderProcessor
audio directly                      |
                                    v
                                  Result
```

The orchestrator is therefore responsible for the **composition of algorithms**.

---

# 6. Keep the trivial-case optimizations

The existing optimizations are useful and should remain.

## Case 1: No pitch shift

If:

```python
semitone_shift == 0
```

then:

$$
2^{-0/12}=1
$$

so there is no pitch-changing resampling step.

Therefore:

```text
semitone_shift == 0
        |
        v
Directly run phase vocoder
```

In code:

```python
if semitone_shift == 0:
    return self.vocoder_process(speed_factor)
```

This avoids:

* creating a temporary `AudioLoader`
* running the resampler
* performing unnecessary interpolation

The optimization belongs in the orchestrator because it is a **pipeline decision**.

---

## Case 2: Resampling already gives the desired duration

After resampling:

```python
true_speed_factor = pitch_time_stretch * speed_factor
```

If:

```python
abs(true_speed_factor - 1) < tolerance
```

then the resampled audio already has the desired final duration.

Therefore:

```text
Resampled audio
      |
      v
Already correct duration
      |
      v
Do NOT run phase vocoder
```

Return the resampled audio directly.

This optimization also belongs in the orchestrator because it decides whether another processing stage is necessary.

---

# 7. Why `vocoder_accurate_process()` belongs in the orchestrator

The current method:

```python
vocoder_accurate_process(
    speed_factor,
    semitone_shift
)
```

is really describing a **workflow**, not a single vocoder algorithm.

Its logic is:

```text
1. Convert semitones → pitch ratio
2. Resample
3. Calculate resulting speed requirement
4. Decide whether phase-vocoder processing is necessary
5. Run phase vocoder if necessary
6. Return final audio
```

That is orchestration.

The actual phase-vocoder algorithm remains:

```python
vocoder_process(speed_factor)
```

This distinction makes the names and responsibilities much clearer.

---

# 8. Writing the output to disk

Writing the final result to disk should also be handled outside `VocoderProcessor`.

A clean interface would be something like:

```python
def process_and_write(
    self,
    output_path: str,
    speed_factor: float,
    semitone_shift: float
) -> None:
    ...
```

The orchestrator can:

1. Determine which processing pipeline is necessary.
2. Run the required processing stages.
3. Obtain the final audio array.
4. Write the final result to disk.

Conceptually:

```text
User request
    |
    v
VocoderOrchestrator
    |
    +--> Resampler
    |
    +--> VocoderProcessor
    |
    v
Final numpy array
    |
    v
Audio I/O
    |
    v
output.wav
```

This keeps the signal-processing code independent from the output destination.

---

# 9. Why not put file writing inside `VocoderProcessor`?

If `VocoderProcessor` starts handling file output, it gradually becomes responsible for unrelated concerns:

```python
processor.vocoder_process(...)
processor.save(...)
processor.resample(...)
processor.pitch_shift(...)
```

Now the class knows about:

* STFT
* phase processing
* resampling
* pitch-shifting workflow
* file paths
* output formats
* temporary resources

That makes the class harder to reason about and test.

Instead:

```text
Resampler
    -> resampling

VocoderProcessor
    -> phase-vocoder transformation

AudioLoader / Audio I/O
    -> loading and writing audio

VocoderOrchestrator
    -> combine the processing stages
```

Each component has a clearer responsibility.

---

# 10. Temporary `AudioLoader` cleanup

The orchestrator is also the appropriate place to manage the temporary resampled `AudioLoader`.

For the non-trivial case:

```text
Create temporary AudioLoader
        |
        v
Run VocoderProcessor
        |
        v
Obtain final numpy array
        |
        v
Unload temporary AudioLoader
        |
        v
Return final result
```

The important rule is:

> Do not unload the temporary loader until every processing stage that needs its audio has finished.

For the `true_speed_factor ≈ 1` optimization, there is an additional lifetime consideration.

If the returned array is directly owned by:

```python
resampled_audioloader.audio_data
```

then unloading the loader must not invalidate the array being returned.

If `unload()` sets `audio_data = None`, use an independent result array before unloading if necessary.

---

# 11. Recommended responsibility table

| Component             | Responsibility                  | Should NOT decide                 |
| --------------------- | ------------------------------- | --------------------------------- |
| `AudioLoader`         | Hold/load/write audio           | How audio should be transformed   |
| `Resampler`           | Resample using interpolation    | Why resampling is needed          |
| `VocoderProcessor`    | STFT + phase-vocoder processing | Which pipeline to run             |
| `VocoderOrchestrator` | Combine processing stages       | Low-level interpolation/STFT math |

The orchestrator becomes the **application-level coordinator**.

---

# 12. Recommended file structure

A clean project structure would be:

```text
src/
│
├── audio_loader.py
│
├── resampler.py
│
├── stft_processor.py
│
├── vocoder_processor.py
│
└── vocoder_orchestrator.py
```

With responsibilities:

```text
audio_loader.py
    └── AudioLoader
         └── audio data + audio I/O

resampler.py
    └── Resampler
         └── Lanczos / interpolation / resampling

stft_processor.py
    └── STFTProcessor
         └── STFT / ISTFT

vocoder_processor.py
    └── VocoderProcessor
         └── phase-vocoder transformation

vocoder_orchestrator.py
    └── VocoderOrchestrator
         ├── choose processing pipeline
         ├── coordinate Resampler
         ├── coordinate VocoderProcessor
         ├── preserve trivial-case optimizations
         ├── manage temporary resources
         └── write final output
```

---

# 13. Example high-level API

The user-facing workflow can eventually look like:

```python
orchestrator = VocoderOrchestrator(audio_loader)

orchestrator.process_and_write(
    output_path="output.wav",
    speed_factor=1.0,
    semitone_shift=5
)
```

The orchestrator internally handles:

```text
semitone shift = +5
        |
        v
pitch_time_stretch = 2^(-5/12)
        |
        v
Resampler
        |
        v
true_speed_factor
        |
        +---- approximately 1? ---- yes --> return resampled result
        |
        no
        |
        v
VocoderProcessor
        |
        v
Final audio
        |
        v
Write to disk
```

The caller does not need to know how the pipeline is implemented.

---

# 14. Why this is useful for future changes

Suppose you later want to add another processing pipeline:

```text
Input
  |
  +--> Resampler
  |
  +--> Noise reduction
  |
  +--> VocoderProcessor
  |
  +--> Normalizer
  |
  v
Output
```

You do not need to make `VocoderProcessor` understand all of these operations.

The orchestrator can coordinate them.

Similarly, if the resampling implementation changes from Lanczos to another interpolation method, the `VocoderProcessor` does not need to change.

If the phase-vocoder implementation changes, the orchestrator does not need to know the internal details.

---

# 15. The core design principle

The most important distinction is:

```text
             "How do I transform audio?"
                         |
                         v
                Processing classes

          Resampler
          VocoderProcessor
          STFTProcessor
```

versus:

```text
             "Which transformations
              should I perform?"
                         |
                         v
                    Orchestrator
```

`VocoderProcessor` should know **how to perform a phase-vocoder operation**.

`VocoderOrchestrator` should know **that accurate pitch shifting requires resampling followed by a compensating phase-vocoder time adjustment**.

That separation preserves the existing mathematics while making the architecture easier to extend, test, and maintain.

---

# 16. Final mental model

Think of the classes as layers:

```text
┌──────────────────────────────────────────┐
│             Application Layer             │
│                                          │
│        VocoderOrchestrator               │
│        "What should happen?"             │
└────────────────────┬─────────────────────┘
                     |
                     v
┌──────────────────────────────────────────┐
│             Processing Layer              │
│                                          │
│   Resampler     VocoderProcessor         │
│   "How is each transformation done?"     │
└────────────────────┬─────────────────────┘
                     |
                     v
┌──────────────────────────────────────────┐
│             Audio / I/O Layer             │
│                                          │
│             AudioLoader                  │
│       "Where is the audio stored?"       │
└──────────────────────────────────────────┘
```

The refactor is therefore **not about changing the vocoder mathematics**.

It is about moving the multi-stage recipe and output workflow out of `VocoderProcessor` so that each class has one clear job.
