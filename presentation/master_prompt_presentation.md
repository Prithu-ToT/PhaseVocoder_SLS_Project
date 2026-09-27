# Master Prompt — CSE220 Phase Vocoder Voice Changer Presentation

## Role

Generate an editable Markdown storyboard for a presentation.

Do NOT generate Beamer, PowerPoint, or LaTeX. But you may later be asked to convert it into beamer

The output should be a slide-by-slide draft that is easy to edit manually.

Audience:
- CSE220 instructors
- Students familiar with DSP basics
- Presentation + viva afterwards

Total presentation target:
~6-8 minutes, including 4 minute *live demontration*

The first **3 minutes** are critical.

The audience must understand:

- what problem exists,
- what our software does,
- what changes in the audio,
- how to spot the changes visually (from spectrograph)

---

# Project Summary

Project:
**Phase Vocoder Voice Changer**

Core capability:

- independent playback speed control
- independent pitch shifting
- transient preservation for speech

The newest pipeline includes:

- STFT
- Phase Vocoder
- Lanczos resampling
- Spectral-flux transient detection
- Phase reset during transients

Verify the pipeline description consistent from vocoder_orchestrator.py file

---

# Global Presentation Rules

1. Markdown only.
2. Each slide starts with:

```md
## Slide X — Title
```

3. Each slide contains:

- Goal
- Visual Suggestion
- High level "how" behind the process

4. Speaker notes should be conversational.
---
**KEEP IN MIND DSP SLIDES ARE NOT FOR SPEAKING/PRESENTING, THEY ARE FOR ANSWERING QUESTIONS ASKED BY INSTRUCTORS**
5. Every DSP slide must clearly state:

- Input
- Output
- Algorithm
- One important equation
---
6. Avoid unnecessary theory before introducing the product.

7. The first minute should make Input vs Output visually obvious.

---

# Multi-Agent Workflow

Execute the following agents sequentially.

Each agent only produces its assigned section.

---

## Agent 1 — Hook & Problem

### Goal

Create the first 3 slides.

Must achieve:

Within the first minute the audience understands:

Input:
- original speech

Output:
- faster/slower
- higher/lower pitch
- natural sounding

### Must include

- before/after spectrogram
- feature list

### Avoid

DSP theory.

### Completion condition

The audience could explain the project after only these slides.

---

## Agent 2 — Solution Overview

Create one high-level pipeline slide.

Allowed complexity:
- very low.

Pipeline should look like:

Audio

↓

STFT

↓

Phase Vocoder

↓

Transient Correction

↓

Resampling

↓

Output

Do not explain internals yet.

Completion condition:
The audience understands the order of processing.

---

## Agent 3 — Processing Pipeline

Create one slide for each processing stage.

Each slide must follow the same template. This slides will *only* be referenced during QnA 

### Template

Input

Output

Algorithm

Key Equation

Why this step exists

Visual suggestion

Viva note

### Required stages

1. Audio Loading
2. STFT
3. Phase Vocoder
4. Transient Detection & Phase Reset
5. Lanczos Resampling
6. Final Output

Completion condition:
Each stage can be understood independently.

---

## Agent 4 — Trade-offs discussion

This is a high-priority slide.

Explain:

Key tradeoff:
- frequency resolution vs time resolution (window size)
- computation vs capability (higher overlap allows more pitch shifts)

Implemented:
- slider for choosing stft window size in 512 to 8192 
- slider for choosing overlap ratio from 50% to 90%

Decision:
Voice Processing : window size = 512-1024, else transient smear
Music Processing : window size = 2048-4096, else music sounds off tune

Limit:
- Compund time stretching of 0.25x to 3x is allowed
- independent pitch shift of $\pm$ 12 semitone (no audio artifacts upto $\pm$ 9)
- independent time scaling from 0.5x to 2x 

Completion condition:
The audience understands why transient correction exists.

---


## Agent 5 — Transient Preservation

This is a high-priority slide.

Explain:

Problem:
Consonants become smeared.

Detection:
Spectral Flux

Equation:

F_t = Σ max(0, |X_t|-|X_{t-1}|)

Decision:
Flux > threshold.

Action:
Reset synthesis phase.

Mention:
3-frame refractory.

Visual:

- smeared consonant
- corrected consonant

Completion condition:
The audience understands why transient correction exists.


---

## Agent 6 — Resampling

Explain:

Why linear interpolation failed.

Mention:

- high-frequency artifacts
- Lanczos sinc approximation
- better preservation

Completion condition:
The tradeoff is clear.

---

## Agent 7 — Results

Create result slides.

Must include:

- speed-only
- pitch-only
- combined
- spectrogram comparisons

Mention expected observations.

Completion condition:
Features are visibly verified.

---

## Agent 8 — Evaluation

Create feature checklist.

Use a table.

Example:

| Feature | Implemented |
|----------|-------------|
| Speed Control | ✔ |
| Pitch Shift | ✔ |
| Independent Control | ✔ |
| Formant Preservation | ✔ |
| Transient Preservation | ✔ |
| Stereo Support | ✔ |

This slide should satisfy marking rubrics.

Completion condition:
A grader can verify features quickly.

---

## Agent 9 — Appendix

Create viva-only slides.

One slide per topic.

Suggested topics:

- STFT overlap-add
- Phase accumulation
- Why Hann window
- Why Lanczos
- Spectral Flux threshold
- Gaussian sigma
- Phase reset
- Computational complexity
- WOLA normalization

Keep these concise.

Completion condition:
Each appendix slide answers one likely theory question.

---

# Visual Asset Suggestions

Whenever a slide requests a visual, describe exactly what should be drawn. Put an image count marker with it. If it is a diagram better designed though beamer, flag that too

Examples:

- waveform comparison (IMG_001)
- spectrogram comparison (IMG_002)
- pipeline diagram (Beamer_Diagram_1)
- phase accumulation illustration (Beamer_Diagram_2)
- Gaussian envelope smoothing
- transient phase reset timeline

Do not generate images.

---

# Timing Targets

| Section | Time |
|----------|------|
| Problem | 40 sec |
| Product Overview | 1 min 20 sec |
| Pipeline | 1 min |
| Tradeoffs | 1 min | 
| Core Processing | Only for QnA |
| Results | During live demo |
| Conclusion | 30 sec |
| Viva Appendix | backup |

---

# Quality Checklist

Before finishing, verify:

- [ ] Input/output obvious within first minute.
- [ ] Feature list appears early.
- [ ] Transient correction replaces old pipeline.
- [ ] Only one high-level pipeline slide.
- [ ] Each processing stage has Input/Output/Algorithm/Equation.
- [ ] Speaker notes are conversational.
- [ ] Markdown is easy to edit.
- [ ] Appendix exists for theory questions.