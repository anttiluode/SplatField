# SplatField

### A driven-dissipative wave medium built from a face model's own atoms — with its complete rulebook

**PerceptionLab | Antti Luode | Helsinki**
> **Do not hype. Do not lie. Just show.**

---

## 0. What this is

Take a trained TinyAvatar / SplatWorld checkpoint. Its decoder maps a latent to a few
hundred Gabor wave packets whose interference *is* the image. As shipped, that is half a
machine: a synthesis half. Every frame is conjured from scratch; the field has no state,
no memory, no law of motion.

This repo is the other half joined on. One keyframe hands the field its basis — the
decoder's own 256 trained packets, geometry frozen — and from then on the field owns the
coefficients under one update:

```
r(t)   = (1−α)·( sigmoid(render(c)) − ½ )  +  α·( I_cam − ½ )
c(t+1) = leak · c(t) + inject · e^{iδ} · Proj( r(t) )
```

α = 1 is a matched-filter tracker. α = 0 is an autonomous field. In between, the camera
is evidence and the field is prior, and the balance point is not a vibe — it is an
eigenvalue.

**What are we looking at?** A discrete field theory in which every microscopic constant
is known. Not an analogy to physics: an actual small physics, with atoms (Gabor packets),
a superposition principle (the render), a dissipation (leak), a coupling (the Gram
matrix), a drive (the camera), and a complete spectral solution of its linear regime.
Unlike any natural medium, nothing here is hidden — the entire system fits in one file
and every law below was checked against ground truth we own.

**What this is not:** biology, and not a resolution fix. 96 px is a basis constraint; a
recurrent field cannot add spatial detail the basis does not carry. The field's territory
is time: memory, persistence, smoothing, bloom.

---

## 1. The laws (certification status attached)

**Spectral loop law [V].** The loop's Jacobian at the origin, for any projection built
from the synthesis adjoint (matched filter included), is similar to a symmetric PSD
matrix. Therefore all its eigenvalues are real and non-negative — a theorem, verified to
machine zero on random and trained bases — and the closed field has exactly two regimes:
**quiescent** (spectral radius < 1) and **static pattern** (radius > 1, arrested by the
sigmoid). Autonomous rotation is impossible without breaking the adjoint symmetry (the
twist e^{iδ} creates it, with spectral onset). Because λ = leak + inject·μ is affine in
inject, one eigendecomposition yields the whole (leak, inject) phase diagram, including
the critical inject — and one power iteration per keyframe is enough to place the
operating point live. Registered runs on both sides of the predicted boundary landed
thirteen orders of magnitude apart (`gabor_loop_verify.py`, 5/5; reproduced in torch on
an independently seeded basis, each spectrum predicting its own loop).

**Two time constants [V].** Memory and responsiveness are separate knobs. With the drive
cut, the field decays at the loop radius ρ: motion energy falls at exactly 2·ln ρ per
step (selftest ratio 1.14 of predicted). At α = 1 the self-term drops out and the rise
pole is the *leak* alone: 1/e time −1/ln(leak), measured 19 steps vs 19.5 predicted.
First registration wrote ρ for both; the selftest killed that in one run. ρ sets how long
the field remembers; leak sets how fast it listens.

**DC law [V].** sigmoid(u) − ½ = ½·tanh(u/2) is odd, so the dark field is an *exact*
fixed point (stays at literal 0.0). Without the subtraction the field accumulates a bias
state matching the linear fixed-point prediction to ratio 0.97.

**Neighbors it plugs into** (certified elsewhere, cited not re-measured): the **fire
law** — rendered crossfade amplitude loss predicted to ~2% from endpoint phases alone
(TinyAvatar, RP1–RP3 all-V) — and the **dispersion law** dΦₖ = −2π fₖ (uₖ·v), certified
on real head motion at slope 0.998, r 0.81, 16,401 pairs, per-packet scramble control
(`phase_orbit.py` v2, all-V). Interpolation, driven motion, autonomy: three behavior
classes, three design equations, one medium.

---

## 2. The observation this repo exists to explain: two worlds

First live run, CelebA checkpoint, subcritical, α ≈ 0.35. What was seen, verbatim from
the lab notes: *"the splats go missing. the field dominates. it starts with face, decays
to field, finds face like things."*

Here is the precise version of that sentence. There are two worlds in this machine and
they obey different rulers:

- **The data world.** The decoder's world — the statistics of 200k faces, the manifold,
  the encoder's opinion of what your webcam shows. This world chose where the 256 packets
  sit, their frequencies, their orientations.
- **The Gram world.** The field's world. Once the basis is frozen, the dynamics see
  *nothing* but the overlap geometry of those packets — the Gram matrix. Not the dataset,
  not the encoder, not the manifold. The field does not care about the machinery it was
  made by. Its rules are different.

A keyframe drops a data-world state (an exact face) into the Gram world. The loop then
does the only thing a linear-regime field can do: it decomposes that state over its own
eigenmodes and damps them at their own rates. Mid-rank modes — the detail — die fastest;
what persists is the content lying along the top eigenmodes. "Starts with face, decays to
field, finds face-like things" is a literal description of a state vector relaxing onto
the slow modes of the loop operator. The blurry ghosts the field settles into are those
modes, rendered.

**Why "the splats go missing" [~], with a gate.** High-frequency packets overlap their
neighbors weakly (Gram off-diagonals fall as a Gaussian in wavevector mismatch), so they
sit in weakly-coupled, low-eigenvalue positions and damp fastest at fixed inject; the
low-frequency packets pool into the collective top modes. Detail dies first, gist
persists — the core/ghost shell structure of the resolution horizon, and the coarse/fine
split of the earlier band experiments, now appearing as a *dynamical* ordering rather
than a training observation. This mechanism is registered, not assumed: **RF1** — during
autonomous decay, per-packet decay rate correlates with packet frequency, Pearson
r ≥ +0.5 (recorded automatically when the drive is cut; scored with
`--score_decay decay_c.npz`; sigma correlation printed as confound check).

**The question the two worlds pose.** Do they agree on anything? Concretely: are the
field's preferred patterns — its top eigenmodes — the same objects as the *eigenfaces* of
the medium's own outputs? If yes, the 1991 eigenfaces re-derive as the **dynamical normal
modes** of a face-atom field: not statistics, resonance. If no, the worlds are genuinely
separate and a keyframe is a foreign object the field merely tolerates. That is **EG2**
(registered below), and either answer is a finding.

---

## 3. Ledger

| tag | claim |
| --- | --- |
| **[V]** | Spectral loop law: real-spectrum theorem, two regimes, affine phase diagram, 13-orders regime separation; reproduced across numpy/torch on independent bases. |
| **[V]** | Two time constants: decay at 2·ln ρ (drive cut), rise at −1/ln leak (α=1); selftest 3/3. |
| **[V]** | DC law: exact dark fixed point with the −½; bias fixed point matches linear prediction 0.97 without it. |
| **[V]** | (cited) Fire law, TinyAvatar RP1–RP3; dispersion law, phase_orbit v2 all-V on real head motion. |
| **[~]** | Two-worlds reading of the first live run: relaxation onto loop eigenmodes. Consistent with everything measured; RF1 and EG2 are its teeth. |
| **[~ open]** | **RF1** detail-dies-first: corr(per-packet decay rate, frequency) ≥ +0.5. Instrumented, not yet run. |
| **[~ open]** | **EG2** eigenmodes-vs-eigenfaces: mean max-cosine of top-4 loop modes against top-8 prior-sample eigenfaces ≥ 2× a pixel-scrambled null. Instrumented, not yet run. |
| **[~ open]** | **F1/F2/F4** live gates: persistence decay within ±15% of 2·ln ρ; α=1 rise within ±25% of −1/ln leak; FIELD beats DIRECT on floater count at equal fit (the bloom bet). Scoreable from `field_log.csv`. |
| **[K]** | **EG1** as registered: in/out face-box energy ratio, median 0.95 vs gate 2.0 on the CelebA keyframe. Gate failed and stays failed. Diagnosis: the metric, not necessarily the claim — CelebA faces fill the frame, so a center-box contrast has nothing to measure even when the rendered modes are visibly head-shaped. EG2 is the sharp replacement; EG1's number is kept as the record of a weak instrument. |
| **[K open]** | Saturated limit-cycle frequency under twist: three registered predictions failed one-signed (measured below linear in all three). Carried from gabor_loop, untouched here. |

---

## 4. What is in this repo, and what each piece was built to figure out

| file | what it is | the question it was built to answer |
| --- | --- | --- |
| `gabor_loop_verify.py` | numpy twin of the closed loop, **executed** with predictions P1–P5 registered before the runs | *Is a closed Gabor field predictable from its spectrum before it runs?* Answer: 5/5 [V], plus the no-rotation theorem falling out of the Jacobian's symmetry. |
| `gabor_loop.py` | PyTorch version of the same loop, spectrum readout in the header, `--twist` / `--gate` / `--dc_check` | *Does the method transfer to GPU and a basis it never saw?* Yes — torch i_crit 0.0433 vs numpy 0.0363 are different seeded bases, and each spectrum predicted its own loop. Also carries the one open problem: the saturated limit-cycle frequency under twist (three one-signed misses, documented in the header). |
| `gabor_loop_log.csv` | the executed active-regime run, 1200 steps of `t, motion_energy, omega_w, max_abs_z` | The instrument data. Whoever attacks the open frequency problem starts here. |
| `splat_field.py` | the joined machine — numpy field core (the certified update), torch only at the model boundary, live app | *What does the field do when its basis is trained and its drive is a camera?* Produced the two-worlds observation of §2. **Note:** the repo currently carries v1; v2 adds the EG2 eigenface comparison and the RF1 drive-cut decay recorder/scorer described in §2 — replace with v2 when merging. |
| `tiny_avatar3.py` | the TINY AVATAR studio (Dataset Prep / Training / Avatar Driver tabs) and, for this repo's purposes, the `FaceFramer` | *Can the whole pipeline live in one app?* Here it plays a supporting role: `splat_field.py` imports its FaceFramer so the live crop matches the training crop (the framing mismatch is a measured artifact source). Needs PyQt6 only if actually imported; splat_field degrades to a Haar/center-crop fallback with a warning otherwise. |
| `model2.pt` | the 96 px / 256-packet CelebA checkpoint (`sd`, `image_size`, `num_packets`), 23 MB, shipped | So every experiment in this repo reproduces on a fresh clone. This is the checkpoint all the trained-basis numbers were measured on. |
| `eigenmodes.png` | the top-8 loop eigenmodes of a trained keyframe basis, rendered | The "face-like things" the field decays into, made visible. EG1's failed metric number (median 0.95) is attached to this image in the ledger; EG2 is the sharp question it raises. |
| `splat_trainer3v2.py` | **not in this repo — fetch from [TinyAvatar](https://github.com/anttiluode/TinyAvatar)** and place next to `splat_field.py` | The model bridge imports it to instantiate `SplatVAE` and load the checkpoint. Without it, only `--selftest` and `--mock` run. |

---

## 5. How to run the system

**Install** (Python 3.10+):

```
pip install numpy opencv-contrib-python torch
pip install PyQt6            # optional — only for tiny_avatar3 / FaceFramer
```

Copy `splat_trainer3v2.py` from the TinyAvatar repo into this folder (see table above).
`opencv-contrib-python` matters on Windows Store Python: the plain wheel there ships
without the Haar data file and face framing silently degrades to center-crop.

**Run order** — each step certifies the ground the next one stands on:

```
# 1. Field core, no model, no camera. Registered gates T1–T3, expect 3/3 [V].
python splat_field.py --selftest

# 2. Re-run the loop certification (P1–P5, numpy, ~a minute on CPU).
python gabor_loop_verify.py

# 3. The autonomous loop on GPU; add --twist 2.2 to see created rotation.
python gabor_loop.py
python gabor_loop.py --twist 2.2

# 4. Trained basis, offline: theorem carry-over (F3), eigenmode gallery,
#    EG1 (expect [K], the weak metric) and EG2 (the open question).
python splat_field.py --model model2.pt --full_eig

# 5. Live.
python splat_field.py --model model2.pt
```

**Live keys:** `k` new keyframe (re-freezes the basis from your face, re-estimates
i_crit) · `d` drive cut (the F1 persistence measurement; toggling it OFF saves
`decay_c.npz`) · `m` FIELD / DIRECT A/B (the F4 bloom bet, scored from
`field_log.csv`) · `a`/`z` α up/down · `+`/`-` inject as a fraction of i_crit ·
`t`/`y` twist (leaves the certified regime; treat as demo) · `space` pause · `q` quit.

**Knobs:** `--leak` sets responsiveness (α=1 rise time is −1/ln leak steps);
`--frac` sets inject as a fraction of the basis's own critical value — below 1 the
field is a memory with decay 2·ln ρ per step, above 1 it holds a pattern; `--alpha`
sets evidence vs prior.

**The F1/RF1 procedure**, concretely: run live at `--frac 0.9`, let the field settle
on your face, press `d`, sit still for ~500 steps watching the face decay into the
eigenmode ghosts, press `d` again. Then:

```
python splat_field.py --score_decay decay_c.npz
```

which scores RF1 (does detail die first, in the registered sense) and prints the
sigma confound. F1 itself is the decay-rate-vs-2·ln ρ comparison from `field_log.csv`
over the same window.

---

## 6. Honest revisions (chronological, kept forever)

1. **F2 pole**: first registered as ρ; selftest measured 19 steps against a predicted
   199 and the formula was wrong, not the field — at α=1 the self-term is gone and the
   pole is leak. Fixed before any live run. Gift: memory depth and responsiveness are
   independent dials.
2. **EG1 metric demoted**: gate [K] as registered (median 0.95); diagnosed as an
   instrument mismatched to frame-filling faces; replaced by EG2 with a proper null. The
   failed number stays in the ledger.
3. **numpy 2.x**: `ndarray.ptp` removed upstream → `np.ptp(img)`.
4. **Windows Store Python** ships cv2 without the Haar data file → cascade load now
   guarded; falls back to center-crop **with a printed warning**, because center-crop
   reintroduces the train/live framing mismatch that the FaceFramer exists to fix.
5. **Repo v1/v2 drift**: the first push carried splat_field v1 (pre-EG2, pre-RF1).
   The numbers in §2 that depend on those instruments are *registered, unmeasured*
   until v2 lands and runs; nothing in the ledger claims otherwise.

---

## 7. Provenance and boundaries

The loop laws were certified in executed numpy with thresholds registered before the
runs; the torch version reproduced them on an independently seeded basis (different
i_crit, each spectrum predicting its own loop — the stronger statement). The fire and
dispersion laws are TinyAvatar's and are cited, not re-measured. The two-worlds section
is an interpretation with two registered experiments attached; until RF1 and EG2 run it
is a reading, and a reviewer is free to prefer a duller one. Nothing here is a claim
about biological systems. Nothing here beats mainstream avatar or generative tooling at
its own game; the point is different — this is a medium small enough to know completely
and now predictable enough to engineer.

*Helsinki, July 2026.*

*Do not hype. Do not lie. Just show.*
