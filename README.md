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

## 4. Files

| file | what |
| --- | --- |
| `splat_field.py` | The joined machine: numpy field core (the certified update), torch only at the model boundary, live app (keys k/d/m/a/z/+/−/t/y), per-keyframe power-iteration i_crit, `--full_eig` eigenmode gallery + EG2, drive-cut decay recorder + `--score_decay` RF1 scorer, `--selftest` (3/3 [V], no model or camera needed). |
| `gabor_loop.py` | The autonomous half alone, torch, spectrum readout in the header. |
| `gabor_loop_verify.py` | The numpy twin that certified the loop laws, 5/5 registered. |

Model bridge assumes `splat_trainer3v2.SplatVAE` importable next to the script (the
checkpoint stores `image_size`, `num_packets`, `sd`).

---

## 5. Honest revisions (chronological, kept forever)

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

---

## 6. Provenance and boundaries

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
