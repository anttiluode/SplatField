#!/usr/bin/env python3
"""
splat_field.py — TinyAvatar with the other half.

The shipped TinyAvatar is one half of a wave medium: a synthesis half.
Webcam -> encoder -> z -> decoder -> packets -> interference -> image.
Every frame is decoded from scratch; the field has no state, no memory,
no dynamics of its own. gabor_loop.py built the missing half in
isolation: a recurrent field z(t+1) = leak*z + inject*Proj(render-0.5)
whose entire behavior is readable from one spectrum (verdict [V],
5/5 registered, numpy twin executed).

This file joins them. The decoder supplies the basis (a keyframe's 256
trained packets, geometry frozen); the field loop owns the coefficients;
the webcam becomes a DRIVE mixed into the self-injection:

    r(t)   = (1-alpha) * ( sigmoid(render(c)) - 0.5 )      # self half
           +    alpha  * ( I_cam - 0.5 )                   # world half
    c(t+1) = leak * c(t) + inject * e^{i*twist} * Proj( r(t) )

alpha=1  : matched-filter tracker with a one-pole memory (leaky decode)
alpha=0  : the autonomous loop of gabor_loop.py on the TRAINED basis
between  : evidence vs prior, placed by eigenvalue instead of by vibe.

Operating point is set as a fraction of the critical inject i_crit,
estimated per keyframe by power iteration on the probed Jacobian
(no full eig needed live; --full_eig does the 2N x 2N probe offline
and dumps the top eigenmode RENDERS — the eigenmode gallery).

WHAT THIS DOES NOT FIX (stated up front): resolution. 96 px is a VRAM
and training constraint; a recurrent field cannot add spatial detail
that the basis does not carry. What the field CAN address is temporal:
bloom/floaters (F4), jitter, dropout-bridging, persistence (F1).

REGISTERED CLAIMS for the live run (thresholds fixed here, scored
from field_log.csv; [V]/[K] only after you run it):

 F1 PERSISTENCE  at subcritical operating point (0.9 x i_crit,
    alpha ~ 0.3), cut the drive (key d). Post-cut field-motion energy
    m(t) decays log-linearly at rate 2*ln(rho_pred) within +/-15%
    (rho_pred printed at keyframe time). The held face is the field
    remembering, on an eigenvalue schedule — not a freeze-frame.
 F2 TRACKING LAG at alpha=1 the self-term drops out and the loop is
    exponential smoothing of matched-filter coefficients with pole =
    leak: step-response 1/e time = -1/ln(leak) steps within +/-25%.
    (First registration wrote -1/ln(rho); the selftest killed it in one
    run — rho governs the alpha->0 decay (F1), leak governs the alpha=1
    rise. Honest revision, made before any live run.)
    Further revisions after the first live run: EG1 metric demoted (see
    EG2); ndarray.ptp -> np.ptp (numpy 2.x removed the method); Haar
    cascade path guarded (Windows Store Python ships cv2 without the
    data file — falls back to center-crop WITH a printed warning, since
    center-crop reintroduces the train/live framing mismatch).
 F3 THEOREM CARRY-OVER on the trained basis, twist=0 Jacobian still has
    max|Im mu| ~ 0 (adjoint symmetry is basis-independent — this is a
    check, not a bet; a violation means a wiring bug, not physics).
 F4 BLOOM A/B  toggle m between DIRECT (per-frame decode, the shipped
    app) and FIELD (this loop) under identical drive. Floater count
    (connected components of |render residual| above tau outside the
    face box, same metric family as live_cortex_gated) is lower in
    FIELD mode at equal fit error. This is the practical payoff bet:
    the field as a temporal regularizer. Genuinely open — register
    before running.

 RF1 DETAIL-DIES-FIRST during autonomous decay (drive cut), per-packet
    amplitude decay rate correlates with spatial frequency, r >= +0.5
    (mechanism [~] behind "the splats go missing": high-f packets sit in
    low-eigenrank, weakly coupled Gram positions and damp fastest; gist
    persists in the collective low-f top modes). Recorded automatically
    while drive is cut; scored offline with --score_decay decay_c.npz.
 EG2 EIGENMODES = EIGENFACES (replaces EG1's metric, which returned [K]
    with median ratio ~0.95 on the CelebA keyframe — diagnosed weak, not
    wrong: CelebA faces FILL the frame, so an in/out box ratio has no
    contrast to measure even when the modes are visibly head-shaped).
    EG2 asks the sharp version: cosine-match top loop eigenmodes against
    PCA eigenfaces of the medium's own prior samples, vs a scrambled
    null. [V] = the data world and the Gram world agree on face-modes.
    [K] = two worlds. Either is a finding.

 OPEN, carried from gabor_loop: saturated limit-cycle frequency under
 twist (three one-signed misses). Untouched here; twist defaults 0.

EIGENMODE GALLERY (--full_eig): registered prediction EG1 — on the
TRAINED basis the top Jacobian eigenmodes render face-structured
(eyes/mouth/contour energy inside the face box > 2x outside), where
the random basis of gabor_loop renders blobs. If EG1 fails, the
supercritical attractor is generic and "the medium wants faces" is
dead for this checkpoint; either answer is a result.

TRAINER: untouched in v1, deliberately. "Train THROUGH the loop"
(unroll k field steps in the trainer so the decoder learns coefficients
whose closed loop is stable/face-shaped) is a named future bet — run
the eigenmode gallery first; it tells you whether that training is
even needed.

USAGE
  python splat_field.py --selftest                # no model, no camera
  python splat_field.py --model model2.pt         # live (needs torch)
  python splat_field.py --model model2.pt --full_eig   # gallery + i_crit
  python splat_field.py --mock                    # live UI, synthetic basis

KEYS (live)
  k  new keyframe (re-freeze basis from current frame, re-estimate rho)
  d  drive cut toggle (alpha -> 0; F1 measurement window)
  m  DIRECT / FIELD A/B toggle (F4)
  a/z  alpha up/down     +/-  inject as fraction of i_crit
  t/y  twist up/down (leaves certified regime; label goes [demo])
  q  quit                 space  pause

The field core below is numpy and is the SAME update certified in
gabor_loop_verify.py. Torch appears only at the model boundary
(encoder forward, decoder forward). If tiny_avatar3.py is importable
its FaceFramer and loader are reused; otherwise minimal fallbacks.

Do not hype. Do not lie. Just show.
"""
import argparse, math, os, sys, time
import numpy as np

# ----------------------------------------------------------------- basis
def basis_from_packets(pk, S):
    """pk: dict with px,py,sigma,theta,freq arrays (N,) in [0,1] image
    coords / radians / cycles-per-unit. Returns G (N, S*S) complex128."""
    N = len(pk["px"])
    xs = (np.arange(S) + 0.5) / S
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    G = np.zeros((N, S * S), complex)
    for k in range(N):
        dx, dy = X - pk["px"][k], Y - pk["py"][k]
        xr = dx * np.cos(pk["theta"][k]) + dy * np.sin(pk["theta"][k])
        env = np.exp(-(dx * dx + dy * dy) / (2 * pk["sigma"][k] ** 2 + 1e-12))
        G[k] = (env * np.exp(1j * 2 * np.pi * pk["freq"][k] * xr)).ravel()
    return G

def random_packets(N, seed):
    r = np.random.default_rng(seed)
    return dict(px=r.uniform(.15, .85, N), py=r.uniform(.15, .85, N),
                sigma=r.uniform(.04, .12, N), theta=r.uniform(-np.pi, np.pi, N),
                freq=r.uniform(3., 8., N))

# ------------------------------------------------------------- field core
def sigmoid(u): return 1.0 / (1.0 + np.exp(-u))

class Field:
    """The certified update, per color channel, geometry frozen.
    c: (N, CH) complex. G: (N, S*S). All numpy."""
    def __init__(self, G, leak=0.95, frac=0.9, twist=0.0, ch=3):
        self.G, self.leak, self.twist = G, leak, twist
        self.n2 = (np.abs(G) ** 2).sum(1)
        self.N, self.ch = G.shape[0], ch
        self.c = np.zeros((self.N, ch), complex)
        self.mu_max = self.power_mu()                 # dominant Jacobian eig
        self.i_crit = (1.0 - leak) / max(self.mu_max, 1e-12)
        self.inject = frac * self.i_crit
        lam = leak + self.inject * self.mu_max
        self.rho = abs(lam)                            # predicted radius

    # -- one linearized Jacobian application: K v = step(eps v)/eps at
    #    leak=0, inject=1, twist=0 (z=0 is exact fixed point, odd nl.)
    def _K(self, v, eps=1e-6):
        f = sigmoid(((eps * v) @ self.G).real) - 0.5
        return (self.G.conj() @ f.astype(complex)) / self.n2 / eps

    def power_mu(self, iters=60, seed=3):
        r = np.random.default_rng(seed)
        v = r.standard_normal(self.N) + 1j * r.standard_normal(self.N)
        v /= np.linalg.norm(v)
        mu = 0.0
        for _ in range(iters):
            w = self._K(v)
            mu = float(np.real(np.vdot(v, w)))         # eigs real >= 0 (thm)
            n = np.linalg.norm(w)
            if n < 1e-30: return 0.0
            v = w / n
        return abs(mu)

    def set_frac(self, frac):
        self.inject = frac * self.i_crit
        self.rho = abs(self.leak + self.inject * self.mu_max)

    def render(self, c=None):
        c = self.c if c is None else c
        F = (c.T @ self.G).real                        # (CH, S*S)
        return sigmoid(F)

    def project(self, r_centered):
        # r_centered: (CH, S*S), already minus 0.5
        return ((self.G.conj() @ r_centered.T) / self.n2[:, None])  # (N,CH)

    def step(self, drive=None, alpha=0.0):
        r_self = self.render() - 0.5
        r = (1 - alpha) * r_self if alpha < 1 else np.zeros_like(r_self)
        if drive is not None and alpha > 0:
            r = r + alpha * (drive - 0.5)
        tw = complex(math.cos(self.twist), math.sin(self.twist))
        self.c = self.leak * self.c + self.inject * tw * self.project(r)
        return self.c

# ------------------------------------------------------- full spectrum / EG
def full_jacobian(field, eps=1e-6):
    N = field.N
    K = np.zeros((2 * N, 2 * N))
    for k in range(N):
        for m, p in enumerate((1.0, 1j)):
            e = np.zeros(N, complex); e[k] = p
            dz = field._K(e, eps)
            K[:N, m * N + k] = dz.real
            K[N:, m * N + k] = dz.imag
    return K

def eigenmode_gallery(field, S, out="eigenmodes.png", top=8, facebox=None):
    K = full_jacobian(field)
    mu = np.linalg.eigvals(K)
    im_max = float(np.abs(mu.imag).max())
    w, V = np.linalg.eigh(0.5 * (K + K.T))             # symm part; eigs real
    order = np.argsort(-w)[:top]
    tiles, ratios = [], []
    for j in order:
        v = V[:N_of(field), j] + 1j * V[N_of(field):, j]
        img = (v[:, None].T @ field.G).real.reshape(S, S)
        img = (img - img.min()) / (np.ptp(img) + 1e-12)
        tiles.append(img)
        if facebox is not None:
            x0, y0, x1, y1 = facebox
            inb = np.abs(img[y0:y1, x0:x1] - img.mean()).mean()
            out_m = np.abs(img - img.mean()).mean() + 1e-12
            ratios.append(inb / out_m)
    grid = np.hstack(tiles)
    try:
        import cv2
        cv2.imwrite(out, (grid * 255).astype(np.uint8))
    except Exception:
        np.save(out + ".npy", grid)
    return im_max, w[order], ratios

def N_of(f): return f.N

# ------------------------------------------------------------- model bridge
def load_model_bridge(path):
    import torch
    import splat_trainer3v2 as ST
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=dev)
    
    # Instantiate the architecture from the trainer
    model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
    model.load_state_dict(ck["sd"])
    model.eval().to(dev)
    
    def encode(img01):
        # img01: HxWx3 float32 RGB in [0, 1]
        xt = torch.from_numpy(np.ascontiguousarray(img01.transpose(2, 0, 1)))[None].float().to(dev)
        with torch.no_grad():
            mu, _ = model.enc(xt)
        return mu

    def packets(z):
        with torch.no_grad():
            # decode parameters
            P = model.ren.activate(model.dec(z).float())
            px, py, sigma, theta, freq, coeff = P
            pk = dict(
                px=px[0].cpu().numpy(),
                py=py[0].cpu().numpy(),
                sigma=sigma[0].cpu().numpy(),
                theta=theta[0].cpu().numpy(),
                freq=freq[0].cpu().numpy()
            )
            # coeff is (1, N, 3, 2) or (1, N, 6) -> complex (N, 3)
            c = coeff[0].cpu().numpy()
            if c.ndim == 3: # (N, 3, 2)
                c_comp = c[..., 0] + 1j * c[..., 1]
            else: # (N, 6)
                cf = c.reshape(-1, 3, 2)
                c_comp = cf[..., 0] + 1j * cf[..., 1]
            return pk, c_comp

    return encode, packets, dev

# ---------------------------------------------------------------- selftest
def selftest():
    """T1 theorem, T2 persistence law, T3 tracking law — synthetic basis,
    thresholds registered in the module docstring. No torch, no camera."""
    N, S = 128, 64
    G = basis_from_packets(random_packets(N, 0), S)
    f = Field(G, leak=0.95, frac=0.9, ch=1)
    print(f"i_crit(power) {f.i_crit:.4f}  rho_pred {f.rho:.4f}")

    K = full_jacobian(f)
    mu = np.linalg.eigvals(K)
    t1 = float(np.abs(mu.imag).max())
    ok1 = t1 <= 1e-8
    print(f"[{'V' if ok1 else 'K'}] T1 theorem: max|Im mu| = {t1:.2e} (<=1e-8)")
    mu_top = float(np.max(mu.real))
    err = abs(mu_top - f.mu_max) / mu_top
    print(f"    power-iter vs full eig: mu {f.mu_max:.5f} vs {mu_top:.5f} "
          f"(rel err {err:.1e})")

    # T2: drive with a fixed target, cut, fit post-cut decay of m(t)
    r = np.random.default_rng(1)
    target = sigmoid((0.3 * (r.standard_normal(N) + 1j *
                     r.standard_normal(N))[None, :] @ G).real)
    m_hist = []
    Fp = f.render()
    for t in range(1500):
        f.step(drive=target if t < 400 else None,
               alpha=0.5 if t < 400 else 0.0)
        Fn = f.render()
        m_hist.append(float(((Fn - Fp) ** 2).mean())); Fp = Fn
    seg = np.log(np.array(m_hist[650:1100]) + 1e-300)
    slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    pred = 2 * math.log(f.rho)
    ratio = slope / pred
    ok2 = 0.85 <= ratio <= 1.15
    print(f"[{'V' if ok2 else 'K'}] T2 persistence: decay {slope:.5f}/step, "
          f"predicted 2ln(rho) {pred:.5f}, ratio {ratio:.3f} (in [0.85,1.15])")

    # T3: step response 1/e time at alpha=1 vs -1/ln(rho)
    f.c[:] = 0
    norms = []
    for t in range(2500):
        f.step(drive=target, alpha=1.0)
        norms.append(np.linalg.norm(f.c))
    cn = np.array(norms) / norms[-1]
    t63 = int(np.argmax(cn >= 1 - 1 / math.e))
    tpred = -1.0 / math.log(f.leak)   # alpha=1: pole is leak
    ratio3 = t63 / tpred
    ok3 = 0.75 <= ratio3 <= 1.25
    print(f"[{'V' if ok3 else 'K'}] T3 tracking: t(1/e) {t63} steps, "
          f"predicted {tpred:.1f}, ratio {ratio3:.3f} (in [0.75,1.25])")

    print("SELFTEST " + ("[V] field core matches the certified laws"
          if all([ok1, ok2, ok3]) else "[K] read the failing line"))
    return all([ok1, ok2, ok3])

def field_freq(f):  return np.asarray(f.pk["freq"]) if hasattr(f, "pk") else np.zeros(f.N)
def field_sigma(f): return np.asarray(f.pk["sigma"]) if hasattr(f, "pk") else np.zeros(f.N)

def score_decay(path):
    """RF1 (registered): during autonomous decay, per-packet amplitude
    decay RATE correlates with packet spatial frequency, Pearson r >= +0.5
    (detail dies first; gist persists — the eigenrank/spatial-frequency
    link stated as [~] in the README, scored here). Confound check: also
    prints corr(rate, sigma)."""
    d = np.load(path)
    amp, freq, sig = d["amp"], d["freq"], d["sigma"]
    T, N = amp.shape
    if T < 60:
        print(f"only {T} recorded steps; need >= 60 for a stable fit"); return
    keep = amp[0] > np.median(amp[0]) * 0.1
    rates = np.full(N, np.nan)
    tt = np.arange(T)
    for k in range(N):
        if keep[k] and amp[-1, k] > 0:
            rates[k] = np.polyfit(tt, np.log(amp[:, k] + 1e-30), 1)[0]
    m = np.isfinite(rates)
    r_f = float(np.corrcoef(-rates[m], freq[m])[0, 1])
    r_s = float(np.corrcoef(-rates[m], sig[m])[0, 1])
    ok = r_f >= 0.5
    print(f"[{'V' if ok else 'K'}] RF1 corr(decay rate, freq) = {r_f:+.3f} "
          f"(gate >= +0.50, {int(m.sum())} packets)   "
          f"confound corr(rate, sigma) = {r_s:+.3f}")

def eg2(field, encode, packets, S, M=48, top=4, seed=7):
    """EG2 (registered, replaces EG1's weak metric): are the loop's top
    eigenmodes the medium's own eigenfaces? Sample M latents from the VAE
    prior, render each on ITS OWN basis, PCA the grayscale renders
    (eigenfaces of the medium), and cosine-match the top loop eigenmodes
    against the top-8 eigenfaces. Null: same statistic against eigenfaces
    of pixel-scrambled renders (one shared permutation). GATE: mean over
    top-4 modes of max|cos| >= 2x the scrambled null. If [V]: the data
    world and the Gram world agree on what a face-mode is — eigenfaces
    re-derived as DYNAMICAL normal modes. If [K]: two worlds, honestly."""
    import torch
    rng = np.random.default_rng(seed)
    zdim = None
    imgs = []
    print(f"EG2: rendering {M} prior samples on their own bases...")
    for i in range(M):
        if zdim is None:
            # probe latent dim from a dummy encode of gray
            z0 = encode(np.full((S, S, 3), 0.5, np.float32))
            zdim = int(z0.shape[-1])
        z = torch.from_numpy(rng.standard_normal((1, zdim)).astype(
            np.float32)).to(z0.device)
        pk_s, c_s = packets(z)
        Gs = basis_from_packets(pk_s, S)
        imgs.append(sigmoid((c_s.T @ Gs).real).mean(0))
        if (i + 1) % 12 == 0: print(f"  {i+1}/{M}")
    X = np.array(imgs); Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    EF = Vt[:8]                                          # eigenfaces
    perm = rng.permutation(S * S)
    Xs = Xc[:, perm]
    _, _, Vs = np.linalg.svd(Xs, full_matrices=False)
    EFs = Vs[:8]
    K = full_jacobian(field)
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    order = np.argsort(-w)[:top]
    def stat(EFm):
        vals = []
        for j in order:
            v = V[:field.N, j] + 1j * V[field.N:, j]
            img = (v[:, None].T @ field.G).real.ravel()
            img = img - img.mean(); img /= (np.linalg.norm(img) + 1e-12)
            vals.append(max(abs(float(img @ e / (np.linalg.norm(e) + 1e-12)))
                            for e in EFm))
        return float(np.mean(vals))
    s_real, s_null = stat(EF), stat(EFs)
    ok = s_real >= 2 * s_null
    print(f"[{'V' if ok else 'K'}] EG2 mode-vs-eigenface mean max|cos| "
          f"{s_real:.3f} vs scrambled null {s_null:.3f} (gate >= 2x)")

# --------------------------------------------------------------- live app
def live(args):
    import cv2
    S = args.size
    if args.mock:
        pk = random_packets(args.N, 0)
        coeff0 = 0.2 * (np.random.default_rng(2).standard_normal((args.N, 3))
                        + 1j * np.random.default_rng(3).standard_normal((args.N, 3)))
        encode = packets = None
    else:
        encode, packets, dev = load_model_bridge(args.model)
        print(f"model on {dev}")

    # FaceFramer: reuse tiny_avatar3's if there, else minimal Haar crop
    framer = None
    try:
        import tiny_avatar3 as ta
        framer = ta.FaceFramer()
    except Exception:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")
        if casc.empty():
            print("WARN: Haar cascade not found -> face framing OFF, "
                  "live crop falls back to CENTER-CROP. This reintroduces "
                  "the train/live framing mismatch (Dataset Prep face-crops "
                  "with 0.35 margin). Fix: put tiny_avatar3.py next to this "
                  "file, or pip install opencv-contrib-python.")
        class _F:
            box = None
            def crop(self, frame):
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                det = casc.detectMultiScale(g, 1.2, 5)
                if len(det):
                    x, y, w, h = max(det, key=lambda d: d[2] * d[3])
                    m = int(0.35 * w)
                    x0, y0 = max(x - m, 0), max(y - m, 0)
                    x1 = min(x + w + m, frame.shape[1])
                    y1 = min(y + h + m, frame.shape[0])
                    self.box = (x0, y0, x1, y1)
                if self.box is None:
                    h0, w0 = frame.shape[:2]; s = min(h0, w0)
                    self.box = ((w0 - s) // 2, (h0 - s) // 2,
                                (w0 + s) // 2, (h0 + s) // 2)
                x0, y0, x1, y1 = self.box
                return frame[y0:y1, x0:x1]
        framer = _F()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("no camera; use --selftest or --mock with --video")

    field = None
    alpha, frac, mode_field, drive_cut, paused = args.alpha, args.frac, True, False, False
    log = open(args.csv, "w"); log.write(
        "t,mode,alpha,frac,rho,motion,fit,floaters,drive_cut\n")
    t = 0; Fp = None
    decay_rec, decay_freq = [], None

    def keyframe(img01):
        nonlocal field, Fp
        if args.mock:
            pk_l, c0 = pk, coeff0
        else:
            z = encode(img01)
            pk_l, c0 = packets(z)
        G = basis_from_packets(pk_l, S)
        field = Field(G, leak=args.leak, frac=frac, twist=args.twist)
        field.pk = pk_l
        field.c = c0.copy()
        Fp = field.render()
        print(f"keyframe: i_crit {field.i_crit:.4f}  rho_pred {field.rho:.4f}"
              f"  (F1 decay pred {2*math.log(field.rho):.5f}/step)")
        if args.full_eig:
            fb = (S // 4, S // 4, 3 * S // 4, 3 * S // 4)
            im_max, top_w, ratios = eigenmode_gallery(field, S,
                                                      facebox=fb)
            print(f"F3 max|Im mu| = {im_max:.2e} (theorem check)")
            med = float(np.median(ratios)) if ratios else 0.0
            print("EG1 in/out face-box energy ratios:",
                  ["%.2f" % r for r in ratios],
                  f"-> median {med:.2f} vs gate 2.0: "
                  f"[{'V' if med > 2.0 else 'K'}]  (see README: metric "
                  f"diagnosed weak for frame-filling faces; EG2 replaces it)")
            if not args.mock:
                eg2(field, encode, packets, S)

    def floaters(img, ref):
        d = (np.abs(img - ref).mean(0).reshape(S, S) * 255).astype(np.uint8)
        _, th = cv2.threshold(d, 40, 255, cv2.THRESH_BINARY)
        n, _ = cv2.connectedComponents(th)
        return max(n - 1, 0)

    while True:
        ok, frame = cap.read()
        if not ok: break
        face = framer.crop(frame)
        img01 = cv2.resize(face, (S, S)).astype(np.float32)[:, :, ::-1] / 255.
        drive = img01.reshape(-1, 3).T                     # (3, S*S)
        if field is None:
            keyframe(img01)
        if not paused:
            if mode_field:
                field.step(drive=None if drive_cut else drive,
                           alpha=0.0 if drive_cut else alpha)
                out = field.render()
                if drive_cut and len(decay_rec) < 800:
                    decay_rec.append(np.abs(field.c).mean(1).copy())
            else:  # DIRECT: per-frame decode (the shipped app's behavior)
                if args.mock:
                    out = field.render()
                else:
                    z = encode(img01)
                    _, c_direct = packets(z)
                    out = field.render(c_direct)
            m = float(((out - Fp) ** 2).mean()) if Fp is not None else 0.0
            fit = float(((out - drive) ** 2).mean())
            fl = floaters(out, drive)
            log.write(f"{t},{'F' if mode_field else 'D'},{alpha:.2f},"
                      f"{frac:.2f},{field.rho:.4f},{m:.3e},{fit:.3e},"
                      f"{fl},{int(drive_cut)}\n")
            Fp = out; t += 1
        disp = (np.clip(out, 0, 1).T.reshape(S, S, 3)[:, :, ::-1] * 255
                ).astype(np.uint8)
        disp = cv2.resize(disp, (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.putText(disp, f"{'FIELD' if mode_field else 'DIRECT'} "
                    f"a={alpha:.2f} i={frac:.2f}xcrit rho={field.rho:.3f}"
                    f"{' CUT' if drive_cut else ''}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("splat_field  (the other half)", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('k'): keyframe(img01)
        elif k == ord('d'):
            drive_cut = not drive_cut
            if drive_cut:
                decay_rec.clear(); decay_freq = None
            elif decay_rec:
                np.savez("decay_c.npz", amp=np.array(decay_rec),
                         freq=field_freq(field), sigma=field_sigma(field))
                print(f"decay_c.npz saved ({len(decay_rec)} steps) — score "
                      f"with --score_decay decay_c.npz  (RF1)")
        elif k == ord('m'): mode_field = not mode_field
        elif k == ord('a'): alpha = min(alpha + 0.05, 1.0)
        elif k == ord('z'): alpha = max(alpha - 0.05, 0.0)
        elif k == ord('+') or k == ord('='):
            frac = min(frac + 0.05, 2.0); field.set_frac(frac)
        elif k == ord('-'):
            frac = max(frac - 0.05, 0.05); field.set_frac(frac)
        elif k == ord('t'):
            field.twist += 0.1; print(f"twist {field.twist:.2f} [demo regime]")
        elif k == ord('y'):
            field.twist -= 0.1; print(f"twist {field.twist:.2f} [demo regime]")
        elif k == ord(' '): paused = not paused
    log.close(); cap.release(); cv2.destroyAllWindows()

# -------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="model2.pt")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--leak", type=float, default=0.95)
    ap.add_argument("--frac", type=float, default=0.9,
                    help="inject as fraction of i_crit (subcritical < 1)")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="drive mix: 0 autonomous, 1 tracker")
    ap.add_argument("--twist", type=float, default=0.0)
    ap.add_argument("--full_eig", action="store_true")
    ap.add_argument("--csv", default="field_log.csv")
    ap.add_argument("--score_decay", default=None,
                    help="score RF1 from a saved decay_c.npz and exit")
    args = ap.parse_args()
    if args.score_decay:
        score_decay(args.score_decay); sys.exit(0)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    live(args)
