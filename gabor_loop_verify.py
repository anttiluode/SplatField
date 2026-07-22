#!/usr/bin/env python3
"""
gabor_loop_verify.py — numpy twin of the closed-loop Gabor iteration.
Verifies, with registered predictions, that the loop's behavior is
predictable BEFORE running it from the spectrum of its probed Jacobian.

Loop:  z(t+1) = leak*z(t) + inject * Proj( sigmoid(render(z)) - 0.5 )

REGISTERED PREDICTIONS (thresholds fixed before the run):
 P1 QUIESCENT  at inject = 0.8*i_crit, spectral radius < 1 -> field-motion
               energy m(t) decays below 1e-12; classifier says QUIESCENT.
 P2 ACTIVE     at inject = 1.3*i_crit, radius > 1 -> m(t) saturates above
               1e-10 (>=100x the floor); |z| stays bounded (sigmoid cap).
 P3 FREQUENCY  measured power-weighted phase velocity in the saturated
               regime within 35% of arg(lambda_dom) predicted at z=0
               (tolerance is generous because saturation renormalizes).
 P4 DC-FIXED   with the -0.5 subtraction, z=0 stays exactly 0 (<=1e-14)
               for 200 steps.
 P5 DC-ACCUM   WITHOUT the subtraction (same quiescent gains), the loop
               accumulates a nonzero DC state |z*| > 1e-3 (>=1e5 x the
               subtracted loop's decay level at the same step count).
"""
import numpy as np

rng = np.random.default_rng(0)
N, S = 256, 64
leak = 0.95

# ---------------- basis ----------------
xs = (np.arange(S) + 0.5) / S
X, Y = np.meshgrid(xs, xs, indexing="xy")
px = rng.uniform(0.15, 0.85, N); py = rng.uniform(0.15, 0.85, N)
sg = rng.uniform(0.04, 0.12, N); th = rng.uniform(-np.pi, np.pi, N)
fr = rng.uniform(3.0, 8.0, N)
G = np.zeros((N, S * S), complex)
for k in range(N):
    dx, dy = X - px[k], Y - py[k]
    xr = dx * np.cos(th[k]) + dy * np.sin(th[k])
    env = np.exp(-(dx * dx + dy * dy) / (2 * sg[k] ** 2))
    G[k] = (env * np.exp(1j * 2 * np.pi * fr[k] * xr)).ravel()
n2 = (np.abs(G) ** 2).sum(1)

def sigmoid(u): return 1.0 / (1.0 + np.exp(-u))
def render(z):  return sigmoid((z @ G).real)
def project(r): return (G.conj() @ r) / n2
def stepf(z, lk, inj, sub=True):
    r = render(z) - (0.5 if sub else 0.0)
    return lk * z + inj * project(r)

# ---------------- Gram structure readout ----------------
H = G @ G.conj().T                     # Hermitian Gram
C = G @ G.T                            # conjugate (anomalous) Gram
print(f"Gram: ||H||_F {np.linalg.norm(H):.1f}  ||C||_F {np.linalg.norm(C):.1f}"
      f"  ratio {np.linalg.norm(C)/np.linalg.norm(H):.3f}"
      "   (rotation can only come from the C part — see note)")

# ---------------- Jacobian probe (realified 2N x 2N) ----------------
eps = 1e-5
K = np.zeros((2 * N, 2 * N))
for k in range(N):
    for m, pert in enumerate((1.0, 1j)):
        e = np.zeros(N, complex); e[k] = eps * pert
        dz = stepf(e, 0.0, 1.0) / eps          # leak=0, inject=1 kernel
        K[:N, m * N + k] = dz.real
        K[N:, m * N + k] = dz.imag
mu = np.linalg.eigvals(K)

def radius(inj): return np.abs(leak + inj * mu).max()
hi = 1.0
while radius(hi) < 1.0: hi *= 2
lo = 0.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if radius(mid) < 1.0 else (lo, mid)
i_crit = 0.5 * (lo + hi)
print(f"leak {leak}  critical inject {i_crit:.4f}  "
      f"(entire (leak,inject) phase diagram from this one spectrum)")

def report_modes(inj, n=5):
    lam = leak + inj * mu
    o = np.argsort(-np.abs(lam))[:n]
    for j in o:
        rho, om = np.abs(lam[j]), abs(np.angle(lam[j]))
        per = (2 * np.pi / om) if om > 1e-9 else np.inf
        print(f"    mode |lam| {rho:.4f}  omega {om:.4f} rad/step  "
              f"period {per:7.1f} steps")
    j = o[0]
    return np.abs(lam[j]), abs(np.angle(lam[j]))

# ---------------- runs ----------------
def run(inj, T, sub=True, z0=None):
    z = (1e-3 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
         if z0 is None else z0.copy())
    Fp = render(z)
    m_hist, w_om = [], []
    for t in range(T):
        z2 = stepf(z, leak, inj, sub)
        Fn = render(z2)
        m_hist.append(float(((Fn - Fp) ** 2).mean()))
        if t >= T - 300:
            w = np.abs(z) ** 2 + 1e-30
            dphi = np.angle(z2 * np.conj(z) + 1e-30)
            w_om.append(float((w * dphi).sum() / w.sum()))
        z, Fp = z2, Fn
    return z, np.array(m_hist), abs(np.mean(w_om)) if w_om else 0.0

print("\n-- QUIESCENT setting: inject = 0.8 * i_crit --")
iq = 0.8 * i_crit
rho_q, om_q = report_modes(iq)
zq, mq, omq_meas = run(iq, 1200)
print(f"predicted radius {rho_q:.4f} (<1)  measured m(final) {mq[-1]:.2e}")
P1 = rho_q < 1 and mq[-1] < 1e-12
print(f"[{'V' if P1 else 'K'}] P1 quiescent as predicted")

print("\n-- ACTIVE setting: inject = 1.3 * i_crit --")
ia = 1.3 * i_crit
rho_a, om_a = report_modes(ia)
za, ma, oma_meas = run(ia, 1200)
m_med = float(np.median(ma[-200:]))
P2 = rho_a > 1 and m_med > 1e-10 and np.abs(za).max() < 1e3
print(f"predicted radius {rho_a:.4f} (>1)  measured m(median,last200) "
      f"{m_med:.2e}  max|z| {np.abs(za).max():.3f}")
print(f"[{'V' if P2 else 'K'}] P2 active as predicted, bounded by sigmoid")

if om_a > 0.05:
    P3 = abs(oma_meas - om_a) <= 0.35 * om_a
    print(f"[{'V' if P3 else 'K'}] P3 rotation: predicted omega {om_a:.4f}, "
          f"measured {oma_meas:.4f} rad/step "
          f"(err {abs(oma_meas-om_a)/om_a*100:.0f}%, tol 35%)")
else:
    P3 = oma_meas < 0.05
    print(f"[{'V' if P3 else 'K'}] P3 dominant mode ~static (omega_pred "
          f"{om_a:.4f}); measured {oma_meas:.4f} (< 0.05 required)")

print("\n-- DC tests --")
z = np.zeros(N, complex)
for _ in range(200):
    z = stepf(z, leak, iq, sub=True)
P4 = np.abs(z).max() <= 1e-14
print(f"[{'V' if P4 else 'K'}] P4 with -0.5: z stays at 0 "
      f"(max|z| {np.abs(z).max():.1e})")
znos, mnos, _ = run(iq, 1200, sub=False, z0=np.zeros(N, complex))
b = project(np.full(S * S, 0.5))
M2 = leak * np.eye(2 * N) + iq * K
v = np.linalg.solve(np.eye(2 * N) - M2, iq * np.concatenate([b.real, b.imag]))
zlin = np.linalg.norm(v)
P5 = np.linalg.norm(znos) > 1e-3
print(f"[{'V' if P5 else 'K'}] P5 without -0.5: DC accumulates to |z*| "
      f"{np.linalg.norm(znos):.4f} (linear pred {zlin:.4f}, "
      f"ratio {np.linalg.norm(znos)/max(zlin,1e-12):.2f}); "
      f"subtracted loop at same step count: {np.sqrt(mq[-1]):.1e}-scale")

print("\nVERDICT " + ("[V] spectrum predicts the loop"
      if all([P1, P2, P3, P4, P5]) else "[K] read the failing line"))
