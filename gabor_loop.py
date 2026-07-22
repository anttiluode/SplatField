"""
gabor_loop.py — closed-loop Gabor field iteration with a pre-run spectral
readout. PyTorch transliteration of gabor_loop_verify.py (numpy twin that
was executed with registered predictions; ledger in the repo discussion).
 
Loop:
    r(t)   = sigmoid( render(z(t)) ) - 0.5          # = 0.5*tanh(raw/2)
    z(t+1) = leak * z(t) + inject * e^{i*twist} * Proj(r(t))
Proj = per-atom matched filter <r, G_k>/||G_k||^2.
 
WHAT THE SPECTRUM READOUT TELLS YOU BEFORE RUNNING (verified numerically):
  * With twist = 0, the loop Jacobian K = (1/4) D^-1 R^T R is similar to a
    symmetric PSD matrix -> ALL eigenvalues mu are REAL >= 0 (measured:
    max|Im mu| = 0.0, min mu = -1.6e-10). Consequence, a THEOREM for any
    projection built from the synthesis adjoint or pseudoinverse:
    NO autonomous rotation exists. The only regimes are
       quiescent  (spectral radius max|leak + inject*mu| < 1)
       static pattern (radius > 1; growth arrested by sigmoid saturation).
  * lambda(inject) = leak + inject * eig(J(twist) K) is AFFINE in inject:
    one eigendecomposition per twist gives the entire (leak, inject) phase
    diagram, including the critical inject i_crit printed below.
  * twist != 0 breaks the adjoint symmetry and CREATES rotation (measured
    1.03 rad/step at twist=2.2 where twist=0 gives 0). Onset i_crit(twist)
    remains exactly spectral. OPEN (honestly): the saturated limit-cycle
    FREQUENCY is not yet quantitatively predicted — measured omega came in
    below the linear estimate in all three registered tests (0.0016 vs
    0.0067 at twist 0.35; 1.031 vs 2.23-at-criticality at twist 2.2).
    One-signed residual, mechanism open (mode competition / harmonic
    pulling in deep saturation). The printed omega* is an UPPER-BOUND
    estimate, not a certificate.
 
VERIFIED IN THE NUMPY TWIN (registered thresholds, then runs):
  [V] onset: inject = 0.8*i_crit -> field motion decayed to 9.5e-24;
      inject = 1.3*i_crit -> saturated at 2.4e-10 median, |z| bounded.
  [V] DC: with -0.5 the zero state is EXACT (max|z| = 0.0 after 200
      steps); without it, DC accumulates to ||z*|| = 2.24, matching the
      linear fixed-point prediction (I - M)^-1 inject*b to ratio 0.97.
  [V] motion floor: frame-difference energy separates the two regimes by
      13 orders of magnitude; floor 1e-12 registered.
 
USAGE:
  python gabor_loop.py                     # spectrum + both-sides run
  python gabor_loop.py --twist 2.2        # rotating regime
  python gabor_loop.py --inject 0.05      # explicit gains
  python gabor_loop.py --gate 0.5        # coherence gate ON (see note)
"""
import argparse, math
import torch
 
# ---------------------------------------------------------------- basis
def build_basis(N, S, seed, device):
    g = torch.Generator().manual_seed(seed)
    xs = (torch.arange(S) + 0.5) / S
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    px = torch.rand(N, generator=g) * 0.7 + 0.15
    py = torch.rand(N, generator=g) * 0.7 + 0.15
    sg = torch.rand(N, generator=g) * 0.08 + 0.04
    th = (torch.rand(N, generator=g) * 2 - 1) * math.pi
    fr = torch.rand(N, generator=g) * 5.0 + 3.0
    G = torch.zeros(N, S * S, dtype=torch.complex128)
    for k in range(N):
        dx, dy = X - px[k], Y - py[k]
        xr = dx * torch.cos(th[k]) + dy * torch.sin(th[k])
        env = torch.exp(-(dx * dx + dy * dy) / (2 * sg[k] ** 2))
        G[k] = (env * torch.exp(1j * 2 * math.pi * fr[k] * xr)).ravel()
    return G.to(device)
 
 
class Loop:
    def __init__(self, G, leak, inject, twist=0.0, gate_sigma=0.0):
        self.G = G
        self.n2 = (G.abs() ** 2).sum(1)
        self.leak, self.inject = leak, inject
        self.tw = complex(math.cos(twist), math.sin(twist))
        self.gate_sigma = gate_sigma
        self.prev_dphi = None
 
    def render(self, z):
        return torch.sigmoid((z @ self.G).real)
 
    def project(self, r):
        return (self.G.conj() @ r.to(self.G.dtype)) / self.n2
 
    def step(self, z, sub=True):
        r = self.render(z) - (0.5 if sub else 0.0)
        inj = self.inject * self.tw * self.project(r)
        if self.gate_sigma > 0 and self.prev_dphi is not None:
            # coherence gate (their eq. 3), applied multiplicatively to the
            # injection per atom. NOTE: absent from eq. 4 as written; with
            # it ON the loop is state-dependent even at small amplitude and
            # the spectral predictions below apply only to gate OFF.
            gate = torch.exp(-(1 - torch.cos(self.prev_dphi))
                             / (4 * self.gate_sigma ** 2))
            inj = inj * gate
        z2 = self.leak * z + inj
        self.prev_dphi = torch.angle(z2 * z.conj() + 1e-30)
        return z2
 
 
# ------------------------------------------------- spectral readout
def probed_jacobian(loop, N, eps=1e-5):
    """Realified 2N x 2N Jacobian of the inject-kernel at z=0, probed by
    finite differences through the ACTUAL step (sigmoid slope included)."""
    save = (loop.leak, loop.inject, loop.tw)
    loop.leak, loop.inject, loop.tw = 0.0, 1.0, 1.0 + 0j
    K = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
    for k in range(N):
        for m, pert in enumerate((1.0, 1j)):
            e = torch.zeros(N, dtype=torch.complex128, device=loop.G.device)
            e[k] = eps * pert
            loop.prev_dphi = None
            dz = loop.step(e) / eps
            K[:N, m * N + k] = dz.real.cpu()
            K[N:, m * N + k] = dz.imag.cpu()
    loop.leak, loop.inject, loop.tw = save
    loop.prev_dphi = None
    return K
 
 
def Jrot(N, d):
    c, s = math.cos(d), math.sin(d)
    T = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
    I = torch.eye(N, dtype=torch.float64)
    T[:N, :N], T[:N, N:] = c * I, -s * I
    T[N:, :N], T[N:, N:] = s * I, c * I
    return T
 
 
def spectrum_readout(K, N, leak, twist):
    ev = torch.linalg.eigvals(Jrot(N, twist) @ K)
    def radius(i):
        return (leak + i * ev).abs().max().item()
    hi = 1.0
    while radius(hi) < 1.0:
        hi *= 2
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if radius(mid) < 1.0 else (lo, mid)
    i_crit = 0.5 * (lo + hi)
    return ev, i_crit
 
 
def mode_table(ev, leak, inject, n=5):
    lam = leak + inject * ev
    idx = torch.argsort(-lam.abs())[:n]
    rows = []
    for j in idx:
        rho, om = lam[j].abs().item(), abs(torch.angle(lam[j]).item())
        rows.append((rho, om, (2 * math.pi / om) if om > 1e-9 else float("inf")))
    return rows
 
 
# ------------------------------------------------- run + metrics
def run(loop, N, T, device, log_csv=None, z0=None):
    g = torch.Generator().manual_seed(1)
    z = (1e-3 * (torch.randn(N, generator=g)
                 + 1j * torch.randn(N, generator=g))).to(
        torch.complex128).to(device) if z0 is None else z0
    Fp = loop.render(z)
    rows, wom = [], []
    for t in range(T):
        z2 = loop.step(z)
        Fn = loop.render(z2)
        m = float(((Fn - Fp) ** 2).mean())          # motion-floor metric
        w = z.abs() ** 2 + 1e-30
        dphi = torch.angle(z2 * z.conj() + 1e-30)
        omega = float((w * dphi).sum() / w.sum())   # power-weighted phase vel
        rows.append((t, m, omega, float(z2.abs().max())))
        if t >= T - 300:
            wom.append(omega)
        z, Fp = z2, Fn
    if log_csv:
        with open(log_csv, "w") as fh:
            fh.write("t,motion_energy,omega_w,max_abs_z\n")
            for r in rows:
                fh.write(",".join(f"{v:.6e}" if i else str(v)
                                  for i, v in enumerate(r)) + "\n")
    m_final = rows[-1][1]
    m_med = sorted(r[1] for r in rows[-200:])[100]
    return z, m_final, m_med, abs(sum(wom) / max(len(wom), 1))
 
 
MOTION_FLOOR = 1e-12   # registered: quiescent iff m(t) ends below this
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leak", type=float, default=0.95)
    ap.add_argument("--inject", type=float, default=None,
                    help="default: runs BOTH 0.8x and 1.3x critical")
    ap.add_argument("--twist", type=float, default=0.0,
                    help="phase advance (rad) on the injection; 0 = the "
                         "loop as specified (provably non-rotating)")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="coherence-gate sigma (0 = off; predictions "
                         "certified for OFF only)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--csv", default="gabor_loop_log.csv")
    ap.add_argument("--dc_check", action="store_true",
                    help="verify z=0 is exact and show the no-subtraction "
                         "counterfactual")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    G = build_basis(args.N, args.size, args.seed, dev)
    loop = Loop(G, args.leak, 1.0, args.twist, args.gate)
    print(f"basis N={args.N} size={args.size} on {dev}; probing Jacobian...")
    K = probed_jacobian(loop, args.N)
    ev, i_crit = spectrum_readout(K, args.N, args.leak, args.twist)
    im_max = float(torch.linalg.eigvals(K).imag.abs().max())
    print(f"twist=0 spectrum check: max|Im mu| = {im_max:.2e} "
          f"(theorem: 0 -> no autonomous rotation without twist)")
    print(f"leak {args.leak}  twist {args.twist}  i_crit {i_crit:.4f}")
 
    def show(inj, label):
        print(f"-- {label}: inject {inj:.4f} --")
        for rho, om, per in mode_table(ev, args.leak, inj):
            print(f"   |lam| {rho:.4f}  omega {om:.4f} rad/step  "
                  f"period {per:8.1f}")
        loop.inject = inj
        loop.prev_dphi = None
        _, m_f, m_med, om_meas = run(loop, args.N, args.steps, dev,
                                     log_csv=args.csv)
        state = "QUIESCENT" if m_f < MOTION_FLOOR else "ACTIVE"
        print(f"   run: m(final) {m_f:.2e}  m(median) {m_med:.2e}  "
              f"omega_measured {om_meas:.4f}  -> {state}")
 
    if args.inject is None:
        show(0.8 * i_crit, "predicted quiescent (0.8 x critical)")
        show(1.3 * i_crit, "predicted active (1.3 x critical)")
    else:
        show(args.inject, "configured")
 
    if args.dc_check:
        loop.inject = 0.8 * i_crit
        z = torch.zeros(args.N, dtype=torch.complex128, device=dev)
        loop.prev_dphi = None
        for _ in range(200):
            z = loop.step(z, sub=True)
        print(f"DC with -0.5: max|z| after 200 steps = "
              f"{float(z.abs().max()):.1e} (exact 0 expected)")
        z = torch.zeros(args.N, dtype=torch.complex128, device=dev)
        loop.prev_dphi = None
        for _ in range(args.steps):
            z = loop.step(z, sub=False)
        print(f"DC without -0.5: ||z*|| = {float(z.abs().pow(2).sum().sqrt()):.4f}"
              f" (accumulated bias state — this is what the subtraction "
              f"prevents)")
 
 
if __name__ == "__main__":
    main()
 