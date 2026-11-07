"""
gated.py — genuine gated channels with ONE shared (u, v, a) across all of the
channel's active degrees.

A real SwiGLU channel is

    a * silu(<u,x>) * <v,x>,      silu(z) = sum_{m>=1} s_m z^m,

so its degree-(m+1) homogeneous component is exactly

    a * s_m * <u,x>^m <v,x> .

The SiLU^2 (tied) channel is the same with v = u.

SiLU Taylor coefficients (verified numerically):
    s_1 = 1/2,  s_2 = 1/4,  s_3 = 0,  s_4 = -1/48,  s_5 = 0,  s_6 = 1/480, ...
so the ACTIVE degrees of a SwiGLU channel are 2, 3, 5, 7, ...

Crucially there is no scaling gauge here: rescaling u_i multiplies the
degree-(m+1) component by t^m, which cannot be undone by the single shared a_i.
This is precisely the coupling the reviewer asks about.
"""
import numpy as np
import scipy.optimize as sopt
import torch

from cpcore import Basis, untied_vec

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

# silu Taylor coefficients s_m, m = 1..8
SILU_S = {1: 0.5, 2: 0.25, 3: 0.0, 4: -1.0 / 48, 5: 0.0, 6: 1.0 / 480,
          7: 0.0, 8: -17.0 / 80640}
ACTIVE_DEGREES = [2, 3, 5, 7]          # m+1 for m with s_m != 0


def silu(z):
    return z / (1.0 + np.exp(-z))


def channel_components(U, V, a, n, degrees, tied=False):
    """Exact homogeneous components of sum_i a_i silu(<u_i,x>) <v_i,x>.

    Returns {k: coeff vector} for k in degrees.  torch, differentiable.
    """
    out = {}
    for k in degrees:
        s = SILU_S[k - 1]
        acc = torch.zeros(Basis(n, k).N, dtype=torch.float64)
        for i in range(U.shape[0]):
            vi = U[i] if tied else V[i]
            acc = acc + a[i] * untied_vec(U[i], vi, n, k)
        out[k] = s * acc
    return out


def fit_gated_components(targets, n, degrees, w, tied, seeds=16, rng=None,
                         maxiter=2000):
    """Fit a genuine gated block to a multi-degree target, IN COEFFICIENT SPACE.

    The model is the exact set of homogeneous components of a real SwiGLU
    (tied=False) or SiLU^2 (tied=True) block of width w: one (u_i, v_i, a_i)
    per channel, shared across every degree, with the gate's own Taylor
    coefficients fixed.

    Returns overall relative loss plus the PER-DEGREE relative errors of the
    best fit, which is what decides whether the separations can hold at every
    degree simultaneously.
    """
    Bs = {k: Basis(n, k) for k in degrees}
    T = {k: torch.tensor(targets[k]) for k in degrees}
    tn2 = sum(Bs[k].norm2(T[k]).item() for k in degrees)
    rng = rng or np.random.default_rng(0)

    def build(th):
        i = 0
        U = th[i:i + w * n].reshape(w, n); i += w * n
        if tied:
            V = U
        else:
            V = th[i:i + w * n].reshape(w, n); i += w * n
        a = th[i:i + w]
        return U, V, a

    def loss_grad(x):
        th = torch.tensor(x, requires_grad=True)
        U, V, a = build(th)
        comp = channel_components(U, V, a, n, degrees, tied=tied)
        L = sum(Bs[k].norm2(comp[k] - T[k]) for k in degrees) / tn2
        L.backward()
        return L.item(), th.grad.numpy().copy()

    dim = w * n * (1 if tied else 2) + w
    best, best_x = np.inf, None
    losses = []
    for s in range(seeds):
        x0 = rng.normal(size=dim) * 0.9
        r = sopt.minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                       ftol=1e-18, gtol=1e-14))
        losses.append(float(r.fun))
        if r.fun < best:
            best, best_x = float(r.fun), r.x

    # per-degree breakdown at the best fit
    th = torch.tensor(best_x)
    U, V, a = build(th)
    comp = channel_components(U, V, a, n, degrees, tied=tied)
    per_deg = {int(k): float(Bs[k].norm2(comp[k] - T[k]) / Bs[k].norm2(T[k]))
               for k in degrees}
    losses = np.array(losses)
    return dict(min=best, median=float(np.median(losses)),
                per_degree=per_deg, losses=losses.tolist(),
                nsucc=int(np.sum(losses < 1e-10)))


# --------------------------------------------------- real block on data
def fit_gated_data(target_fn, n, w, tied, X, y, seeds=8, rng=None, maxiter=3000):
    """Fit the FULL real block (actual silu, all degrees) to samples by L-BFGS."""
    rng = rng or np.random.default_rng(0)
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    yv = float((yt ** 2).mean())

    def loss_grad(x):
        th = torch.tensor(x, requires_grad=True)
        i = 0
        U = th[i:i + w * n].reshape(w, n); i += w * n
        if tied:
            V = U
        else:
            V = th[i:i + w * n].reshape(w, n); i += w * n
        a = th[i:i + w]
        g = Xt @ U.T                                  # (P, w)
        val = Xt @ V.T                                # (P, w)
        pred = ((g * torch.sigmoid(g)) * val) @ a
        L = ((pred - yt) ** 2).mean() / yv
        L.backward()
        return L.item(), th.grad.numpy().copy()

    dim = w * n * (1 if tied else 2) + w
    losses, xs = [], []
    for s in range(seeds):
        x0 = rng.normal(size=dim) * 0.9
        r = sopt.minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                       ftol=1e-18, gtol=1e-14))
        losses.append(float(r.fun)); xs.append(r.x)
    losses = np.array(losses)
    return dict(min=float(losses.min()), median=float(np.median(losses)),
                theta=xs[int(np.argmin(losses))], losses=losses.tolist())


def multi_degree_target(n, degrees, seed, kind="single_untied"):
    """Build a genuine multi-degree target.

    kind='single_untied': sum_k c_k <p,x>^{k-1} <q,x>  with ONE fixed (p,q).
        Per-degree untied rank is 1 at every degree, per-degree tied rank is k.
        This is the case where per-degree theory is most optimistic.
    kind='dense': independent random dense form at each degree.
    """
    from cpcore import random_dense
    rng = np.random.default_rng(seed)
    if kind == "dense":
        return {k: random_dense(n, k, seed * 100 + k) for k in degrees}
    p = torch.tensor(rng.normal(size=n))
    q = torch.tensor(rng.normal(size=n))
    p = p / p.norm(); q = q / q.norm()
    c = rng.normal(size=len(degrees))
    out = {}
    for ci, k in zip(c, degrees):
        vec = ci * untied_vec(p, q, n, k)
        out[k] = (vec / torch.sqrt(Basis(n, k).norm2(vec))).numpy()
    return out
