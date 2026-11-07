"""
cpcore.py — coefficient-space harness for tied/untied CP-MLP blocks.

Conventions
-----------
A homogeneous degree-k form in n variables is stored as its *monomial*
coefficient vector m over the exponent multi-indices alpha with |alpha| = k:
        f(x) = sum_alpha m_alpha x^alpha .

Norms are apolar (Bombieri):  ||f||^2 = sum_alpha (alpha! / k!) m_alpha^2,
so that ||<u,x>^k|| = ||u||^k exactly.  All reported losses are

        rel_loss = ||f_hat - f||^2_apolar / ||f||^2_apolar .

Atoms
-----
tied    :  lambda * <u,x>^k
untied  :  c      * <u,x>^{k-1} <v,x>

Multi-degree models share the direction(s) across degrees and carry one
free output coefficient per degree per atom:

tied    :  sum_i sum_{k in K} lam_{i,k} <u_i,x>^k
untied  :  sum_i sum_{k in K} c_{i,k}  <u_i,x>^{k-1} <v_i,x>

Gradients come from torch autograd; the optimizer is scipy L-BFGS-B.
"""
import itertools
from math import comb, factorial

import numpy as np
import scipy.optimize as sopt
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# ----------------------------------------------------------------- bases
class Basis:
    """Monomial basis for degree-k forms in n variables."""

    _cache = {}

    def __new__(cls, n, k):
        key = (n, k)
        if key in cls._cache:
            return cls._cache[key]
        self = super().__new__(cls)
        self.n, self.k = n, k
        self.alphas = [a for a in itertools.product(range(k + 1), repeat=n)
                       if sum(a) == k]
        self.N = len(self.alphas)
        self.index = {a: i for i, a in enumerate(self.alphas)}
        E = np.array(self.alphas, dtype=np.int64)
        self.E = torch.tensor(E, dtype=torch.float64)          # (N, n)
        # multinomial coefficients k!/alpha!
        mult = np.array([factorial(k) // int(np.prod([factorial(int(t)) for t in a]))
                         for a in self.alphas], dtype=np.float64)
        self.mult = torch.tensor(mult)
        # apolar weights: ||f||^2 = sum (alpha!/k!) m^2  -> y = sqrt(alpha!/k!) m
        self.apo = torch.tensor(1.0 / np.sqrt(mult))           # sqrt(alpha!/k!)
        cls._cache[key] = self
        return self

    def pow_vec(self, u):
        """monomial coefficients of <u,x>^k  (torch, differentiable)."""
        return self.mult * torch.prod(u.unsqueeze(0) ** self.E, dim=1)

    def norm2(self, m):
        return torch.sum((self.apo * m) ** 2)


class MulLinear:
    """Multiply a degree-(k-1) coefficient vector by a linear form."""

    _cache = {}

    def __new__(cls, n, k):
        key = (n, k)
        if key in cls._cache:
            return cls._cache[key]
        self = super().__new__(cls)
        lo, hi = Basis(n, k - 1), Basis(n, k)
        rows, cols = [], []
        for i, a in enumerate(lo.alphas):
            for j in range(n):
                b = list(a)
                b[j] += 1
                rows.append(hi.index[tuple(b)])
                cols.append(i * n + j)
        self.n, self.k = n, k
        self.lo, self.hi = lo, hi
        self.rows = torch.tensor(rows, dtype=torch.long)
        self.cols = torch.tensor(cols, dtype=torch.long)
        cls._cache[key] = self
        return self

    def __call__(self, c_lo, v):
        """c_lo: (N_{k-1},) coefficients; v: (n,) -> (N_k,)"""
        prod = (c_lo.unsqueeze(1) * v.unsqueeze(0)).reshape(-1)   # (N_{k-1}*n,)
        out = torch.zeros(self.hi.N, dtype=c_lo.dtype)
        return out.index_add(0, self.rows, prod[self.cols])


def untied_vec(u, v, n, k):
    """monomial coefficients of <u,x>^{k-1} <v,x>."""
    if k == 1:
        return v.clone()
    return MulLinear(n, k)(Basis(n, k - 1).pow_vec(u), v)


# ------------------------------------------------------- generic ranks
def Ndim(n, k):
    return comb(n + k - 1, k)


def generic_tied(n, k):
    """Alexander-Hirschowitz generic Waring rank (with the four exceptions)."""
    base = -(-Ndim(n, k) // n)
    if k == 2:
        return n
    if (n, k) in {(3, 4), (4, 4), (5, 3), (5, 4)}:
        return base + 1
    return base


def generic_untied(n, k):
    """Generic rank for sums of <u,x>^{k-1}<v,x> (tangential variety).

    Parameter count gives ceil(N/(2n-1)); the tangential variety to the
    Veronese is defective for cubics in the cases below (Abo-Vannieuwenhoven),
    where the generic rank is one larger.
    """
    base = -(-Ndim(n, k) // (2 * n - 1))
    if k == 3 and n in (3, 4, 5):
        return base + 1
    return base


# --------------------------------------------------- single-degree fits
def _pack(parts):
    return np.concatenate([p.ravel() for p in parts])


def fit_single(target, n, k, w, tied, seeds=12, rng=None, maxiter=800,
               return_best_theta=False):
    """Fit w atoms of one degree to `target` (numpy monomial coeff vector).

    Returns dict with median / min relative loss over `seeds` restarts.
    """
    B = Basis(n, k)
    t = torch.tensor(target)
    tn2 = B.norm2(t).item()
    rng = rng or np.random.default_rng(0)

    def unpack(th):
        i = 0
        U = th[i:i + w * n].reshape(w, n); i += w * n
        if tied:
            V = None
        else:
            V = th[i:i + w * n].reshape(w, n); i += w * n
        C = th[i:i + w]
        return U, V, C

    def loss_grad(x):
        th = torch.tensor(x, requires_grad=True)
        U, V, C = unpack(th)
        acc = torch.zeros(B.N, dtype=torch.float64)
        for i in range(w):
            if tied:
                acc = acc + C[i] * B.pow_vec(U[i])
            else:
                acc = acc + C[i] * untied_vec(U[i], V[i], n, k)
        L = B.norm2(acc - t) / tn2
        L.backward()
        return L.item(), th.grad.numpy().copy()

    dim = w * n * (1 if tied else 2) + w
    losses, thetas = [], []
    for s in range(seeds):
        x0 = rng.normal(size=dim) * 0.7
        r = sopt.minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                       ftol=1e-18, gtol=1e-14))
        losses.append(float(r.fun)); thetas.append(r.x)
    losses = np.array(losses)
    out = dict(median=float(np.median(losses)), min=float(losses.min()),
               losses=losses.tolist())
    if return_best_theta:
        out["theta"] = thetas[int(np.argmin(losses))]
    return out


# ---------------------------------------------------- multi-degree fits
def fit_multi(targets, n, degrees, w, tied, seeds=12, rng=None, maxiter=1200,
              return_losses=True):
    """Fit a multi-degree block with directions shared across degrees.

    targets : dict {k: numpy monomial coeff vector}
    """
    Bs = {k: Basis(n, k) for k in degrees}
    T = {k: torch.tensor(targets[k]) for k in degrees}
    tn2 = sum(Bs[k].norm2(T[k]).item() for k in degrees)
    rng = rng or np.random.default_rng(0)
    nd = len(degrees)

    def loss_grad(x):
        th = torch.tensor(x, requires_grad=True)
        i = 0
        U = th[i:i + w * n].reshape(w, n); i += w * n
        if tied:
            V = None
        else:
            V = th[i:i + w * n].reshape(w, n); i += w * n
        C = th[i:i + w * nd].reshape(w, nd)
        L = 0.0
        for di, k in enumerate(degrees):
            acc = torch.zeros(Bs[k].N, dtype=torch.float64)
            for a in range(w):
                if tied:
                    acc = acc + C[a, di] * Bs[k].pow_vec(U[a])
                else:
                    acc = acc + C[a, di] * untied_vec(U[a], V[a], n, k)
            L = L + Bs[k].norm2(acc - T[k])
        L = L / tn2
        L.backward()
        return L.item(), th.grad.numpy().copy()

    dim = w * n * (1 if tied else 2) + w * nd
    losses = []
    for s in range(seeds):
        x0 = rng.normal(size=dim) * 0.7
        r = sopt.minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                       ftol=1e-18, gtol=1e-14))
        losses.append(float(r.fun))
    losses = np.array(losses)
    return dict(median=float(np.median(losses)), min=float(losses.min()),
                losses=losses.tolist())


def pred_multi_tied(n, degrees):
    """Parameter-count threshold for the shared-direction tied model.

    Each atom carries n direction parameters minus 1 scaling redundancy
    (rescaling u_i is absorbable degree-by-degree into lam_{i,k}), plus one
    coefficient per degree.
    """
    N = sum(Ndim(n, k) for k in degrees)
    return -(-N // (n - 1 + len(degrees)))


def pred_multi_untied(n, degrees):
    """Same count for the shared-direction untied model: two directions,
    two scaling redundancies, one coefficient per degree."""
    N = sum(Ndim(n, k) for k in degrees)
    return -(-N // (2 * n - 2 + len(degrees)))


# ------------------------------------------------------------- targets
def random_dense(n, k, seed):
    """Dense random degree-k form: iid Gaussian coefficients in the
    *apolar-orthonormal* basis (so the target is isotropic in Sym^k)."""
    B = Basis(n, k)
    rng = np.random.default_rng(seed)
    y = rng.normal(size=B.N)                     # apolar-orthonormal coords
    m = y / B.apo.numpy()                        # back to monomial coeffs
    return m / np.sqrt(float(B.norm2(torch.tensor(m))))


def block_family(n, k, s, seed):
    """F_{k,s} = sum_{i<=s} c_i <l_i,x>^{k-1} <m_i,x>, 2s independent forms."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(2 * s, n))
    acc = torch.zeros(Basis(n, k).N, dtype=torch.float64)
    for i in range(s):
        acc = acc + untied_vec(torch.tensor(L[2 * i]), torch.tensor(L[2 * i + 1]), n, k)
    m = acc.numpy()
    return m / np.sqrt(float(Basis(n, k).norm2(acc)))
