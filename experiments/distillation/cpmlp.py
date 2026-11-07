"""
Shared harness for the CP-MLP distillation experiments.

Blocks (n -> d, width w):
  untied "SwiGLU"  F(x) = sum_i  a_i * silu(<u_i,x>) * <v_i,x>     params w(2n+d)
  tied   "SiLU^2"  F(x) = sum_i  a_i * silu(<u_i,x>) * <u_i,x>     params w(n+d)

Loss is relative MSE  ||F(X)-Y||_F^2 / ||Y||_F^2, minimised by L-BFGS-B with
analytic gradients.  Everything is float64 numpy; no torch.
"""
import numpy as np
from scipy.optimize import minimize

N_IN, N_OUT = 8, 8


# ---------------------------------------------------------------- gate
def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def silu(z):
    return z * _sig(z)


def dsilu(z):
    s = _sig(z)
    return s * (1.0 + z * (1.0 - s))


# ---------------------------------------------------------------- packing
def nparams(w, tied, n=N_IN, d=N_OUT):
    return w * (n + d) if tied else w * (2 * n + d)


def unpack(theta, w, tied, n=N_IN, d=N_OUT):
    i = 0
    U = theta[i:i + n * w].reshape(n, w); i += n * w
    if tied:
        V = None
    else:
        V = theta[i:i + n * w].reshape(n, w); i += n * w
    C = theta[i:i + w * d].reshape(w, d)
    return U, V, C


def pack(U, V, C):
    parts = [U.ravel()] + ([] if V is None else [V.ravel()]) + [C.ravel()]
    return np.concatenate(parts)


# ---------------------------------------------------------------- forward
def forward(theta, X, w, tied):
    U, V, C = unpack(theta, w, tied)
    A = X @ U
    B = A if tied else X @ V
    H = silu(A) * B
    return H @ C


def loss_grad(theta, X, Y, w, tied, Ynorm2):
    U, V, C = unpack(theta, w, tied)
    A = X @ U
    B = A if tied else X @ V
    gA = silu(A)
    H = gA * B
    R = H @ C - Y
    L = np.sum(R * R) / Ynorm2

    dF = (2.0 / Ynorm2) * R
    gC = H.T @ dF
    dH = dF @ C.T
    if tied:
        # H = silu(A)*A  ->  dH/dA = dsilu(A)*A + silu(A)
        dA = dH * (dsilu(A) * A + gA)
        gU = X.T @ dA
        return L, np.concatenate([gU.ravel(), gC.ravel()])
    else:
        dA = dH * B * dsilu(A)
        dB = dH * gA
        gU = X.T @ dA
        gV = X.T @ dB
        return L, np.concatenate([gU.ravel(), gV.ravel(), gC.ravel()])


# ---------------------------------------------------------------- fitting
def init(w, tied, seed, n=N_IN, d=N_OUT):
    rng = np.random.default_rng(seed)
    U = rng.normal(scale=1.0 / np.sqrt(n), size=(n, w))
    V = None if tied else rng.normal(scale=1.0 / np.sqrt(n), size=(n, w))
    C = rng.normal(scale=1.0 / np.sqrt(w), size=(w, d))
    return pack(U, V, C)


def fit(X, Y, w, tied, seed, maxiter=3000, restarts=1, Xte=None, Yte=None):
    """Fit and return (train_loss, test_loss).  Best of `restarts` inits."""
    Yn2 = float(np.sum(Y * Y))
    best, best_th = np.inf, None
    for r in range(restarts):
        th0 = init(w, tied, 10000 * r + seed)
        res = minimize(loss_grad, th0, args=(X, Y, w, tied, Yn2), jac=True,
                       method='L-BFGS-B',
                       options=dict(maxiter=maxiter, maxfun=maxiter * 2,
                                    ftol=1e-16, gtol=1e-12))
        if res.fun < best:
            best, best_th = float(res.fun), res.x
    if Xte is None:
        return best, best
    Rte = forward(best_th, Xte, w, tied) - Yte
    te = float(np.sum(Rte * Rte) / np.sum(Yte * Yte))
    return best, te
