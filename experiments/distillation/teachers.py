"""
Three teacher FFN layers, all trained on the SAME task at the SAME parameter
budget (192 params each), so that the only thing that differs between them is
the architectural family the teacher belongs to:

  T_untied   SwiGLU  width  8   ->  8*(2*8+8) = 192
  T_tied     SiLU^2  width 12   -> 12*( 8 +8) = 192
  T_neutral  tanh MLP width 12  -> 12*( 8 +8) = 192   (neither gated family)

Task: a fixed random wide tanh network R^8 -> R^8 (width 64).  It is NOT
exactly representable at width 8/12, so each trained teacher keeps its own
architecture's fingerprint rather than collapsing onto a shared exact solution.
"""
import json
import numpy as np
from scipy.optimize import minimize
import cpmlp as M

N_IN, N_OUT = M.N_IN, M.N_OUT
TASK_SEED = 7


# ------------------------------------------------------------------ task
def task_fn():
    rng = np.random.default_rng(TASK_SEED)
    W1 = rng.normal(scale=1.0 / np.sqrt(N_IN), size=(N_IN, 64))
    W2 = rng.normal(scale=1.0 / np.sqrt(64), size=(64, N_OUT))
    b1 = rng.normal(scale=0.5, size=64)
    return lambda X: np.tanh(X @ W1 + b1) @ W2


def gauss(n, seed):
    return np.random.default_rng(seed).normal(size=(n, N_IN))


# ------------------------------------------------------ neutral tanh MLP
def mlp_forward(theta, X, w):
    U = theta[:N_IN * w].reshape(N_IN, w)
    C = theta[N_IN * w:].reshape(w, N_OUT)
    return np.tanh(X @ U) @ C


def mlp_lg(theta, X, Y, w, Yn2):
    U = theta[:N_IN * w].reshape(N_IN, w)
    C = theta[N_IN * w:].reshape(w, N_OUT)
    A = X @ U
    H = np.tanh(A)
    R = H @ C - Y
    L = np.sum(R * R) / Yn2
    dF = (2.0 / Yn2) * R
    gC = H.T @ dF
    dA = (dF @ C.T) * (1.0 - H * H)
    gU = X.T @ dA
    return L, np.concatenate([gU.ravel(), gC.ravel()])


def fit_mlp(X, Y, w, seed, maxiter=4000):
    rng = np.random.default_rng(seed)
    th0 = np.concatenate([
        rng.normal(scale=1 / np.sqrt(N_IN), size=N_IN * w),
        rng.normal(scale=1 / np.sqrt(w), size=w * N_OUT)])
    Yn2 = float(np.sum(Y * Y))
    r = minimize(mlp_lg, th0, args=(X, Y, w, Yn2), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-16, gtol=1e-12))
    return float(r.fun), r.x


# ------------------------------------------------------------------ build
def build(restarts=6, ntrain=4000):
    f = task_fn()
    X = gauss(ntrain, 101)
    Y = f(X)
    Yn2 = float(np.sum(Y * Y))
    out = {}

    # untied SwiGLU width 8
    # (the original also called M.fit here and discarded the result -- removed)
    best = (np.inf, None)
    for r in range(restarts):
        th0 = M.init(8, False, 500 + r)
        res = minimize(M.loss_grad, th0, args=(X, Y, 8, False, Yn2), jac=True,
                       method='L-BFGS-B',
                       options=dict(maxiter=4000, maxfun=8000, ftol=1e-16, gtol=1e-12))
        if res.fun < best[0]:
            best = (float(res.fun), res.x)
    out['untied'] = dict(kind='swiglu', w=8, loss=best[0], theta=best[1].tolist())

    # tied SiLU^2 width 12
    best = (np.inf, None)
    for r in range(restarts):
        th0 = M.init(12, True, 600 + r)
        res = minimize(M.loss_grad, th0, args=(X, Y, 12, True, Yn2), jac=True,
                       method='L-BFGS-B',
                       options=dict(maxiter=4000, maxfun=8000, ftol=1e-16, gtol=1e-12))
        if res.fun < best[0]:
            best = (float(res.fun), res.x)
    out['tied'] = dict(kind='silu2', w=12, loss=best[0], theta=best[1].tolist())

    # neutral tanh MLP width 12
    best = (np.inf, None)
    for r in range(restarts):
        L, th = fit_mlp(X, Y, 12, 700 + r)
        if L < best[0]:
            best = (L, th)
    out['neutral'] = dict(kind='mlp_tanh', w=12, loss=best[0], theta=best[1].tolist())

    json.dump(out, open('teachers.json', 'w'))
    for k, v in out.items():
        print(f"teacher {k:8s} {v['kind']:9s} w={v['w']:2d} "
              f"params=192 train rel.MSE={v['loss']:.4e}", flush=True)
    return out


def teacher_apply(spec, X):
    th = np.asarray(spec['theta'])
    if spec['kind'] == 'swiglu':
        return M.forward(th, X, spec['w'], False)
    if spec['kind'] == 'silu2':
        return M.forward(th, X, spec['w'], True)
    return mlp_forward(th, X, spec['w'])


if __name__ == '__main__':
    build()
