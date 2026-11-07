"""Coefficient-space fitting of tied and untied CP-MLP atoms (Appendix D.9-D.11).

A homogeneous degree-k polynomial in d variables is stored as its symmetric
coefficient tensor T in (R^d)^{(x)k}. The Frobenius norm of T is the apolar norm
used in Appendix D.9, so all losses here are relative squared apolar errors

    loss = ||P - T||_F^2 / ||T||_F^2 .

Atom families (scalar output, so the output factor is a coefficient c_a):

    tied   : c_a * u_a^{(x)k}
    untied : c_a * Sym(u_a^{(x)(k-1)} (x) v_a)

Gradients are analytic. They use the identity <R, Sym(X)> = <R, X> for a
symmetric residual R, so every inner product reduces to a multilinear
contraction of R with the atom's vectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Symmetric tensors
# ---------------------------------------------------------------------------

def outer_power(u: np.ndarray, k: int) -> np.ndarray:
    t = np.ones(())
    for _ in range(k):
        t = np.multiply.outer(t, u)
    return t


def sym_atom_untied(u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """Sym(u^{(x)(k-1)} (x) v): average over the k slots where v can sit."""
    d = u.shape[0]
    out = np.zeros((d,) * k)
    for j in range(k):
        t = np.ones(())
        for s in range(k):
            t = np.multiply.outer(t, v if s == j else u)
        out += t
    return out / k


def contract(R: np.ndarray, vecs: list[np.ndarray]) -> np.ndarray:
    """Contract the leading modes of R with the given vectors (in order)."""
    t = R
    for x in vecs:
        t = np.tensordot(x, t, axes=([0], [0]))
    return t


def symmetrize(T: np.ndarray) -> np.ndarray:
    import itertools
    k = T.ndim
    perms = list(itertools.permutations(range(k)))
    return sum(np.transpose(T, p) for p in perms) / len(perms)


def monomial_tensor(d: int, exps: dict[tuple[int, ...], float]) -> np.ndarray:
    """Symmetric tensor of sum_alpha f_alpha x^alpha.

    `exps` maps an exponent tuple alpha (length d) to its coefficient f_alpha.
    The tensor entry for each index tuple with multiplicity alpha equals
    f_alpha * alpha! / k!, so that T[x, ..., x] reproduces the polynomial.
    """
    k = sum(next(iter(exps)))
    T = np.zeros((d,) * k)
    import itertools
    for idx in itertools.product(range(d), repeat=k):
        alpha = tuple(idx.count(i) for i in range(d))
        if alpha in exps:
            mult = math.factorial(k) / np.prod([math.factorial(a) for a in alpha])
            T[idx] = exps[alpha] / mult
    return T


def random_symmetric(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian coefficients in an apolar-orthonormal basis of Sym^k(R^d).

    Projecting an isotropic Gaussian tensor onto the symmetric subspace gives
    an isotropic Gaussian on that subspace.
    """
    return symmetrize(rng.standard_normal((d,) * k))


def sym_dim(d: int, k: int) -> int:
    return math.comb(d + k - 1, k)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Width-m tied or untied family fitted to one or several degrees.

    degrees: tuple of degrees K. With |K| > 1 the channels share directions
    (u, and v for untied) and have a free coefficient per degree (D.11).
    """
    family: str            # "tied" or "untied"
    d: int
    m: int
    degrees: tuple[int, ...]

    @property
    def n_params(self) -> int:
        per = self.d if self.family == "tied" else 2 * self.d
        return self.m * (per + len(self.degrees))

    def unpack(self, theta):
        m, d, nk = self.m, self.d, len(self.degrees)
        U = theta[: m * d].reshape(m, d)
        off = m * d
        V = None
        if self.family == "untied":
            V = theta[off: off + m * d].reshape(m, d)
            off += m * d
        C = theta[off: off + m * nk].reshape(m, nk)
        return U, V, C

    def atom(self, u, v, k):
        return outer_power(u, k) if self.family == "tied" else sym_atom_untied(u, v, k)

    # -- vectorised helpers over channels a --------------------------------
    @staticmethod
    def _batched_outer(vecs):
        """vecs: list of (m, d) arrays -> (m, d, ..., d) batched outer product."""
        t = vecs[0]
        for x in vecs[1:]:
            t = t[..., None] * x.reshape((x.shape[0],) + (1,) * (t.ndim - 1) + (x.shape[1],))
        return t

    def _atoms(self, U, V, k):
        """(m, d^k) batched atoms for degree k."""
        if self.family == "tied":
            return self._batched_outer([U] * k)
        out = 0.0
        for j in range(k):
            out = out + self._batched_outer([V if s == j else U for s in range(k)])
        return out / k

    @staticmethod
    def _bcontract(R, vecs):
        """Batched contraction: R[x_a^(1), ..., x_a^(r), .] for each channel a."""
        t = np.tensordot(vecs[0], R, axes=([1], [0]))          # (m, ...)
        for x in vecs[1:]:
            t = np.einsum("ai,ai...->a...", x, t)
        return t

    def predict(self, theta) -> list[np.ndarray]:
        U, V, C = self.unpack(theta)
        return [np.tensordot(C[:, j], self._atoms(U, V, k), axes=(0, 0))
                for j, k in enumerate(self.degrees)]

    def sizes(self, theta) -> np.ndarray:
        """Homogeneous atom sizes A_a (single-degree case)."""
        U, V, C = self.unpack(theta)
        k = self.degrees[0]
        nu = np.linalg.norm(U, axis=1)
        if self.family == "tied":
            return np.abs(C[:, 0]) * nu ** k
        return np.abs(C[:, 0]) * nu ** (k - 1) * np.linalg.norm(V, axis=1)

    def loss_grad(self, theta, targets, lam=0.0):
        """Relative apolar loss (summed over degrees) + lam * sum_a A_a^2."""
        U, V, C = self.unpack(theta)
        P = self.predict(theta)
        denom = sum(float(np.sum(T * T)) for T in targets)
        gU = np.zeros_like(U)
        gV = None if V is None else np.zeros_like(V)
        gC = np.zeros_like(C)
        loss = 0.0
        for j, (k, T, Pk) in enumerate(zip(self.degrees, targets, P)):
            R = Pk - T
            loss += float(np.sum(R * R)) / denom
            s = 2.0 / denom
            if self.family == "tied":
                r_um1 = self._bcontract(R, [U] * (k - 1)) if k > 1 else np.broadcast_to(R, (self.m,) + R.shape)
                gC[:, j] += s * np.einsum("ai,ai->a", r_um1, U)
                gU += s * (C[:, j] * k)[:, None] * r_um1
            else:
                r_um1 = self._bcontract(R, [U] * (k - 1)) if k > 1 else np.broadcast_to(R, (self.m,) + R.shape)
                gC[:, j] += s * np.einsum("ai,ai->a", r_um1, V)
                gV += s * C[:, j][:, None] * r_um1
                if k >= 2:
                    r_um2_v = self._bcontract(R, [U] * (k - 2) + [V])
                    gU += s * (C[:, j] * (k - 1))[:, None] * r_um2_v
        if lam > 0.0:
            assert len(self.degrees) == 1
            k = self.degrees[0]
            nu2 = np.sum(U * U, axis=1)
            c = C[:, 0]
            if self.family == "tied":
                A2 = c ** 2 * nu2 ** k
                loss += lam * float(A2.sum())
                gC[:, 0] += lam * 2 * c * nu2 ** k
                gU += lam * (c ** 2 * k * nu2 ** (k - 1))[:, None] * 2 * U
            else:
                nv2 = np.sum(V * V, axis=1)
                A2 = c ** 2 * nu2 ** (k - 1) * nv2
                loss += lam * float(A2.sum())
                gC[:, 0] += lam * 2 * c * nu2 ** (k - 1) * nv2
                gU += lam * (c ** 2 * (k - 1) * nu2 ** max(k - 2, 0) * nv2)[:, None] * 2 * U
                gV += lam * (c ** 2 * nu2 ** (k - 1))[:, None] * 2 * V
        grad = [gU.ravel()]
        if gV is not None:
            grad.append(gV.ravel())
        grad.append(gC.ravel())
        return loss, np.concatenate(grad)

    def data_loss(self, theta, targets) -> float:
        P = self.predict(theta)
        denom = sum(float(np.sum(T * T)) for T in targets)
        return sum(float(np.sum((Pk - T) ** 2)) for Pk, T in zip(P, targets)) / denom


def fit(model: Model, targets, rng, theta0=None, lam=0.0, init_scale=1.0,
        maxiter=20000):
    """One L-BFGS-B run. Returns (data_loss, theta)."""
    if theta0 is None:
        theta0 = init_scale * rng.standard_normal(model.n_params)
    res = minimize(model.loss_grad, theta0, args=(targets, lam), jac=True,
                   method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter,
                            "ftol": 1e-32, "gtol": 1e-18, "maxcor": 30})
    return model.data_loss(res.x, targets), res.x


def best_of(model, targets, rng, restarts, **kw):
    best = (np.inf, None)
    losses = []
    for _ in range(restarts):
        L, th = fit(model, targets, rng, **kw)
        losses.append(L)
        if L < best[0]:
            best = (L, th)
    return best[0], best[1], losses


# ---------------------------------------------------------------------------
# Intrinsic condition number (Breiding-Vannieuwenhoven style)
# ---------------------------------------------------------------------------

def _atom_jacobian(model: Model, u, v, c, k, eps=1e-7):
    """Columns: derivatives of the vectorised atom c*atom(u,v) w.r.t. (u, v, c)."""
    def f(p):
        d = model.d
        uu = p[:d]
        vv = p[d:2 * d] if model.family == "untied" else None
        cc = p[-1]
        return (cc * model.atom(uu, vv, k)).ravel()
    p0 = np.concatenate([u] + ([v] if v is not None else []) + [[c]])
    cols = []
    for i in range(p0.size):
        e = np.zeros_like(p0)
        e[i] = eps
        cols.append((f(p0 + e) - f(p0 - e)) / (2 * eps))
    return np.stack(cols, axis=1)


def condition_number(model: Model, theta, tol=1e-6):
    """kappa = 1 / sigma_min of the stacked orthonormal tangent bases.

    Each atom contributes an orthonormal basis of the tangent space of its
    family at that atom (scaling redundancy removed by taking the column space).
    Returns (kappa, sigma_min). kappa is None ("n.i.") when the tangent spaces
    are linearly dependent, i.e. the decomposition is not locally unique.
    """
    U, V, C = model.unpack(theta)
    k = model.degrees[0]
    bases = []
    for a in range(model.m):
        J = _atom_jacobian(model, U[a], None if V is None else V[a], C[a, 0], k)
        Uj, s, _ = np.linalg.svd(J, full_matrices=False)
        r = int(np.sum(s > tol * s[0]))
        bases.append(Uj[:, :r])
    B = np.concatenate(bases, axis=1)
    s = np.linalg.svd(B, compute_uv=False)
    smin = float(s[-1])
    if B.shape[1] > sym_dim(model.d, k) or smin < 1e-12:
        return None, smin
    return 1.0 / smin, smin
