from mol_obj import *
from pyscf import gto, scf, fci, ao2mo
import mlx.core as mx
from mlx.optimizers import Adam
from functools import reduce
from MLX_functions import *
from noci_funcs import *


import time
import mlx.core as mx
from dataclasses import dataclass

# Assumed available from your codebase:
# - expm_skew(X): returns U = exp(X) for skew/antisymmetric X
# - noci_elements_rhf_mlx(Ui, Uj, phi0, h1e, eri, e_nuc): returns (Sij, Hij)
# - Adam optimizer from mlx.optimizers (or your wrapper)


@dataclass
class NOCISystem:
    norb: int
    nelec: tuple  # (nalpha, nbeta)
    phi0: mx.array
    h1: mx.array
    eri: mx.array
    e_nuc: mx.array

    @classmethod
    def from_mol_obj(cls, mol_obj):
        norb = int(mol_obj.norb)

        # electron counts
        # adapt to your RHFIntegrals attribute names:
        if hasattr(mol_obj, "nelec"):
            nelec = tuple(mol_obj.nelec)  # (na, nb)
        else:
            nelec = (int(mol_obj.N_alpha), int(mol_obj.N_beta))

        # build or load phi0
        if hasattr(mol_obj, "phi0"):
            phi0 = mx.array(mol_obj.phi0)
        else:
            nocc = int(nelec[0])  # RHF: use alpha count
            phi0 = mx.eye(norb)[:, :nocc]

        return cls(
            norb=norb,
            nelec=nelec,
            phi0=phi0,
            h1=mx.array(mol_obj.h1),
            eri=mx.array(mol_obj.eri),
            e_nuc=mx.array(mol_obj.e_nuc),
        )


class NOCI:
    """
    Holds current NOCI subspace (Xs/Us/C/E0) and provides:
      - greedy expansion (residual maximization)
      - joint optimization at fixed NU
    """
    def __init__(self, system: NOCISystem, Xs_init=None, include_identity=True):
        self.sys = system
        self.norb = system.norb

        # Current subspace parameters (Xs) and rotations (Us)
        self.Xs = None   # shape (NU, norb, norb)
        self.Us = []     # list of U matrices
        self.C = None
        self.E0 = None

        # Initialize
        if Xs_init is None:
            Xs_init = mx.zeros((0, self.norb, self.norb))

        if include_identity:
            # identity determinant corresponds to X = 0
            X0 = mx.zeros((1, self.norb, self.norb))
            self.Xs = mx.concatenate([X0, Xs_init], axis=0)
        else:
            self.Xs = Xs_init

        self._refresh_Us()
        self._solve_lowdin()

    # ---------- core linear algebra / NOCI bookkeeping ----------

    def _refresh_Us(self):
        """Recompute Us from Xs."""
        NU = self.Xs.shape[0]
        # list comprehension is ok; MLX will build graph; you can also stack if preferred
        self.Us = [expm_skew(self.Xs[j, :, :]) for j in range(NU)]

    def _build_S_H_from_Us(self, Us):
        d = len(Us)
        Hij = mx.zeros((d, d))
        Sij = mx.zeros((d, d))
        for j in range(d):
            for k in range(j, d):
                s, h = noci_elements_rhf_mlx(
                    Us[j], Us[k],
                    self.sys.phi0, self.sys.h1, self.sys.eri, self.sys.e_nuc
                )
                Sij[j, k] = s
                Hij[j, k] = h
                if k > j:
                    Sij[k, j] = s
                    Hij[k, j] = h
        return Sij, Hij

    def _lowdin_diag(self, Sij, Hij, eps=1e-12):
        w, U = mx.linalg.eigh(Sij, stream=mx.cpu)
    
        # Clamp: anything below eps is set to eps
        w_safe = mx.maximum(w, eps)
        s_inv_sqrt = mx.diag(w_safe**-0.5)
    
        X = U @ s_inv_sqrt @ U.conj().T
        H_orth = X.conj().T @ Hij @ X
    
        e, V = mx.linalg.eigh(H_orth, stream=mx.cpu)
        C = X @ V[:, 0]
        return e[0], C


    def _solve_lowdin(self):
        """Update E0 and C for current Us."""
        Sij, Hij = self._build_S_H_from_Us(self.Us)
        E0, C = self._lowdin_diag(Sij, Hij)
        self.E0, self.C = E0, C
        return E0, C

    def energy_from_state(self):
        return self.E0

    # ---------- greedy expansion: add one determinant at a time ----------

    def _greedy_loss_factory(self, E0, C, Us_current):
        """
        Build the greedy objective:
          r(X) = sum_i C[i] * ( h(U_new, U_i) - E0 * s(U_new, U_i) )
          loss = - r(X)^2
        """
        Ns = len(Us_current)

        def loss(params):
            U_new = expm_skew(params["X"])
            r = mx.array(0.0)
            for i in range(Ns):
                s, h = noci_elements_rhf_mlx(
                    U_new, Us_current[i],
                    self.sys.phi0, self.sys.h1, self.sys.eri, self.sys.e_nuc
                )
                r = r + C[i] * (h - E0 * s)
            return -r * r

        return loss

    def expand_X(self, k=1, lr=1e-2, inner_steps=100, seed=None, verbose=True):
        """
        Greedily add k new rotations:
          1) optimize new X to maximize residual
          2) append to subspace
          3) re-solve Löwdin -> new (E0, C)
        """
        if seed is not None:
            mx.random.seed(seed)

        for t in range(k):
            E0 = self.E0
            C  = self.C
            Us_current = list(self.Us)  # capture current subspace

            # init new skew param
            X_new = {"X": mx.random.normal((self.norb, self.norb))}

            loss = self._greedy_loss_factory(E0, C, Us_current)
            loss_fn = mx.compile(loss)
            grad_fn = mx.grad(loss_fn, argnums=0)

            # optimizer (replace with your Adam)
            optimizer = Adam(learning_rate=lr)

            t0 = time.time()
            for step in range(inner_steps):
                grads = grad_fn(X_new)
                L = loss_fn(X_new)
                X_new = optimizer.apply_gradients(grads, X_new)
                mx.eval(L)
            t1 = time.time()

            # append new X / U
            self.Xs = mx.concatenate([self.Xs, X_new["X"][None, :, :]], axis=0)
            self.Us.append(expm_skew(X_new["X"]))

            # re-solve coefficients/energy
            E0_new, C_new = self._solve_lowdin()

            if verbose:
                print(f"[expand {t+1}/{k}] greedy loss={L.item(): .6e}  "
                      f"elapsed={t1-t0: .2f}s  E0={float(E0_new): .12f}")

        return self.E0, self.C

    # ---------- joint optimization at fixed NU ----------

    def _full_energy_loss_factory(self, C_fixed):
        """
        Given fixed coefficients C (e.g., from Löwdin), return
          E(X) = (C^T H(X) C) / (C^T S(X) C)
        which matches your loss(params, C).
        """
        NU = self.Xs.shape[0]

        def loss(params):
            Us = [expm_skew(params["Xs"][j, :, :]) for j in range(NU)]
            E = mx.array(0.0)
            D = mx.array(0.0)
            for j in range(NU):
                for k in range(j, NU):
                    s, h = noci_elements_rhf_mlx(
                        Us[j], Us[k],
                        self.sys.phi0, self.sys.h1, self.sys.eri, self.sys.e_nuc
                    )
                    if j == k:
                        E = E + C_fixed[j] * C_fixed[k] * h
                        D = D + C_fixed[j] * C_fixed[k] * s
                    else:
                        E = E + 2 * C_fixed[j] * C_fixed[k] * h
                        D = D + 2 * C_fixed[j] * C_fixed[k] * s
            return E / D

        return loss

    def _full_energy_loss(self, params, C):
        NU = params["Xs"].shape[0]
        Us = [expm_skew(params["Xs"][j, :, :]) for j in range(NU)]
    
        E = mx.array(0.0)
        D = mx.array(0.0)
    
        for j in range(NU):
            for k in range(j, NU):
                s, h = noci_elements_rhf_mlx(
                    Us[j], Us[k],
                    self.sys.phi0, self.sys.h1, self.sys.eri, self.sys.e_nuc
                )
                if j == k:
                    E = E + C[j] * C[k] * h
                    D = D + C[j] * C[k] * s
                else:
                    E = E + 2 * C[j] * C[k] * h
                    D = D + 2 * C[j] * C[k] * s
        return E / D

    def optimize_X(self, outer_steps=20, inner_steps=50, lr=1e-2, lowdin_every=1, verbose=True):
        params = {"Xs": self.Xs}
        optimizer = Adam(learning_rate=lr)
    
        # compile ONCE (key point)
        loss_fn = mx.compile(self._full_energy_loss)
        grad_fn = mx.grad(loss_fn, argnums=0)
    
        for it in range(outer_steps):
            if (it % lowdin_every) == 0:
                self.Xs = params["Xs"]
                self._refresh_Us()
                self._solve_lowdin()
    
            C_fixed = self.C  # snapshot for this outer iter
    
            for step in range(inner_steps):
                grads = grad_fn(params, C_fixed)
                L = loss_fn(params, C_fixed)
                params = optimizer.apply_gradients(grads, params)
                mx.eval(L)
    
            if verbose:
                print(f"[opt it={it+1}/{outer_steps}] loss(E)={L.item(): .12f}  current E0={float(self.E0): .12f}")
    
        self.Xs = params["Xs"]
        self._refresh_Us()
        self._solve_lowdin()
        return self.E0, self.C

    # def optimize_X(self, outer_steps=20, inner_steps=50, lr=1e-2, lowdin_every=1, verbose=True):
    #     """
    #     Practical joint optimization strategy (stable in practice):
    #       - Alternate:
    #           (A) update C via Löwdin on current Xs (every lowdin_every outer steps)
    #           (B) take several gradient steps on Xs holding C fixed
    #     This matches your “full optimization” style but keeps it stable.
    #     """
    #     params = {"Xs": self.Xs}
    #     optimizer = Adam(learning_rate=lr)

    #     for it in range(outer_steps):
    #         if (it % lowdin_every) == 0:
    #             # ensure Us matches Xs before solving
    #             self.Xs = params["Xs"]
    #             self._refresh_Us()
    #             self._solve_lowdin()

    #         C_fixed = self.C
    #         loss = self._full_energy_loss(params,C_fixed)
    #         loss_fn = mx.compile(loss)
    #         grad_fn = mx.grad(loss_fn, argnums=0)

    #         for step in range(inner_steps):
    #             grads = grad_fn(params,C_fixed)
    #             L = loss_fn(params,C_fixed)
    #             params = optimizer.apply_gradients(grads, params)
    #             mx.eval(L)

    #         if verbose:
    #             print(f"[opt it={it+1}/{outer_steps}] "
    #                   f"loss(E)={L.item(): .12f}  current E0={float(self.E0): .12f}")

    #     # finalize internal state
    #     self.Xs = params["Xs"]
    #     self._refresh_Us()
    #     self._solve_lowdin()
    #     return self.E0, self.C

    
        

    

    

    

    