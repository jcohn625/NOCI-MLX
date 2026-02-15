from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pyscf import scf, ao2mo


@dataclass
class RHFIntegrals:
    """Container for RHF + MO-basis 1e/2e integrals from a PySCF mol."""

    mol: "pyscf.gto.Mole"
    mf: Optional["pyscf.scf.hf.RHF"] = None

    # Stored properties (MO basis)
    C: Optional[np.ndarray] = None
    h1: Optional[np.ndarray] = None                 # (nmo, nmo)
    eri: Optional[np.ndarray] = None                # (nmo, nmo, nmo, nmo) chemist: (pq|rs)
    e_nuc: Optional[float] = None
    e_hf: Optional[float] = None

    def build(
        self,
        *,
        conv_tol: float = 1e-10,
        max_cycle: int = 100,
        verbose: int = 0,
        dtype=np.float64,
        use_density_fit: bool = False,
        auxbasis: str = "weigend",
        nmo: Optional[int] = None,
    ) -> "RHFIntegrals":
        """
        Run RHF and build MO-basis h1/eri/e_nuc.

        Args:
          use_density_fit: if True, runs mf.density_fit() before kernel().
          auxbasis: DF aux basis if use_density_fit=True.
          nmo: if set, truncate to first nmo MOs (useful for benchmarks / active-space-like tests).
        """
        # 1) RHF
        mf = scf.RHF(self.mol)
        mf.verbose = verbose
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        if use_density_fit:
            mf = mf.density_fit(auxbasis=auxbasis)
        e_hf = mf.kernel()

        if not mf.converged:
            raise RuntimeError("RHF did not converge. Try looser conv_tol/max_cycle or better initial guess.")

        C = np.asarray(mf.mo_coeff, dtype=dtype)
        norb = C.shape[1]
        if nmo is not None:
            if not (1 <= nmo <= norb):
                raise ValueError(f"nmo must be in [1,{norb}], got {nmo}")
            C = C[:, :nmo]
            norb = nmo

        # 2) MO-basis 1e integrals: h1 = C^T hcore C
        hcore = mf.get_hcore()
        h1 = (C.T @ hcore @ C).astype(dtype, copy=False)

        # 3) MO-basis 2e integrals: eri[p,q,r,s] = (pq|rs) in chemist notation
        eri = ao2mo.kernel(self.mol, C, compact=False)
        eri = np.asarray(eri, dtype=dtype).reshape(norb, norb, norb, norb)

        # 4) Nuclear repulsion
        e_nuc = float(mf.energy_nuc())

        # Save
        self.mf = mf
        self.C = C
        self.h1 = h1
        self.eri = eri
        self.e_nuc = e_nuc
        self.e_hf = float(e_hf)
        return self

    @property
    def norb(self) -> int:
        if self.h1 is None:
            raise AttributeError("Call .build() first.")
        return self.h1.shape[0]

    @property
    def nelec(self) -> Tuple[int, int]:
        # RHF mol.nelec returns (na, nb)
        return self.mol.nelec
