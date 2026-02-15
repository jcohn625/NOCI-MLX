# Non-Orthogonal Configuration Interaction (NOCI) 
Wave function ansatz is of the form:
$$|\psi\rangle = \sum_i^K c_i U_i|HF\rangle$$
where the $U_i$'s are RHF-type orbital rotations

* Matrix elements are computed with generalized Wick contractions
* MLX version is just a test beta...
  * Currently building a more optimized numba version which allows UHF and GHF type rotations
  * This version will be more scalable to larger system
```python
from pyscf import gto, scf, fci, ao2mo
from MLX_functions import *
from mol_obj import *
from NOCI import *

ints = RHFIntegrals(mol) # Takes pyscf mol object
ints.build()

sys = NOCISystem.from_mol_obj(ints) # convert ints objects to MLX arrays
noci = NOCI(sys, include_identity=True) # initalize NOCI class
noci.expand_X(k=20, lr=1e-2, inner_steps=50) # Greedily expand NOCI basis by optimizing overlap with current residual
noci.optimize_X(outer_steps=50, inner_steps=1, lr=5e-3, lowdin_every=1) # global optimization
```
