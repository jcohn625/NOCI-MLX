from mol_obj import *
from pyscf import gto, scf, fci, ao2mo
import mlx.core as mx
from mlx.optimizers import Adam
from functools import reduce
from MLX_functions import *


def noci_elements_rhf_mlx(R_i, R_j, phi0, h1, eri, e_nuc):
    
    # Build spatial determinants
    phi_i = R_i @ phi0
    phi_j = R_j @ phi0

    # Overlap in spatial sector
    s = phi_j.T @ phi_i
    det_s = custom_det(s)

    # Spin-orbital overlap
    S_ij = det_s**2

    # Transition 1-RDM (spatial)
    gamma = phi_i @ custom_inv(s) @ phi_j.conj().T

    # Spin-summed one-electron term
    e1 = 2 * mx.einsum('pq,qp->', h1, gamma.T, stream=mx.gpu)

    # Spin-summed two-electron term
    # term1 = 2 * mx.einsum('pqrs,qp,rs->', eri, gamma.T, gamma.T, stream=mx.gpu)
    # term2 =     mx.einsum('pqrs,qr,ps->', eri, gamma.T, gamma.T, stream=mx.gpu)

    term1 = 2 * mx.einsum('pqrs,qp,sr->', eri, gamma.T, gamma.T, stream=mx.gpu)
    term2 =     mx.einsum('pqrs,qr,sp->', eri, gamma.T, gamma.T, stream=mx.gpu)
    e2 = term1 - term2
    #print(e_nuc,e1,e2)
    E_loc = e1 + e2 +e_nuc  
    H_ij = S_ij * E_loc

    return S_ij, H_ij

def matrix_elmts(X):
    d = X.shape[0] # subspace dimension
    Hij = mx.zeros((d,d))
    Sij = mx.zeros((d,d))
    Us = [expm_skew(X[j,:,:]) for j in range(d)]
    for j in range(d):
        for k in range(j,d):
            Sij[j,k],Hij[j,k] = noci_elements_rhf_mlx(Us[j], Us[k], phi0, h1e, eri, e_nuc)
            if k>j:
                Sij[k,j] = Sij[j,k]
                Hij[k,j] = Hij[j,k]

    return Sij,Hij
    
    
def lowdin_diag(X):
    Sij,Hij = matrix_elmts(X)
    eigvals, U = mx.linalg.eigh(Sij,stream=mx.cpu)
    s_inv_sqrt = mx.diag(eigvals**-0.5)

    X = U @ s_inv_sqrt @ U.conj().T   
    H_orth = X.conj().T @ Hij @ X

    aa,bb = mx.linalg.eigh(H_orth,stream=mx.cpu)
    return X@bb[:,0]

def matrix_elmts_U(Us):
        d = len(Us)
        Hij = mx.zeros((d,d))
        Sij = mx.zeros((d,d))
        #Us = [expm_skew(X[j,:,:]) for j in range(d)]
        for j in range(d):
            for k in range(j,d):
                Sij[j,k],Hij[j,k] = noci_elements_rhf_mlx(Us[j], Us[k], phi0, h1e, eri, e_nuc)
                if k>j:
                    Sij[k,j] = Sij[j,k]
                    Hij[k,j] = Hij[j,k]
    
        return Sij,Hij

def lowdin_diag_U(Us):
    Sij,Hij = matrix_elmts_U(Us)
    eigvals, U = mx.linalg.eigh(Sij,stream=mx.cpu)
    s_inv_sqrt = mx.diag(eigvals**-0.5)

    X = U @ s_inv_sqrt @ U.conj().T   
    H_orth = X.conj().T @ Hij @ X

    aa,bb = mx.linalg.eigh(H_orth,stream=mx.cpu)
    return aa[0],X@bb[:,0]

