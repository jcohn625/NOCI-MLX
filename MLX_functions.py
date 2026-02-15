import mlx.core as mx
import numpy as np



@mx.custom_function
def custom_inv(X):
    # Forward Pass: Return ONLY the result
    return mx.linalg.solve(X, mx.eye(X.shape[0]),stream=mx.cpu)

@custom_inv.vjp
def custom_inv_vjp(X, cotangent, X_inv):
    """
    X: The original input matrix
    cotangent: The gradient from the downstream loss (dL/dX_inv)
    X_inv: The output of the forward pass (re-used for efficiency)
    """
    # The gradient of X^-1 is -X^-T @ cotangent @ X^-T
    X_T = mx.transpose(X)
    
    # Solve X^T @ M = cotangent for stability
    grad_intermediate = mx.linalg.solve(X_T, cotangent,stream=mx.cpu)
    
    # Return the gradient w.r.t X
    return -mx.matmul(grad_intermediate, mx.transpose(X_inv))

def expm_skew(X):
    # 1. Create the skew-symmetric matrix S
    S = X - mx.transpose(X)
    
    # 2. Use eigenvalue decomposition. 
    # For a skew-symmetric matrix, we can use eigh on (S @ S.T) 
    # or use a Taylor expansion for a more direct autograd path.
    # BEST PRACTICE for Autograd: Use a Scaling and Squaring Taylor expansion.
    
    def taylor_expm(M, steps=8):
        # I + M + M^2/2! + M^3/3! ...
        res = mx.eye(M.shape[0])
        term = mx.eye(M.shape[0])
        for i in range(1, steps + 1):
            term = mx.matmul(term, M) / i
            res = res + term
        return res

    # Scaling and squaring trick to improve Taylor convergence
    scaling_factor = 4
    S_scaled = S / (2**scaling_factor)
    U = taylor_expm(S_scaled)
    
    for _ in range(scaling_factor):
        U = mx.matmul(U, U)
    return U

@mx.custom_function
def custom_eigh(A):
    # Forward Pass: Returns (eigenvalues, eigenvectors)
    return mx.linalg.eigh(A,stream=mx.cpu)

@custom_eigh.vjp
def custom_eigh_vjp(A, cotangents, output):
    """
    A: Input symmetric matrix
    cotangents: Tuple (dL/dw, dL/dV) where w=eigenvalues, V=eigenvectors
    output: Tuple (w, V) from forward pass
    """
    w, V = output
    dL_dw, dL_dV = cotangents
    
    # 1. Gradient from eigenvalues: V @ diag(dL_dw) @ V^T
    # Using broadcasting for efficiency
    grad_A = mx.matmul(V * dL_dw, V.T)
    
    # 2. Gradient from eigenvectors: V @ (F * (V^T @ dL_dV)) @ V^T
    # Compute F matrix: F_ij = 1 / (w_j - w_i) for i != j, else 0
    # Add small epsilon to denominator to avoid division by zero (degenerate case)
    diff = w[None, :] - w[:, None]
    F = 1.0 / (diff + 1e-12)
    mx.diagonal(F)[:] = 0.0 # Zero out diagonal where i == j
    
    # Symmetric gradient contribution from V
    V_T_dL_dV = mx.matmul(V.T, dL_dV)
    mid_term = F * V_T_dL_dV
    grad_A += mx.matmul(V, mx.matmul(mid_term, V.T))
    
    # Ensure the returned gradient is symmetric
    return 0.5 * (grad_A + grad_A.T)

@mx.custom_function
def custom_det(A):
    # MLX doesn't have mx.linalg.det, so we use LU-based prod(diag(U))
    LU, pivots = mx.linalg.lu_factor(A,stream=mx.cpu)
    diag_U = mx.diagonal(LU)
    
    # Sign adjustment for pivots (same as previous example)
    n = A.shape[-1]
    num_swaps = mx.sum(pivots != mx.arange(n, dtype=pivots.dtype))
    sign = mx.where(num_swaps % 2 == 0, 1.0, -1.0)
    
    return sign * mx.prod(diag_U)

@custom_det.vjp
def custom_det_vjp(A, cotangent, det_val):
    """
    A: Input matrix
    cotangent: The gradient of the loss with respect to det_val (dL/d_det)
    det_val: The output of the forward pass
    """
    # Gradient formula: dL/dA = (dL/d_det) * det(A) * (A^-1)^T
    # We solve A^T @ G = cotangent * det(A) * I
    # to find (A^-1)^T * (cotangent * det(A))
    
    A_T = mx.transpose(A)
    identity = mx.eye(A.shape[0])
    
    # Scale the identity by the cotangent and current determinant
    target = cotangent * det_val * identity
    
    # Solve for stability instead of explicit inversion
    grad_A = mx.linalg.solve(A_T, target,stream=mx.cpu)
    
    return grad_A