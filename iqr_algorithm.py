"""
Infinite-dimensional QR algorithm implementation.

This module implements the infinite-dimensional QR algorithm as described in:
"On the infinite-dimensional QR algorithm" by Colbrook & Hansen (2019).

The algorithm computes extremal parts of the spectrum and corresponding
eigenvectors with convergence rates and error control.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from scipy.linalg import qr, norm
from linear_operator import Operator


class InfiniteQR:
    """
    Infinite-dimensional QR algorithm for spectral computation.
    
    This class implements the infinite-dimensional QR algorithm that can
    compute extremal eigenvalues and eigenvectors of operators on l^2(N)
    with convergence rates and error control.
    """
    
    def __init__(self, operator: Operator, 
                 initial_dim: int = 10,
                 max_iterations: int = 100,
                 tolerance: float = 1e-10,
                 convergence_threshold: float = 1e-8):
        """
        Initialize the infinite-dimensional QR algorithm.
        
        Args:
            operator: The operator to compute spectrum for
            initial_dim: Initial dimension for finite approximation
            max_iterations: Maximum number of QR iterations
            tolerance: Tolerance for convergence
            convergence_threshold: Threshold for determining convergence
        """
        self.operator = operator
        self.initial_dim = initial_dim
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.convergence_threshold = convergence_threshold
        
        # Storage for convergence history
        self.convergence_history = []
        self.eigenvalue_history = []
    
    def compute_extremal_spectrum(self, 
                                  num_eigenvalues: int = 5,
                                  which: str = 'largest',
                                  adaptive_dim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute extremal eigenvalues and eigenvectors.
        
        Args:
            num_eigenvalues: Number of extremal eigenvalues to compute
            which: 'largest' or 'smallest' eigenvalues
            adaptive_dim: Whether to adaptively increase dimension
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvectors are columns
        """
        n = self.initial_dim
        
        # Get initial matrix approximation
        A = self.operator.get_matrix_approximation(n)
        
        # Ensure we compute enough eigenvalues
        k = min(num_eigenvalues, n - 1)
        
        eigenvalues = None
        eigenvectors = None
        
        for iteration in range(self.max_iterations):
            # Perform QR decomposition
            Q, R = qr(A)
            
            # Compute new iterate: A_{k+1} = RQ
            A = R @ Q
            
            # Extract eigenvalues (diagonal elements for convergence)
            current_eigenvalues = np.diag(A)
            
            # Sort eigenvalues by real part (for complex) or value (for real)
            if which == 'largest':
                # Sort by real part descending, then by absolute value
                idx = np.argsort(np.real(current_eigenvalues))[::-1]
            else:
                # Sort by real part ascending
                idx = np.argsort(np.real(current_eigenvalues))
            
            sorted_eigenvalues = current_eigenvalues[idx]
            
            # Check convergence
            if eigenvalues is not None:
                eigenvalue_change = np.abs(sorted_eigenvalues[:k] - eigenvalues[:k])
                max_change = np.max(eigenvalue_change)
                
                self.convergence_history.append(max_change)
                self.eigenvalue_history.append(sorted_eigenvalues[:k].copy())
                
                if max_change < self.convergence_threshold:
                    # Converged, extract eigenvectors
                    eigenvalues = sorted_eigenvalues[:k]
                    
                    # Compute eigenvectors by solving (A - lambda*I)v = 0
                    eigenvectors = self._compute_eigenvectors(A, eigenvalues, idx[:k])
                    
                    # If adaptive dimension and not enough eigenvalues, increase dimension
                    if adaptive_dim and len(eigenvalues) < num_eigenvalues:
                        n = min(n * 2, n + 20)  # Increase dimension
                        A = self.operator.get_matrix_approximation(n)
                        k = min(num_eigenvalues, n - 1)
                        eigenvalues = None
                        eigenvectors = None
                        continue
                    
                    break
            
            eigenvalues = sorted_eigenvalues[:k]
        
        # If not converged, use current approximation
        if eigenvectors is None:
            eigenvalues = sorted_eigenvalues[:k]
            eigenvectors = self._compute_eigenvectors(A, eigenvalues, idx[:k])
        
        return eigenvalues, eigenvectors
    
    def _compute_eigenvectors(self, 
                             A: np.ndarray, 
                             eigenvalues: np.ndarray,
                             indices: np.ndarray) -> np.ndarray:
        """
        Compute eigenvectors corresponding to given eigenvalues.
        
        Args:
            A: Matrix (after QR iterations)
            eigenvalues: Computed eigenvalues
            indices: Indices of eigenvalues in sorted order
            
        Returns:
            Matrix with eigenvectors as columns
        """
        n = A.shape[0]
        k = len(eigenvalues)
        eigenvectors = np.zeros((n, k), dtype=complex)
        
        for i, eigval in enumerate(eigenvalues):
            # Solve (A - lambda*I)v = 0
            # Use shifted QR or inverse iteration
            try:
                # Use inverse iteration for better stability
                I = np.eye(n)
                # Shift slightly to avoid singularity
                shift = eigval + 1e-10
                B = A - shift * I
                
                # Use QR to solve for eigenvector
                # Start with random vector
                v = np.random.randn(n) + 1j * np.random.randn(n)
                v = v / norm(v)
                
                # Inverse iteration
                for _ in range(5):
                    try:
                        v_new = np.linalg.solve(B, v)
                        v_new = v_new / norm(v_new)
                        v = v_new
                    except np.linalg.LinAlgError:
                        # If singular, use eigendecomposition
                        eigvals, eigvecs = np.linalg.eig(A)
                        closest_idx = np.argmin(np.abs(eigvals - eigval))
                        v = eigvecs[:, closest_idx]
                        break
                
                eigenvectors[:, i] = v
            except:
                # Fallback: use direct eigendecomposition
                eigvals, eigvecs = np.linalg.eig(A)
                closest_idx = np.argmin(np.abs(eigvals - eigval))
                eigenvectors[:, i] = eigvecs[:, closest_idx]
        
        return eigenvectors
    
    def compute_spectrum_with_error_control(self,
                                            num_eigenvalues: int = 5,
                                            which: str = 'largest',
                                            max_dim: int = 100) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Compute spectrum with error control by increasing dimension.
        
        This implements the error control mechanism described in the paper,
        where we increase the dimension until convergence is achieved.
        
        Args:
            num_eigenvalues: Number of eigenvalues to compute
            which: 'largest' or 'smallest'
            max_dim: Maximum dimension to try
            
        Returns:
            Tuple of (eigenvalues, eigenvectors, error_estimates)
        """
        dimensions = []
        eigenvalue_sequences = []
        error_estimates = []
        
        n = self.initial_dim
        
        while n <= max_dim:
            # Compute eigenvalues at current dimension
            self.initial_dim = n
            eigvals, eigvecs = self.compute_extremal_spectrum(
                num_eigenvalues=num_eigenvalues,
                which=which,
                adaptive_dim=False
            )
            
            dimensions.append(n)
            eigenvalue_sequences.append(eigvals.copy())
            
            # Estimate error by comparing with previous dimension
            if len(eigenvalue_sequences) > 1:
                prev_eigvals = eigenvalue_sequences[-2]
                error = np.max(np.abs(eigvals[:len(prev_eigvals)] - prev_eigvals))
                error_estimates.append(error)
                
                # Check if error is acceptable
                if error < self.tolerance:
                    break
            else:
                error_estimates.append(np.inf)
            
            # Increase dimension
            n = min(n * 2, n + 20, max_dim)
        
        # Return final results
        final_eigvals = eigenvalue_sequences[-1]
        final_eigvecs = eigvecs
        
        return final_eigvals, final_eigvecs, error_estimates
    
    def get_convergence_history(self) -> Tuple[List[float], List[np.ndarray]]:
        """
        Get convergence history from last computation.
        
        Returns:
            Tuple of (convergence_history, eigenvalue_history)
        """
        return self.convergence_history, self.eigenvalue_history

