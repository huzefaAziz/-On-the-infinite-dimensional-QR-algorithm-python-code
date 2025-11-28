"""
Operator classes for infinite-dimensional operators on l^2(N).

This module provides base classes and utilities for working with
bounded linear operators on the canonical separable Hilbert space l^2(N).
"""

import numpy as np
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod


class Operator(ABC):
    """
    Abstract base class for bounded linear operators on l^2(N).
    
    An operator T: l^2(N) -> l^2(N) is represented by its action on
    sequences in l^2(N).
    """
    
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the operator to a vector x in l^2(N).
        
        Args:
            x: Input vector (finite-dimensional approximation)
            
        Returns:
            Result of applying the operator to x
        """
        pass
    
    @abstractmethod
    def get_matrix_approximation(self, n: int) -> np.ndarray:
        """
        Get a finite-dimensional matrix approximation of size n x n.
        
        Args:
            n: Size of the approximation
            
        Returns:
            n x n matrix approximation
        """
        pass
    
    def is_self_adjoint(self) -> bool:
        """Check if the operator is self-adjoint."""
        return False
    
    def is_normal(self) -> bool:
        """Check if the operator is normal (T*T = TT*)."""
        return False


class MatrixOperator(Operator):
    """
    Operator defined by a matrix representation.
    
    For infinite-dimensional operators, this represents a finite
    truncation that can be extended.
    """
    
    def __init__(self, matrix: np.ndarray, extend_func: Optional[Callable] = None):
        """
        Initialize operator from a matrix.
        
        Args:
            matrix: Initial matrix representation
            extend_func: Optional function to extend matrix for larger dimensions
        """
        self.matrix = matrix
        self.extend_func = extend_func
        self._is_self_adjoint = None
        self._is_normal = None
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply operator to vector x."""
        n = len(x)
        if n > self.matrix.shape[0]:
            if self.extend_func:
                extended_matrix = self.extend_func(n)
            else:
                # Default: pad with zeros
                extended_matrix = np.zeros((n, n))
                extended_matrix[:self.matrix.shape[0], :self.matrix.shape[1]] = self.matrix
            return extended_matrix @ x
        return self.matrix[:n, :n] @ x
    
    def get_matrix_approximation(self, n: int) -> np.ndarray:
        """Get n x n matrix approximation."""
        if n <= self.matrix.shape[0]:
            return self.matrix[:n, :n]
        elif self.extend_func:
            return self.extend_func(n)
        else:
            # Pad with zeros
            result = np.zeros((n, n))
            result[:self.matrix.shape[0], :self.matrix.shape[1]] = self.matrix
            return result
    
    def is_self_adjoint(self) -> bool:
        """Check if matrix is Hermitian."""
        if self._is_self_adjoint is None:
            self._is_self_adjoint = np.allclose(self.matrix, self.matrix.conj().T)
        return self._is_self_adjoint


class FunctionalOperator(Operator):
    """
    Operator defined by a functional representation.
    
    The operator is defined by how it acts on basis elements.
    """
    
    def __init__(self, action: Callable[[int, int], complex], 
                 is_self_adjoint: bool = False, is_normal: bool = False):
        """
        Initialize operator from functional action.
        
        Args:
            action: Function (i, j) -> T_{ij} giving matrix elements
            is_self_adjoint: Whether operator is self-adjoint
            is_normal: Whether operator is normal
        """
        self.action = action
        self._is_self_adjoint = is_self_adjoint
        self._is_normal = is_normal
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply operator to vector x."""
        n = len(x)
        result = np.zeros(n, dtype=complex)
        for i in range(n):
            for j in range(n):
                result[i] += self.action(i, j) * x[j]
        return result
    
    def get_matrix_approximation(self, n: int) -> np.ndarray:
        """Get n x n matrix approximation."""
        matrix = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.action(i, j)
        return matrix
    
    def is_self_adjoint(self) -> bool:
        """Check if operator is self-adjoint."""
        return self._is_self_adjoint
    
    def is_normal(self) -> bool:
        """Check if operator is normal."""
        return self._is_normal

