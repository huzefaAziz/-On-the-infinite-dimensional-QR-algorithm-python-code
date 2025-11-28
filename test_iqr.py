"""
Simple tests for the infinite-dimensional QR algorithm implementation.
"""

import numpy as np
from linear_operator import MatrixOperator, FunctionalOperator
from iqr_algorithm import InfiniteQR


def test_diagonal_operator():
    """Test with a simple diagonal operator."""
    print("Test 1: Diagonal Operator")
    print("-" * 40)
    
    # Create diagonal matrix with eigenvalues 1, 2, 3, ..., 10
    n = 10
    matrix = np.diag(np.arange(1, n + 1, dtype=float))
    operator = MatrixOperator(matrix)
    
    iqr = InfiniteQR(operator, initial_dim=n, max_iterations=50)
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=3,
        which='largest'
    )
    
    expected = np.array([10.0, 9.0, 8.0])
    error = np.max(np.abs(eigenvalues - expected))
    
    print(f"Computed: {eigenvalues}")
    print(f"Expected: {expected}")
    print(f"Max error: {error}")
    
    assert error < 1e-6, f"Error too large: {error}"
    print("✓ Test passed!\n")


def test_self_adjoint_operator():
    """Test with a self-adjoint operator."""
    print("Test 2: Self-Adjoint Operator")
    print("-" * 40)
    
    # Create a symmetric matrix
    n = 15
    np.random.seed(42)
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric
    
    operator = MatrixOperator(A)
    
    iqr = InfiniteQR(operator, initial_dim=n, max_iterations=200, convergence_threshold=1e-10)
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=3,
        which='largest'
    )
    
    # Compare with numpy's eigendecomposition
    true_eigenvalues = np.linalg.eigvalsh(A)
    true_eigenvalues = np.sort(true_eigenvalues)[::-1][:3]
    
    error = np.max(np.abs(eigenvalues - true_eigenvalues))
    
    print(f"Computed: {eigenvalues}")
    print(f"True (numpy): {true_eigenvalues}")
    print(f"Max error: {error}")
    
    assert error < 2e-3, f"Error too large: {error}"
    print("✓ Test passed!\n")


def test_functional_operator():
    """Test with a functional operator."""
    print("Test 3: Functional Operator")
    print("-" * 40)
    
    def action(i, j):
        if i == j:
            return float(i + 1)
        return 0.0
    
    operator = FunctionalOperator(action, is_self_adjoint=True)
    
    iqr = InfiniteQR(operator, initial_dim=10, max_iterations=50)
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=3,
        which='largest'
    )
    
    expected = np.array([10.0, 9.0, 8.0])
    error = np.max(np.abs(eigenvalues - expected))
    
    print(f"Computed: {eigenvalues}")
    print(f"Expected: {expected}")
    print(f"Max error: {error}")
    
    assert error < 1e-5, f"Error too large: {error}"
    print("✓ Test passed!\n")


def test_error_control():
    """Test error control mechanism."""
    print("Test 4: Error Control")
    print("-" * 40)
    
    n = 8
    matrix = np.diag(np.arange(1, n + 1, dtype=float))
    
    def extend_matrix(size):
        result = np.zeros((size, size))
        diag = np.arange(1, size + 1, dtype=float)
        np.fill_diagonal(result, diag)
        return result
    
    operator = MatrixOperator(matrix, extend_func=extend_matrix)
    
    iqr = InfiniteQR(operator, initial_dim=n, tolerance=1e-6)
    eigenvalues, eigenvectors, error_estimates = iqr.compute_spectrum_with_error_control(
        num_eigenvalues=3,
        which='largest',
        max_dim=20
    )
    
    print(f"Final eigenvalues: {eigenvalues}")
    print(f"Error estimates: {error_estimates}")
    print(f"Final error: {error_estimates[-1] if error_estimates else 'N/A'}")
    
    assert len(eigenvalues) == 3, "Should compute 3 eigenvalues"
    print("✓ Test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Infinite-Dimensional QR Algorithm Tests")
    print("=" * 50)
    print()
    
    try:
        test_diagonal_operator()
        test_self_adjoint_operator()
        test_functional_operator()
        test_error_control()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

