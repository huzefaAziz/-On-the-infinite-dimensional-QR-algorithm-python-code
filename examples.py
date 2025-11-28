"""
Example implementations demonstrating the infinite-dimensional QR algorithm.

These examples correspond to various operators discussed in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_operator import MatrixOperator, FunctionalOperator
from iqr_algorithm import InfiniteQR


def example_1_self_adjoint_operator():
    """
    Example 1: Self-adjoint operator with known spectrum.
    
    Consider a self-adjoint operator with eigenvalues 1, 2, 3, ...
    """
    print("Example 1: Self-adjoint Operator")
    print("=" * 50)
    
    # Create a diagonal self-adjoint operator
    n_init = 20
    diag = np.arange(1, n_init + 1, dtype=float)
    matrix = np.diag(diag)
    
    def extend_matrix(n):
        """Extend matrix to size n x n."""
        result = np.zeros((n, n))
        diag_extended = np.arange(1, n + 1, dtype=float)
        np.fill_diagonal(result, diag_extended)
        return result
    
    operator = MatrixOperator(matrix, extend_func=extend_matrix)
    
    # Create IQR algorithm instance
    iqr = InfiniteQR(operator, initial_dim=20, max_iterations=50)
    
    # Compute largest eigenvalues
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=5,
        which='largest'
    )
    
    print(f"Computed eigenvalues: {eigenvalues}")
    print(f"Expected eigenvalues: [20, 19, 18, 17, 16]")
    print(f"Error: {np.abs(eigenvalues - np.array([20, 19, 18, 17, 16]))}")
    
    return eigenvalues, eigenvectors


def example_2_toeplitz_operator():
    """
    Example 2: Toeplitz operator.
    
    A Toeplitz operator with symbol generating a specific spectrum.
    """
    print("\nExample 2: Toeplitz Operator")
    print("=" * 50)
    
    def toeplitz_action(i, j):
        """Define Toeplitz operator action."""
        k = i - j
        if k == 0:
            return 1.0
        elif abs(k) == 1:
            return 0.5
        else:
            return 0.0
    
    operator = FunctionalOperator(toeplitz_action, is_self_adjoint=True)
    
    iqr = InfiniteQR(operator, initial_dim=30, max_iterations=100)
    
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=5,
        which='largest'
    )
    
    print(f"Computed eigenvalues: {eigenvalues}")
    
    return eigenvalues, eigenvectors


def example_3_compact_operator():
    """
    Example 3: Compact operator with rapidly decaying eigenvalues.
    
    This demonstrates the algorithm on operators with eigenvalues
    that decay rapidly.
    """
    print("\nExample 3: Compact Operator")
    print("=" * 50)
    
    def compact_action(i, j):
        """Define compact operator with decaying eigenvalues."""
        if i == j:
            return 1.0 / (i + 1)**2
        else:
            return 0.0
    
    operator = FunctionalOperator(compact_action, is_self_adjoint=True)
    
    iqr = InfiniteQR(operator, initial_dim=25, max_iterations=100)
    
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=5,
        which='largest'
    )
    
    print(f"Computed eigenvalues: {eigenvalues}")
    print(f"Expected (approximate): [1.0, 0.25, 0.111..., 0.0625, 0.04]")
    
    return eigenvalues, eigenvectors


def example_4_error_control():
    """
    Example 4: Demonstrating error control mechanism.
    
    This shows how the algorithm adaptively increases dimension
    to achieve desired accuracy.
    """
    print("\nExample 4: Error Control Demonstration")
    print("=" * 50)
    
    # Create operator with known spectrum
    n_init = 10
    diag = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    matrix = np.diag(diag)
    
    def extend_matrix(n):
        result = np.zeros((n, n))
        diag_extended = np.arange(n, 0, -1, dtype=float)
        np.fill_diagonal(result, diag_extended)
        return result
    
    operator = MatrixOperator(matrix, extend_func=extend_matrix)
    
    iqr = InfiniteQR(operator, initial_dim=10, tolerance=1e-6)
    
    eigenvalues, eigenvectors, error_estimates = iqr.compute_spectrum_with_error_control(
        num_eigenvalues=5,
        which='largest',
        max_dim=50
    )
    
    print(f"Final eigenvalues: {eigenvalues}")
    print(f"Error estimates: {error_estimates}")
    print(f"Final error: {error_estimates[-1] if error_estimates else 'N/A'}")
    
    return eigenvalues, eigenvectors, error_estimates


def example_5_non_normal_operator():
    """
    Example 5: Non-normal operator.
    
    Demonstrates the algorithm on a non-normal operator.
    """
    print("\nExample 5: Non-Normal Operator")
    print("=" * 50)
    
    def non_normal_action(i, j):
        """Define a non-normal operator."""
        if i == j:
            return complex(i + 1, 0.1 * i)
        elif j == i + 1:
            return 0.5
        else:
            return 0.0
    
    operator = FunctionalOperator(non_normal_action, is_self_adjoint=False, is_normal=False)
    
    iqr = InfiniteQR(operator, initial_dim=20, max_iterations=100)
    
    eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
        num_eigenvalues=5,
        which='largest'
    )
    
    print(f"Computed eigenvalues: {eigenvalues}")
    
    return eigenvalues, eigenvectors


def plot_convergence(iqr: InfiniteQR, title: str = "Convergence History"):
    """
    Plot convergence history from IQR algorithm.
    
    Args:
        iqr: InfiniteQR instance with convergence history
        title: Plot title
    """
    conv_history, eig_history = iqr.get_convergence_history()
    
    if not conv_history:
        print("No convergence history available.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(conv_history, 'b-o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Max Eigenvalue Change')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def run_all_examples():
    """Run all examples and display results."""
    print("Running Infinite-Dimensional QR Algorithm Examples")
    print("=" * 60)
    
    results = {}
    
    try:
        results['example1'] = example_1_self_adjoint_operator()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        results['example2'] = example_2_toeplitz_operator()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        results['example3'] = example_3_compact_operator()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        results['example4'] = example_4_error_control()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        results['example5'] = example_5_non_normal_operator()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    
    return results


if __name__ == "__main__":
    run_all_examples()

