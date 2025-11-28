# Infinite-Dimensional QR Algorithm

Python implementation of the infinite-dimensional QR algorithm as described in:

**"On the infinite-dimensional QR algorithm"** by Matthew J. Colbrook and Anders C. Hansen  
*Numerische Mathematik* **143**, 17–83 (2019)  
DOI: https://doi.org/10.1007/s00211-019-01047-5

## Overview

This implementation provides tools for computing spectra and eigenvectors of infinite-dimensional operators on the canonical separable Hilbert space l²(ℕ). The algorithm can compute extremal parts of the spectrum and corresponding eigenvectors with convergence rates and error control.

### Key Features

- **Infinite-dimensional operator support**: Work with operators on l²(ℕ) through finite-dimensional approximations
- **Convergence control**: Adaptive dimension increase with error estimation
- **Extremal spectrum computation**: Compute largest or smallest eigenvalues and eigenvectors
- **Error control**: Built-in mechanisms for monitoring and controlling approximation errors

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from linear_operator import MatrixOperator
from iqr_algorithm import InfiniteQR
import numpy as np

# Create an operator (example: diagonal matrix)
matrix = np.diag([1, 2, 3, 4, 5])
operator = MatrixOperator(matrix)

# Initialize the infinite-dimensional QR algorithm
iqr = InfiniteQR(operator, initial_dim=10, max_iterations=100)

# Compute largest eigenvalues and eigenvectors
eigenvalues, eigenvectors = iqr.compute_extremal_spectrum(
    num_eigenvalues=5,
    which='largest'
)

print(f"Eigenvalues: {eigenvalues}")
```

## Usage Examples

### Example 1: Self-Adjoint Operator

```python
from examples import example_1_self_adjoint_operator

eigenvalues, eigenvectors = example_1_self_adjoint_operator()
```

### Example 2: Error Control

```python
from examples import example_4_error_control

eigenvalues, eigenvectors, error_estimates = example_4_error_control()
```

### Running All Examples

```python
from examples import run_all_examples

results = run_all_examples()
```

## Module Structure

- **`linear_operator.py`**: Base classes for operators on l²(ℕ)
  - `Operator`: Abstract base class
  - `MatrixOperator`: Operator from matrix representation
  - `FunctionalOperator`: Operator from functional action

- **`iqr_algorithm.py`**: Main algorithm implementation
  - `InfiniteQR`: Infinite-dimensional QR algorithm class

- **`examples.py`**: Example implementations demonstrating various use cases

## Algorithm Description

The infinite-dimensional QR algorithm extends the classical QR algorithm to infinite-dimensional operators. Key aspects:

1. **Finite-dimensional approximation**: Operators are approximated by finite-dimensional matrices
2. **QR iterations**: Standard QR decomposition and iteration steps
3. **Adaptive dimension**: Dimension can be increased adaptively for better accuracy
4. **Error control**: Convergence is monitored and dimension is increased until desired accuracy is achieved

### Convergence

The algorithm converges to extremal eigenvalues with the following properties:
- For self-adjoint operators: Two limits suffice (SCI = 1)
- For general operators: May require more limits
- Error control is available for certain operator classes

## Mathematical Background

Given a bounded linear operator T on l²(ℕ), the algorithm computes:

1. A finite-dimensional approximation T_n of size n × n
2. QR iterations: T_n^{(k+1)} = R_k Q_k where T_n^{(k)} = Q_k R_k
3. Convergence to extremal eigenvalues as k → ∞
4. Adaptive dimension increase: n → ∞ for error control

## References

- Colbrook, M.J., Hansen, A.C. On the infinite-dimensional QR algorithm. *Numer. Math.* **143**, 17–83 (2019). https://doi.org/10.1007/s00211-019-01047-5

## License

This implementation is provided for educational and research purposes. The original paper is published under Creative Commons Attribution 4.0 International License.

## Notes

- The implementation focuses on the computational aspects described in the paper
- For theoretical details and proofs, refer to the original paper
- Performance may vary depending on operator properties (self-adjoint, normal, compact, etc.)

