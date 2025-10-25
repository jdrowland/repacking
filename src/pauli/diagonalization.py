"""Diagonalization of commuting Pauli groups.

Ported from https://github.com/rmlarose/calibrate-ibm/blob/main/diagonalize.py
"""

from typing import List, Tuple
from itertools import product
import numpy as np
import cirq
from src.pauli.conjugation import conjugate_pauli_by_clifford


def group_commutes(stabilizer_matrix: np.ndarray) -> bool:
    """Test if a group commutes."""
    nq = stabilizer_matrix.shape[0] // 2
    j = np.zeros((2 * nq, 2 * nq), dtype=bool)
    for i in range(nq):
        j[i, i+nq] = True
        j[i+nq, i] = True
    ip = np.mod(stabilizer_matrix.T @ j @ stabilizer_matrix, 2).astype(bool)
    return np.all(np.invert(ip))


def get_stabilizer_matrix_from_paulis(stabilizers, qubits):
    """Convert Pauli strings to stabilizer matrix representation."""
    numq = len(qubits)
    nump = len(stabilizers)
    stabilizer_matrix = np.zeros((2*numq, nump))

    for i, paulistring in enumerate(stabilizers):
        for key, value in paulistring.items():
            # Extract qubit index
            qubit_idx = qubits.index(key)

            if value == cirq.X:
                stabilizer_matrix[qubit_idx + numq, i] = 1
            elif value == cirq.Y:
                stabilizer_matrix[qubit_idx, i] = 1
                stabilizer_matrix[qubit_idx + numq, i] = 1
            elif value == cirq.Z:
                stabilizer_matrix[qubit_idx, i] = 1

    return stabilizer_matrix


def binary_gaussian_elimination(matrix: np.ndarray) -> np.ndarray:
    """Do Gaussian elimination on the matrix to get it into RREF."""
    next_row = 0
    mat = matrix.copy()

    for j in range(mat.shape[1]):
        found = False
        for i in range(next_row, mat.shape[0]):
            if mat[i, j]:
                found = True
                if i != next_row:
                    temp = mat[next_row, :].copy()
                    mat[next_row, :] = mat[i, :]
                    mat[i, :] = temp
                break

        if found:
            for i in range(next_row+1, mat.shape[0]):
                if mat[i, j]:
                    mat[i, :] ^= mat[next_row, :]
            next_row += 1

    return mat


def binary_matrix_rank(mat: np.ndarray) -> int:
    """Get rank of binary matrix."""
    mat_reduced = binary_gaussian_elimination(mat)
    num_pivots = 0
    next_pivot = 0

    for j in range(mat_reduced.shape[1]):
        if next_pivot < mat_reduced.shape[0] - 1:
            all_zero_below = np.all(np.invert(mat_reduced[(next_pivot+1):, j]))
        else:
            all_zero_below = True
        if mat_reduced[next_pivot, j] and all_zero_below:
            num_pivots += 1
            next_pivot += 1

    return num_pivots


def get_linearly_independent_set(stabilizer_matrix: np.ndarray) -> np.ndarray:
    """Use Gaussian elimination to get linearly-independent set of vectors."""
    bool_sm = stabilizer_matrix.astype(bool)
    reduced_matrix = binary_gaussian_elimination(bool_sm)

    next_pivot = 0
    pivot_columns: List[int] = []
    for j in range(reduced_matrix.shape[1]):
        if next_pivot >= reduced_matrix.shape[0]:
            break
        if reduced_matrix[next_pivot, j]:
            pivot_columns.append(j)
            next_pivot += 1

    independent_columns = stabilizer_matrix[:, pivot_columns]
    return independent_columns


def get_measurement_circuit(stabilizer_matrix, qubits):
    """Generate measurement circuit from stabilizer matrix."""
    numq = len(stabilizer_matrix) // 2
    nump = len(stabilizer_matrix[0])
    z_matrix = stabilizer_matrix.copy()[:numq]
    x_matrix = stabilizer_matrix.copy()[numq:]

    measurement_circuit = cirq.Circuit()
    qreg = list(qubits)

    # Find combination of rows to make X matrix have rank nump
    for row_combination in product(['X', 'Z'], repeat=numq):
        candidate_matrix = np.array([
            z_matrix[i] if c=="Z" else x_matrix[i] for i, c in enumerate(row_combination)
        ])

        rank = binary_matrix_rank(candidate_matrix.astype(bool))
        if rank == nump:
            for i, c in enumerate(row_combination):
                if c == "Z":
                    z_matrix[i] = x_matrix[i]
                    measurement_circuit.append(cirq.H.on(qreg[i]))
            x_matrix = candidate_matrix
            break

    # Forward elimination
    for j in range(min(nump, numq)):
        if x_matrix[j,j] == 0:
            found = False
            for i in range(j + 1, numq):
                if x_matrix[i, j] != 0:
                    found = True
                    break

            if found:
                x_row = x_matrix[i].copy()
                x_matrix[i] = x_matrix[j]
                x_matrix[j] = x_row

                z_row = z_matrix[i].copy()
                z_matrix[i] = z_matrix[j]
                z_matrix[j] = z_row

                measurement_circuit.append(cirq.SWAP.on(qreg[j], qreg[i]))

        for i in range(j + 1, numq):
            if x_matrix[i,j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2
                measurement_circuit.append(cirq.CNOT.on(qreg[j], qreg[i]))

    # Backward elimination
    for j in range(nump-1, 0, -1):
        for i in range(j):
            if x_matrix[i, j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2
                measurement_circuit.append(cirq.CNOT.on(qreg[j], qreg[i]))

    # Eliminate Z matrix
    for i in range(nump):
        if z_matrix[i,i] == 1:
            for p in range(nump):
                z_matrix[i, p] = (z_matrix[i, p] + x_matrix[i, p]) % 2
            measurement_circuit.append(cirq.S.on(qreg[i]))

        for j in range(i):
            if z_matrix[i,j] == 1:
                for p in range(nump):
                    z_matrix[i, p] = (z_matrix[i, p] + x_matrix[j, p]) % 2
                    z_matrix[j, p] = (z_matrix[j, p] + x_matrix[i, p]) % 2
                measurement_circuit.append(cirq.CZ.on(qreg[j], qreg[i]))

    # Final Hadamards
    for i in range(nump):
        row = x_matrix[i].copy()
        x_matrix[i] = z_matrix[i]
        z_matrix[i] = row
        measurement_circuit.append(cirq.H.on(qreg[i]))

    return measurement_circuit, np.concatenate((z_matrix, x_matrix))


def is_pauli_diagonal(pstring: cirq.PauliString) -> bool:
    """Test if a PauliString is diagonal."""
    for _, pauli in pstring.items():
        if not (pauli == cirq.I or pauli == cirq.Z):
            return False
    return True


def diagonalize_pauli_strings(
    paulis: List[cirq.PauliString],
    qs: List[cirq.Qid]
) -> Tuple[cirq.Circuit, List[cirq.PauliString]]:
    """Diagonalize a set of Pauli strings.

    Returns:
        measurement_circuit: Circuit that rotates to measurement basis
        conjugated_strings: Pauli strings after rotation (all diagonal)
    """
    stabilizer_matrix = get_stabilizer_matrix_from_paulis(paulis, qs)

    # Remove identity columns if any
    for j in range(stabilizer_matrix.shape[1]):
        if np.all(np.invert(stabilizer_matrix[:, j].astype(bool))):
            stabilizer_matrix = np.delete(stabilizer_matrix, j, 1)
            break

    assert group_commutes(stabilizer_matrix), "Paulis do not commute!"

    reduced_stabilizer_matrix = get_linearly_independent_set(stabilizer_matrix)
    measurement_circuit, diag_stabilizer_matrix = get_measurement_circuit(
        reduced_stabilizer_matrix, qs
    )

    conjugated_strings: List[cirq.PauliString] = []
    for pstring in paulis:
        # Use manual conjugation to avoid Cirq bugs
        conjugated_string = conjugate_pauli_by_clifford(pstring, measurement_circuit, qs)
        assert is_pauli_diagonal(conjugated_string), \
            f"Pauli string {conjugated_string} is not diagonal. Originally was {pstring}"
        conjugated_strings.append(conjugated_string)

    return measurement_circuit, conjugated_strings
