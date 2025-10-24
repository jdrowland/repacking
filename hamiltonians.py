import cirq
import numpy as np
from typing import Tuple, List
from itertools import product


def build_hamiltonian_matrix(hamiltonian: cirq.PauliSum, qubits: List[cirq.Qid]) -> np.ndarray:
    dim = 2 ** len(qubits)
    matrix = np.zeros((dim, dim), dtype=complex)

    for pauli_string in hamiltonian:
        coeff = pauli_string.coefficient
        term_matrix = np.eye(1, dtype=complex)

        for qubit in qubits:
            if qubit in pauli_string.qubits:
                gate = pauli_string[qubit]
                term_matrix = np.kron(term_matrix, cirq.unitary(gate))
            else:
                term_matrix = np.kron(term_matrix, np.eye(2))

        matrix += coeff * term_matrix

    return matrix


def create_one_qubit_hamiltonian() -> Tuple[cirq.PauliSum, list[cirq.Qid], np.ndarray]:
    q0 = cirq.LineQubit(0)
    qubits = [q0]

    hamiltonian = 0.5 * cirq.X(q0) + 0.3 * cirq.Z(q0)

    hamiltonian_matrix = np.array([[0.3, 0.5], [0.5, -0.3]])
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    ground_state = eigenvectors[:, 0]

    return hamiltonian, qubits, ground_state


def create_simple_test_hamiltonian() -> Tuple[cirq.PauliSum, list[cirq.Qid]]:
    q0 = cirq.LineQubit(0)
    qubits = [q0]
    hamiltonian = cirq.X(q0) + cirq.Z(q0)
    return hamiltonian, qubits


def get_exact_energy(hamiltonian: cirq.PauliSum, state: np.ndarray, qubits: list[cirq.Qid]) -> float:
    hamiltonian_matrix = build_hamiltonian_matrix(hamiltonian, qubits)
    energy = np.real(np.conj(state) @ hamiltonian_matrix @ state)
    return energy


def generate_all_pauli_strings(qubits: List[cirq.Qid]) -> List[cirq.PauliString]:
    n = len(qubits)
    pauli_gates = [cirq.I, cirq.X, cirq.Y, cirq.Z]
    all_paulis = []

    for gates in product(pauli_gates, repeat=n):
        if all(g == cirq.I for g in gates):
            continue

        pauli = cirq.PauliString()
        for qubit, gate in zip(qubits, gates):
            if gate != cirq.I:
                pauli *= gate(qubit)

        all_paulis.append(pauli)

    return all_paulis


def create_sparse_random_hamiltonian(
    num_qubits: int,
    sparsity: float = 0.1,
    coeff_range: Tuple[float, float] = (-1.0, 1.0),
    seed: int = None
) -> Tuple[cirq.PauliSum, List[cirq.Qid], np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    qubits = cirq.LineQubit.range(num_qubits)
    all_paulis = generate_all_pauli_strings(qubits)

    num_terms = int(len(all_paulis) * sparsity)
    selected_indices = np.random.choice(len(all_paulis), size=num_terms, replace=False)
    selected_paulis = [all_paulis[i] for i in selected_indices]

    coeffs = np.random.uniform(coeff_range[0], coeff_range[1], size=num_terms)

    hamiltonian = sum(p * float(c) for c, p in zip(coeffs, selected_paulis))

    hamiltonian_matrix = build_hamiltonian_matrix(hamiltonian, qubits)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    ground_state = eigenvectors[:, 0]

    return hamiltonian, qubits, ground_state
