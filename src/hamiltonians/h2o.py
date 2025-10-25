"""H2O Hamiltonian loading utilities."""
import openfermion as of
import cirq
import numpy as np
from typing import Tuple, List


def load_h2o_hamiltonian(hdf5_file: str = "monomer_eqb.hdf5") -> Tuple[cirq.PauliSum, List[cirq.Qid]]:
    """Load H2O Hamiltonian from HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file containing molecular data

    Returns:
        hamiltonian_cirq: The Hamiltonian as a Cirq PauliSum
        qubits: List of qubits (14 qubits for H2O)
    """
    hamiltonian_fermion = of.jordan_wigner(
        of.get_fermion_operator(
            of.chem.MolecularData(filename=hdf5_file).get_molecular_hamiltonian()
        )
    )
    hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
    qubits = cirq.LineQubit.range(14)
    return hamiltonian_cirq, qubits


def compute_hf_energy(hamiltonian: cirq.PauliSum, qubits: List[cirq.Qid], n_electrons: int) -> float:
    """Compute exact Hartree-Fock energy for a Hamiltonian.

    Args:
        hamiltonian: The Hamiltonian as a Cirq PauliSum
        qubits: List of qubits
        n_electrons: Number of electrons

    Returns:
        The Hartree-Fock energy
    """
    hf_energy = 0.0
    for pauli_string in hamiltonian:
        coeff = pauli_string.coefficient
        exp_value = 1.0
        for qubit in qubits:
            if qubit in pauli_string.qubits:
                gate = pauli_string[qubit]
                qubit_idx = qubit.x
                if qubit_idx in range(n_electrons):
                    if gate == cirq.Z:
                        exp_value *= -1
                    elif gate in [cirq.X, cirq.Y]:
                        exp_value = 0.0
                        break
                else:
                    if gate in [cirq.X, cirq.Y]:
                        exp_value = 0.0
                        break
        hf_energy += np.real(coeff * exp_value)
    return hf_energy


if __name__ == "__main__":
    print("=" * 80)
    print("Loading H2O Hamiltonian")
    print("=" * 80)

    print("\nLoading from monomer_eqb.hdf5...")
    hamiltonian_cirq, qubits = load_h2o_hamiltonian()

    print(f"Hamiltonian type: {type(hamiltonian_cirq)}")
    print(f"Number of terms: {len(list(hamiltonian_cirq))}")
    print(f"Number of qubits: {len(qubits)}")

    print("\n" + "=" * 80)
    print("Analyzing coefficient distribution")
    print("=" * 80)

    coefficients = []
    for pauli_string in hamiltonian_cirq:
        coefficients.append(abs(pauli_string.coefficient))

    coefficients = np.array(coefficients)
    print(f"\nMin coefficient: {coefficients.min():.6e}")
    print(f"Max coefficient: {coefficients.max():.6e}")
    print(f"Mean coefficient: {coefficients.mean():.6e}")
    print(f"Median coefficient: {np.median(coefficients):.6e}")

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p:2d}th: {np.percentile(coefficients, p):.6e}")

    print("\n" + "=" * 80)
    print("Analyzing Pauli string structure")
    print("=" * 80)

    string_lengths = []
    for pauli_string in hamiltonian_cirq:
        string_lengths.append(len(pauli_string))

    string_lengths = np.array(string_lengths)
    print(f"\nMin length (qubit support): {string_lengths.min()}")
    print(f"Max length (qubit support): {string_lengths.max()}")
    print(f"Mean length: {string_lengths.mean():.2f}")

    print(f"\nLength distribution:")
    unique, counts = np.unique(string_lengths, return_counts=True)
    for length, count in zip(unique, counts):
        print(f"  Length {length:2d}: {count:4d} terms ({count/len(string_lengths)*100:5.1f}%)")

    print("\n" + "=" * 80)
    print("Sample terms (top 10 by coefficient)")
    print("=" * 80)

    terms_sorted = sorted(hamiltonian_cirq, key=lambda ps: abs(ps.coefficient), reverse=True)
    for i, pauli_string in enumerate(terms_sorted[:10]):
        print(f"{i+1:2d}. {abs(pauli_string.coefficient):10.6f} * {pauli_string}")

    print("\n" + "=" * 80)
