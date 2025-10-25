import cirq
import numpy as np
from typing import Dict


def pauli_strings_commute(pauli1: cirq.PauliString, pauli2: cirq.PauliString) -> bool:
    common_qubits = set(pauli1.qubits) & set(pauli2.qubits)

    anticommute_count = 0
    for qubit in common_qubits:
        if pauli1[qubit] != pauli2[qubit]:
            anticommute_count += 1

    return anticommute_count % 2 == 0


def extract_coefficients(pauli_sum: cirq.PauliSum) -> Dict[cirq.PauliString, float]:
    coefficients = {}
    for pauli_string in pauli_sum:
        coeff = float(np.real(pauli_string.coefficient))
        pauli_without_coeff = pauli_string.with_coefficient(1.0)
        coefficients[pauli_without_coeff] = coeff
    return coefficients
