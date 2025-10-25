import cirq
import numpy as np
from typing import List


def conjugate_pauli_by_clifford(pauli: cirq.PauliString, circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> cirq.PauliString:
    original_coefficient = pauli.coefficient
    num_qubits = len(qubits)

    tableau = cirq.CliffordTableau(num_qubits)
    state = cirq.CliffordTableauSimulationState(
        tableau=tableau,
        qubits=qubits,
        prng=np.random.RandomState()
    )

    for moment in circuit:
        for op in moment:
            cirq.act_on(op, state)

    result = cirq.PauliString({}, coefficient=1.0)

    for qubit, pauli_gate in pauli.items():
        qubit_idx = qubits.index(qubit)

        if pauli_gate == cirq.X:
            transformed = _tableau_row_to_pauli(state.tableau, qubit_idx, False, qubits)
        elif pauli_gate == cirq.Y:
            x_transform = _tableau_row_to_pauli(state.tableau, qubit_idx, False, qubits)
            z_transform = _tableau_row_to_pauli(state.tableau, qubit_idx, True, qubits)
            transformed = x_transform * z_transform
        elif pauli_gate == cirq.Z:
            transformed = _tableau_row_to_pauli(state.tableau, qubit_idx, True, qubits)
        else:
            continue

        result = result * transformed

    return result.with_coefficient(result.coefficient * original_coefficient)


def _tableau_row_to_pauli(tableau: cirq.CliffordTableau, qubit_idx: int, is_z: bool, qubits: List[cirq.Qid]) -> cirq.PauliString:
    row_idx = (qubit_idx + len(qubits)) if is_z else qubit_idx

    x_row = tableau.xs[row_idx]
    z_row = tableau.zs[row_idx]
    sign = tableau.rs[row_idx]

    pauli_dict = {}
    for i, qubit in enumerate(qubits):
        if x_row[i] and z_row[i]:
            pauli_dict[qubit] = cirq.Y
        elif x_row[i]:
            pauli_dict[qubit] = cirq.X
        elif z_row[i]:
            pauli_dict[qubit] = cirq.Z

    coefficient = -1 if sign else 1
    return cirq.PauliString(pauli_dict, coefficient=coefficient)
