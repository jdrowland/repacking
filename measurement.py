import cirq
from typing import List
from diagonalize_paulis import diagonalize_pauli_strings


class MeasurementSetup:
    def __init__(
        self,
        group_index: int,
        paulis: List[cirq.PauliString],
        diagonalized_paulis: List[cirq.PauliString],
        qubits: List[cirq.Qid],
        basis_rotation: cirq.Circuit
    ):
        self.group_index = group_index
        self.paulis = paulis
        self.diagonalized_paulis = diagonalized_paulis
        self.qubits = qubits
        self.basis_rotation = basis_rotation


def create_measurement_setups(
    groups: List[List[cirq.PauliString]],
    qubits: List[cirq.Qid]
) -> List[MeasurementSetup]:
    setups = []
    for i, group in enumerate(groups):
        circuit, diagonalized_paulis = diagonalize_pauli_strings(group, qubits)
        setup = MeasurementSetup(i, group, diagonalized_paulis, qubits, circuit)
        setups.append(setup)
    return setups
