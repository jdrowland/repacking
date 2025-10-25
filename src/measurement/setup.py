import cirq
from typing import List
from src.pauli.diagonalization import diagonalize_pauli_strings
from src.pauli.conjugation import conjugate_pauli_by_clifford


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
    """Create measurement setups by generating new diagonalization circuits.

    Args:
        groups: List of Pauli groups to measure
        qubits: Qubits to use

    Returns:
        List of measurement setups with newly generated circuits
    """
    setups = []
    for i, group in enumerate(groups):
        circuit, diagonalized_paulis = diagonalize_pauli_strings(group, qubits)
        setup = MeasurementSetup(i, group, diagonalized_paulis, qubits, circuit)
        setups.append(setup)
    return setups


def create_posthoc_setups(
    baseline_groups,
    baseline_setups: List[MeasurementSetup],
    posthoc_groups,
    qubits: List[cirq.Qid]
) -> List[MeasurementSetup]:
    """Create measurement setups for post-hoc analysis reusing baseline circuits.

    This is the CORRECT way to analyze post-hoc groups when measurements were
    collected using baseline circuits. For each group:
    - Reuses the baseline circuit (no regeneration!)
    - For Paulis that were in baseline: reuses their diagonalized forms
    - For Paulis added by post-hoc: computes their diagonalization under the baseline circuit

    This ensures we're extracting additional information from the SAME measurements,
    not analyzing measurements from different circuits.

    Use this for:
    - Analyzing hardware data (collected with baseline circuits)
    - Fair simulation comparisons (baseline and post-hoc use same circuits)

    Args:
        baseline_groups: The baseline measurement groups
        baseline_setups: The baseline measurement setups (with circuits)
        posthoc_groups: The post-hoc measurement groups (with added Paulis)
        qubits: Qubits to use

    Returns:
        List of measurement setups using baseline circuits with post-hoc Paulis
    """
    posthoc_setups = []

    for i in range(len(baseline_setups)):
        baseline_setup = baseline_setups[i]
        baseline_paulis_set = set(baseline_groups.groups[i])
        posthoc_paulis = posthoc_groups.groups[i]

        # CRITICAL: Reuse the baseline circuit
        circuit = baseline_setup.basis_rotation

        # For all Paulis in post-hoc group, compute diagonalized form
        diagonalized_paulis = []
        for pauli in posthoc_paulis:
            if pauli in baseline_paulis_set:
                # This Pauli was in baseline - reuse its diagonalized form
                idx = baseline_setup.paulis.index(pauli)
                diagonalized_paulis.append(baseline_setup.diagonalized_paulis[idx])
            else:
                # This Pauli was added by post-hoc - compute its diagonalized form
                # under the baseline circuit
                diag_pauli = conjugate_pauli_by_clifford(pauli, circuit, qubits)
                diagonalized_paulis.append(diag_pauli)

        setup = MeasurementSetup(
            group_index=i,
            paulis=posthoc_paulis,
            diagonalized_paulis=diagonalized_paulis,
            qubits=qubits,
            basis_rotation=circuit  # Use baseline circuit!
        )
        posthoc_setups.append(setup)

    return posthoc_setups
