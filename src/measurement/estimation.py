import cirq
import numpy as np
from typing import Dict, List
from src.grouping.measurement_groups import MeasurementGroups


def compute_pauli_expectation(pauli_diagonalized: cirq.PauliString, counts: Dict[str, int], qubits: List[cirq.Qid]) -> float:
    if len(pauli_diagonalized) == 0:
        return 1.0

    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    qubit_to_index = {q: i for i, q in enumerate(qubits)}
    coefficient = float(np.real(pauli_diagonalized.coefficient))

    expectation = 0.0
    for bitstring, count in counts.items():
        parity = sum(int(bitstring[qubit_to_index[q]]) for q, g in pauli_diagonalized.items() if g == cirq.Z)
        eigenvalue = (-1) ** (parity % 2)
        expectation += eigenvalue * count

    return coefficient * expectation / total_shots


def compute_pauli_variance(expectation: float, num_measurements: int) -> float:
    if num_measurements == 0:
        return float('inf')
    return (1.0 - expectation ** 2) / num_measurements


class EstimationResults:
    def __init__(self, expectations: Dict[cirq.PauliString, float], variances: Dict[cirq.PauliString, float],
                 measurement_counts: Dict[cirq.PauliString, int], coefficients: Dict[cirq.PauliString, float]):
        self.expectations = expectations
        self.variances = variances
        self.measurement_counts = measurement_counts
        self.coefficients = coefficients
        self.energy = self._compute_energy()
        self.energy_variance = self._compute_energy_variance()

    def _compute_energy(self) -> float:
        return sum(coeff * self.expectations.get(pauli, 0.0) for pauli, coeff in self.coefficients.items())

    def _compute_energy_variance(self) -> float:
        return sum(coeff ** 2 * self.variances.get(pauli, 0.0) for pauli, coeff in self.coefficients.items())

    def energy_std(self) -> float:
        return np.sqrt(self.energy_variance)


def estimate_from_groups(measurement_groups: MeasurementGroups, measurement_setups: List,
                        group_counts: List[Dict[str, int]], shots_per_group: List[int],
                        qubits: List[cirq.Qid]) -> EstimationResults:
    expectations = {}
    variances = {}
    measurement_counts = {}

    for pauli in measurement_groups.coefficients.keys():
        group_indices = measurement_groups.get_group_containing(pauli)

        if len(group_indices) == 0:
            expectations[pauli] = 1.0 if len(pauli) == 0 else 0.0
            variances[pauli] = 0.0
            measurement_counts[pauli] = 0
            continue

        total_shots = 0
        weighted_expectation = 0.0

        for group_idx in group_indices:
            setup = measurement_setups[group_idx]
            pauli_idx = setup.paulis.index(pauli)
            pauli_diagonalized = setup.diagonalized_paulis[pauli_idx]

            exp_value = compute_pauli_expectation(pauli_diagonalized, group_counts[group_idx], qubits)
            weighted_expectation += exp_value * shots_per_group[group_idx]
            total_shots += shots_per_group[group_idx]

        expectations[pauli] = weighted_expectation / total_shots if total_shots > 0 else 0.0
        measurement_counts[pauli] = total_shots
        variances[pauli] = compute_pauli_variance(expectations[pauli], total_shots)

    return EstimationResults(expectations, variances, measurement_counts, measurement_groups.coefficients)
