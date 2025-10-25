import cirq
from typing import List, Dict


class MeasurementGroups:
    def __init__(self, groups: List[List[cirq.PauliString]], coefficients: Dict[cirq.PauliString, float]):
        self.groups = groups
        self.coefficients = coefficients
        self.measurement_counts = self._compute_measurement_counts()
        self._pauli_to_groups = self._build_pauli_to_groups_map()

    def _compute_measurement_counts(self) -> Dict[cirq.PauliString, int]:
        counts = {}
        for group in self.groups:
            for pauli in group:
                counts[pauli] = counts.get(pauli, 0) + 1
        return counts

    def _build_pauli_to_groups_map(self) -> Dict[cirq.PauliString, List[int]]:
        pauli_to_groups = {}
        for group_idx, group in enumerate(self.groups):
            for pauli in group:
                if pauli not in pauli_to_groups:
                    pauli_to_groups[pauli] = []
                pauli_to_groups[pauli].append(group_idx)
        return pauli_to_groups

    def num_groups(self) -> int:
        return len(self.groups)

    def get_group_containing(self, pauli: cirq.PauliString) -> List[int]:
        return self._pauli_to_groups.get(pauli, [])
