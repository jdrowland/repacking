import cirq
from typing import List, Dict
from utils import pauli_strings_commute, extract_coefficients


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


def sorted_insertion_grouping(pauli_sum: cirq.PauliSum) -> MeasurementGroups:
    coefficients = extract_coefficients(pauli_sum)

    sorted_terms = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    non_identity_terms = [(pauli, coeff) for pauli, coeff in sorted_terms if len(pauli) > 0]

    groups = []
    for pauli, coeff in non_identity_terms:
        placed = False
        for group in groups:
            if all(pauli_strings_commute(pauli, existing) for existing in group):
                group.append(pauli)
                placed = True
                break

        if not placed:
            groups.append([pauli])

    return MeasurementGroups(groups, coefficients)
