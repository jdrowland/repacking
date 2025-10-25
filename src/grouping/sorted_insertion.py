import cirq
from src.pauli.operations import pauli_strings_commute, extract_coefficients
from src.grouping.measurement_groups import MeasurementGroups


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
