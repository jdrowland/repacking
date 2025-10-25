import cirq
import heapq
from typing import List
from src.grouping.measurement_groups import MeasurementGroups
from src.pauli.operations import pauli_strings_commute


def have_disjoint_support(pauli1: cirq.PauliString, pauli2: cirq.PauliString) -> bool:
    qubits1 = set(pauli1.qubits)
    qubits2 = set(pauli2.qubits)
    return qubits1.isdisjoint(qubits2)


def greedy_repacking(
    baseline_groups: MeasurementGroups,
    verbose: bool = False
) -> MeasurementGroups:

    groups = [list(group) for group in baseline_groups.groups]
    coefficients = baseline_groups.coefficients

    pauli_to_groups = {}
    measurement_counts = {}

    for group_idx, group in enumerate(groups):
        for pauli in group:
            if pauli not in pauli_to_groups:
                pauli_to_groups[pauli] = []
            pauli_to_groups[pauli].append(group_idx)
            measurement_counts[pauli] = measurement_counts.get(pauli, 0) + 1

    heap = []
    for pauli, coeff in coefficients.items():
        if len(pauli) == 0:
            continue

        c_squared = coeff ** 2
        N_i = measurement_counts[pauli]
        variance_contribution = c_squared / N_i

        heapq.heappush(heap, (-variance_contribution, id(pauli), pauli))

    if verbose:
        print(f"Initial heap size: {len(heap)}")
        print(f"Initial groups: {len(groups)}")

    iterations = 0
    successful_insertions = 0

    while heap:
        neg_var, _, pauli = heapq.heappop(heap)
        variance_contribution = -neg_var

        current_groups = pauli_to_groups.get(pauli, [])
        if not current_groups:
            continue

        last_group_idx = max(current_groups)

        inserted = False
        for group_idx in range(last_group_idx + 1, len(groups)):
            group = groups[group_idx]

            can_insert = True
            for existing_pauli in group:
                if have_disjoint_support(pauli, existing_pauli):
                    continue

                if not pauli_strings_commute(pauli, existing_pauli):
                    can_insert = False
                    break

            if can_insert:
                group.append(pauli)
                pauli_to_groups[pauli].append(group_idx)
                measurement_counts[pauli] += 1

                c_squared = coefficients[pauli] ** 2
                N_i = measurement_counts[pauli]
                new_variance_contribution = c_squared / N_i

                heapq.heappush(heap, (-new_variance_contribution, id(pauli), pauli))

                inserted = True
                successful_insertions += 1
                break

        iterations += 1

        if verbose and iterations % 100 == 0:
            print(f"  Iteration {iterations}: heap size {len(heap)}, successful insertions {successful_insertions}")

    if verbose:
        print(f"\nRepacking complete:")
        print(f"  Total iterations: {iterations}")
        print(f"  Successful insertions: {successful_insertions}")
        print(f"  Final groups: {len(groups)}")

        total_measurements = sum(measurement_counts.values())
        baseline_measurements = sum(baseline_groups.measurement_counts.values())
        print(f"  Baseline measurements: {baseline_measurements}")
        print(f"  Repacked measurements: {total_measurements}")
        print(f"  Increase: {total_measurements - baseline_measurements} ({(total_measurements/baseline_measurements - 1)*100:.1f}%)")

    return MeasurementGroups(groups, coefficients)
