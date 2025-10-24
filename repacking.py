import cirq
import heapq
import numpy as np
from typing import List, Dict, Set
from grouping import MeasurementGroups
from utils import pauli_strings_commute
from conjugate_pauli import conjugate_pauli_by_clifford
from diagonalize_paulis import is_pauli_diagonal


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


def posthoc_repacking(
    baseline_groups: MeasurementGroups,
    measurement_setups: List,
    qubits: List[cirq.Qid],
    verbose: bool = False
) -> MeasurementGroups:
    """Post-hoc repacking: add paulis from previous groups if they're diagonal.

    For each group, checks paulis from ALL previous groups (not just the one
    directly before) to see if they commute with the group AND are diagonal
    under the group's diagonalizing unitary. If so, adds them to the group.

    Args:
        baseline_groups: The baseline measurement groups (already measured)
        measurement_setups: The measurement setups with circuits for each group
        qubits: List of qubits
        verbose: Whether to print progress information

    Returns:
        New MeasurementGroups with additional paulis added to groups
    """
    groups = [list(group) for group in baseline_groups.groups]
    coefficients = baseline_groups.coefficients

    total_added = 0

    for group_idx in range(len(groups)):
        current_group = groups[group_idx]
        current_group_set = set(current_group)
        circuit = measurement_setups[group_idx].basis_rotation

        added_to_group = 0

        # Check paulis from all previous groups
        for prev_group_idx in range(group_idx):
            prev_group = groups[prev_group_idx]

            for pauli in prev_group:
                # Skip if already in current group (handle duplicates)
                if pauli in current_group_set:
                    continue

                # First check commutativity (cheap)
                can_insert = True
                for existing_pauli in current_group:
                    if have_disjoint_support(pauli, existing_pauli):
                        continue
                    if not pauli_strings_commute(pauli, existing_pauli):
                        can_insert = False
                        break

                if not can_insert:
                    continue

                # Check if diagonal under the circuit (expensive)
                try:
                    conjugated = conjugate_pauli_by_clifford(pauli, circuit, qubits)
                    if is_pauli_diagonal(conjugated):
                        current_group.append(pauli)
                        current_group_set.add(pauli)
                        added_to_group += 1
                        total_added += 1
                except Exception as e:
                    # If conjugation fails, skip this pauli
                    if verbose:
                        print(f"  ⚠ Group {group_idx}: Failed to conjugate {pauli}: {str(e)[:60]}")
                    continue

        if verbose and added_to_group > 0:
            print(f"  Group {group_idx}: added {added_to_group} paulis from previous groups")

    if verbose:
        print(f"\nPost-hoc repacking complete:")
        print(f"  Total paulis added: {total_added}")

    return MeasurementGroups(groups, coefficients)
