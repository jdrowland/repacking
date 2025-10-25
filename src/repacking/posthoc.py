import cirq
from typing import List
from src.grouping.measurement_groups import MeasurementGroups
from src.pauli.operations import pauli_strings_commute
from src.pauli.conjugation import conjugate_pauli_by_clifford
from src.pauli.diagonalization import is_pauli_diagonal


def have_disjoint_support(pauli1: cirq.PauliString, pauli2: cirq.PauliString) -> bool:
    qubits1 = set(pauli1.qubits)
    qubits2 = set(pauli2.qubits)
    return qubits1.isdisjoint(qubits2)


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
