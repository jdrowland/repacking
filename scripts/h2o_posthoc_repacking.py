import openfermion as of
import cirq
import numpy as np
import time
from src.grouping.cache import GroupingCache
from src.repacking.posthoc import posthoc_repacking
from src.measurement.setup import create_measurement_setups
from src.pauli.operations import pauli_strings_commute
from src.grouping.measurement_groups import MeasurementGroups

print("=" * 80)
print("Testing Post-hoc Repacking on H2O")
print("=" * 80)

print("\n[1/4] Loading baseline from cache...")
start = time.time()

hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(
        of.chem.MolecularData(filename="data/hamiltonians/monomer_eqb.hdf5").get_molecular_hamiltonian()
    )
)
hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
baseline_cache = GroupingCache.load("data/caches/h2o_baseline_cache.pkl", hamiltonian_cirq)

print(f"Loaded in {time.time() - start:.2f}s")
print(f"  - {baseline_cache.num_groups()} groups")
print(f"  - {baseline_cache.num_terms()} terms")

print("\n" + "=" * 80)
print("Baseline Statistics")
print("=" * 80)

baseline_measurements = sum(baseline_cache.measurement_groups.measurement_counts.values())
print(f"\nTotal measurements (baseline): {baseline_measurements}")
print(f"Average measurements per term: {baseline_measurements / baseline_cache.num_terms():.2f}")

group_sizes = [len(group) for group in baseline_cache.measurement_groups.groups]
print(f"\nGroup sizes:")
print(f"  - Min: {min(group_sizes)}")
print(f"  - Max: {max(group_sizes)}")
print(f"  - Mean: {np.mean(group_sizes):.2f}")

print("\n[2/4] Running post-hoc repacking...")
start = time.time()

posthoc_groups = posthoc_repacking(
    baseline_cache.measurement_groups,
    baseline_cache.measurement_setups,
    baseline_cache.qubits,
    verbose=True
)

repacking_time = time.time() - start
print(f"\nPost-hoc repacking completed in {repacking_time:.2f}s")

print("\n" + "=" * 80)
print("Post-hoc Repacking Statistics")
print("=" * 80)

posthoc_measurements = sum(posthoc_groups.measurement_counts.values())
print(f"\nTotal measurements (post-hoc): {posthoc_measurements}")
print(f"Average measurements per term: {posthoc_measurements / baseline_cache.num_terms():.2f}")
print(f"Increase: {posthoc_measurements - baseline_measurements} ({(posthoc_measurements/baseline_measurements - 1)*100:.1f}%)")

group_sizes_posthoc = [len(group) for group in posthoc_groups.groups]
print(f"\nGroup sizes after post-hoc repacking:")
print(f"  - Min: {min(group_sizes_posthoc)}")
print(f"  - Max: {max(group_sizes_posthoc)}")
print(f"  - Mean: {np.mean(group_sizes_posthoc):.2f}")

print(f"\nTerms measured multiple times:")
multiple_measurements = sum(1 for count in posthoc_groups.measurement_counts.values() if count > 1)
print(f"  - Count: {multiple_measurements}/{len(posthoc_groups.measurement_counts)}")
print(f"  - Percentage: {multiple_measurements/len(posthoc_groups.measurement_counts)*100:.1f}%")

max_measurements = max(posthoc_groups.measurement_counts.values())
print(f"  - Max measurements for single term: {max_measurements}")

print("\n[3/4] Verifying repacked groups are mutually commuting...")

# Deduplicate groups (should not be necessary but let's be safe)
def deduplicate_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

deduplicated_groups = [deduplicate_preserve_order(group) for group in posthoc_groups.groups]
posthoc_groups = MeasurementGroups(deduplicated_groups, posthoc_groups.coefficients)

for group_idx, group in enumerate(deduplicated_groups):
    for i, pauli1 in enumerate(group):
        for j, pauli2 in enumerate(group):
            if i >= j:
                continue
            if not pauli_strings_commute(pauli1, pauli2):
                print(f"  ✗ Group {group_idx}: {pauli1} and {pauli2} don't commute!")
                print(f"    This is a bug in the post-hoc repacking algorithm!")
                exit(1)

print(f"  ✓ All {len(deduplicated_groups)} groups are mutually commuting")

print("\nGenerating circuits (reusing baseline where possible)...")
start = time.time()

from src.pauli.diagonalization import is_pauli_diagonal, diagonalize_pauli_strings
from src.measurement.setup import MeasurementSetup
from src.pauli.conjugation import conjugate_pauli_by_clifford

def verify_diagonalization(group_idx, original_paulis, diagonalized_paulis, circuit):
    """Verify that all diagonalized Paulis are actually diagonal (Z/I only)."""
    for i, (orig, diag) in enumerate(zip(original_paulis, diagonalized_paulis)):
        if not is_pauli_diagonal(diag):
            error_msg = f"""
ERROR: Non-diagonal result in group {group_idx}, term {i}
  Original Pauli:      {orig}
  Diagonalized result: {diag}
  Circuit depth:       {len(circuit)}
"""
            raise ValueError(error_msg)

posthoc_setups = []
exact_match_reused = 0
conjugated_added = 0
regenerated = 0

for i, group in enumerate(deduplicated_groups):
    baseline_group = baseline_cache.measurement_groups.groups[i]
    baseline_group_set = set(baseline_group)
    posthoc_group_set = set(group)

    # Case 1: Exact match - just reuse
    if baseline_group_set == posthoc_group_set:
        setup = baseline_cache.measurement_setups[i]
        verify_diagonalization(i, setup.paulis, setup.diagonalized_paulis, setup.basis_rotation)
        posthoc_setups.append(setup)
        exact_match_reused += 1
    else:
        # Case 2: Paulis were added - conjugate them through baseline circuit
        try:
            baseline_setup = baseline_cache.measurement_setups[i]
            circuit = baseline_setup.basis_rotation

            # Build diagonalized list by reusing baseline results and conjugating added Paulis
            baseline_diag_dict = {p: d for p, d in zip(baseline_setup.paulis, baseline_setup.diagonalized_paulis)}
            full_diagonalized = []

            for p in group:
                if p in baseline_diag_dict:
                    # Reuse baseline result
                    full_diagonalized.append(baseline_diag_dict[p])
                else:
                    # It's an added Pauli - conjugate through the circuit
                    transformed = conjugate_pauli_by_clifford(p, circuit, baseline_cache.qubits)
                    full_diagonalized.append(transformed)

            setup = MeasurementSetup(i, group, full_diagonalized, baseline_cache.qubits, circuit)
            verify_diagonalization(i, group, full_diagonalized, circuit)
            posthoc_setups.append(setup)
            conjugated_added += 1
        except Exception as e:
            # Conjugation failed - regenerate circuit
            print(f"  ⚠ Group {i}: Conjugation failed, regenerating circuit: {str(e)[:80]}")
            circuit, diagonalized = diagonalize_pauli_strings(group, baseline_cache.qubits)
            setup = MeasurementSetup(i, group, diagonalized, baseline_cache.qubits, circuit)
            verify_diagonalization(i, group, diagonalized, circuit)
            posthoc_setups.append(setup)
            regenerated += 1

    if (i + 1) % 10 == 0:
        print(f"  Progress: {exact_match_reused} reused, {conjugated_added} conjugated, {regenerated} regenerated ({i+1}/{len(deduplicated_groups)} groups)")

circuit_time = time.time() - start
print(f"\nCircuit generation complete in {circuit_time:.2f}s:")
print(f"  {exact_match_reused} exact matches (reused baseline)")
print(f"  {conjugated_added} with additions (conjugated added paulis)")
print(f"  {regenerated} regenerated (conjugation failed)")

print("\n[4/4] Saving post-hoc cache...")

posthoc_cache = GroupingCache(
    hamiltonian_cirq=hamiltonian_cirq,
    qubits=baseline_cache.qubits,
    measurement_groups=posthoc_groups,
    measurement_setups=posthoc_setups
)

cache_filename = "data/caches/h2o_posthoc_cache.pkl"
posthoc_cache.save(cache_filename)

print(f"Saved to: {cache_filename}")

print("\n" + "=" * 80)
print("Variance Comparison (Theoretical)")
print("=" * 80)

total_budget = 100000
shots_per_group = total_budget // baseline_cache.num_groups()

baseline_variance = 0.0
for pauli, coeff in baseline_cache.measurement_groups.coefficients.items():
    N_i = baseline_cache.measurement_groups.measurement_counts.get(pauli, 0)
    if N_i > 0:
        baseline_variance += coeff**2 / (N_i * shots_per_group)

posthoc_variance = 0.0
for pauli, coeff in posthoc_groups.coefficients.items():
    N_i = posthoc_groups.measurement_counts.get(pauli, 0)
    if N_i > 0:
        posthoc_variance += coeff**2 / (N_i * shots_per_group)

print(f"\nWith {total_budget:,} total shots, uniform allocation:")
print(f"  Baseline:  {baseline_cache.num_groups()} groups × {shots_per_group:,} shots")
print(f"  Post-hoc:  {posthoc_groups.num_groups()} groups × {shots_per_group:,} shots")

print(f"\nTheoretical variance (assuming <P_i>=0):")
print(f"  Baseline:  V_E = {baseline_variance:.6e}")
print(f"  Post-hoc:  V_E = {posthoc_variance:.6e}")
print(f"  Improvement: {baseline_variance/posthoc_variance:.4f}x ({(1 - posthoc_variance/baseline_variance)*100:.1f}% reduction)")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Post-hoc time:         {repacking_time:8.2f} s")
print(f"Circuit generation:    {circuit_time:8.2f} s")
print(f"Total:                 {repacking_time + circuit_time:8.2f} s")
print(f"\nBaseline groups:       {baseline_cache.num_groups():8d}")
print(f"Post-hoc groups:       {posthoc_groups.num_groups():8d} (same)")
print(f"Measurement increase:  {posthoc_measurements - baseline_measurements:8d} ({(posthoc_measurements/baseline_measurements - 1)*100:.1f}%)")
print(f"Variance reduction:    {(1 - posthoc_variance/baseline_variance)*100:8.1f}% (theoretical)")
print("=" * 80)
