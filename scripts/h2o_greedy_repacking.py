#!/usr/bin/env python3
"""Test greedy repacking on H2O Hamiltonian."""
import time
import numpy as np
from src.hamiltonians.h2o import load_h2o_hamiltonian
from src.grouping.cache import GroupingCache
from src.repacking.greedy import greedy_repacking
from src.measurement.setup import MeasurementSetup, create_measurement_setups
from src.pauli.diagonalization import is_pauli_diagonal, diagonalize_pauli_strings
from src.pauli.conjugation import conjugate_pauli_by_clifford
from src.pauli.operations import pauli_strings_commute
from src.grouping.measurement_groups import MeasurementGroups

print("=" * 80)
print("Testing Greedy Repacking on H2O")
print("=" * 80)

print("\n[1/4] Loading baseline from cache...")
start = time.time()

hamiltonian_cirq, _ = load_h2o_hamiltonian("data/hamiltonians/monomer_eqb.hdf5")
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

print("\n[2/4] Running greedy repacking...")
start = time.time()

repacked_groups = greedy_repacking(baseline_cache.measurement_groups, verbose=True)

repacking_time = time.time() - start
print(f"\nRepacking completed in {repacking_time:.2f}s")

print("\n" + "=" * 80)
print("Repacking Statistics")
print("=" * 80)

repacked_measurements = sum(repacked_groups.measurement_counts.values())
print(f"\nTotal measurements (repacked): {repacked_measurements}")
print(f"Average measurements per term: {repacked_measurements / repacked_groups.num_groups():.2f}")
print(f"Increase: {repacked_measurements - baseline_measurements} ({(repacked_measurements/baseline_measurements - 1)*100:.1f}%)")

group_sizes_repacked = [len(group) for group in repacked_groups.groups]
print(f"\nGroup sizes after repacking:")
print(f"  - Min: {min(group_sizes_repacked)}")
print(f"  - Max: {max(group_sizes_repacked)}")
print(f"  - Mean: {np.mean(group_sizes_repacked):.2f}")

print(f"\nTerms measured multiple times:")
multiple_measurements = sum(1 for count in repacked_groups.measurement_counts.values() if count > 1)
print(f"  - Count: {multiple_measurements}/{len(repacked_groups.measurement_counts)}")
print(f"  - Percentage: {multiple_measurements/len(repacked_groups.measurement_counts)*100:.1f}%")

max_measurements = max(repacked_groups.measurement_counts.values())
print(f"  - Max measurements for single term: {max_measurements}")

print("\n[3/4] Verifying repacked groups are mutually commuting...")

# Deduplicate groups
def deduplicate_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

deduplicated_groups = [deduplicate_preserve_order(group) for group in repacked_groups.groups]
repacked_groups = MeasurementGroups(deduplicated_groups, repacked_groups.coefficients)

for group_idx, group in enumerate(deduplicated_groups):
    for i, pauli1 in enumerate(group):
        for j, pauli2 in enumerate(group):
            if i >= j:
                continue
            if not pauli_strings_commute(pauli1, pauli2):
                print(f"  ✗ Group {group_idx}: {pauli1} and {pauli2} don't commute!")
                print(f"    This is a bug in the repacking algorithm!")
                exit(1)

print(f"  ✓ All {len(deduplicated_groups)} groups are mutually commuting")

print("\nGenerating circuits...")
start = time.time()

repacked_setups = []
exact_match_reused = 0
regenerated = 0

for i, group in enumerate(deduplicated_groups):
    baseline_group_set = set(baseline_cache.measurement_groups.groups[i])
    repacked_group_set = set(group)

    if baseline_group_set == repacked_group_set:
        repacked_setups.append(baseline_cache.measurement_setups[i])
        exact_match_reused += 1
    else:
        circuit, diagonalized = diagonalize_pauli_strings(group, baseline_cache.qubits)
        setup = MeasurementSetup(i, group, diagonalized, baseline_cache.qubits, circuit)
        repacked_setups.append(setup)
        regenerated += 1

circuit_time = time.time() - start
print(f"\nCircuit generation complete in {circuit_time:.2f}s:")
print(f"  {exact_match_reused} exact matches (reused baseline)")
print(f"  {regenerated} regenerated (non-diagonal changes)")

print("\n[4/4] Saving repacked cache...")

repacked_cache = GroupingCache(
    hamiltonian_cirq=hamiltonian_cirq,
    qubits=baseline_cache.qubits,
    measurement_groups=repacked_groups,
    measurement_setups=repacked_setups
)

cache_filename = "data/caches/h2o_repacked_cache.pkl"
repacked_cache.save(cache_filename)

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

repacked_variance = 0.0
for pauli, coeff in repacked_groups.coefficients.items():
    N_i = repacked_groups.measurement_counts.get(pauli, 0)
    if N_i > 0:
        repacked_variance += coeff**2 / (N_i * shots_per_group)

print(f"\nWith {total_budget:,} total shots, uniform allocation:")
print(f"  Baseline: {baseline_cache.num_groups()} groups × {shots_per_group:,} shots")
print(f"  Repacked: {repacked_groups.num_groups()} groups × {shots_per_group:,} shots")

print(f"\nTheoretical variance (assuming <P_i>=0):")
print(f"  Baseline: V_E = {baseline_variance:.6e}")
print(f"  Repacked: V_E = {repacked_variance:.6e}")
print(f"  Improvement: {baseline_variance/repacked_variance:.4f}x ({(1 - repacked_variance/baseline_variance)*100:.1f}% reduction)")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Repacking time:        {repacking_time:8.2f} s")
print(f"Circuit generation:    {circuit_time:8.2f} s")
print(f"Total:                 {repacking_time + circuit_time:8.2f} s")
print(f"\nBaseline groups:       {baseline_cache.num_groups():8d}")
print(f"Repacked groups:       {repacked_groups.num_groups():8d} (same)")
print(f"Measurement increase:  {repacked_measurements - baseline_measurements:8d} ({(repacked_measurements/baseline_measurements - 1)*100:.1f}%)")
print(f"Variance reduction:    {(1 - repacked_variance/baseline_variance)*100:8.1f}% (theoretical)")
print("=" * 80)
