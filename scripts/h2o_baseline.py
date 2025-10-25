#!/usr/bin/env python3
"""Generate baseline H2O measurement groups and cache."""
import time
from src.hamiltonians.h2o import load_h2o_hamiltonian
from src.grouping.sorted_insertion import sorted_insertion_grouping
from src.measurement.setup import create_measurement_setups
from src.grouping.cache import GroupingCache

print("=" * 80)
print("Generating H2O Baseline Cache")
print("=" * 80)

print("\n[1/3] Loading H2O Hamiltonian...")
start_time = time.time()

hamiltonian_cirq, qubits = load_h2o_hamiltonian("data/hamiltonians/monomer_eqb.hdf5")

load_time = time.time() - start_time
print(f"Loaded in {load_time:.2f} seconds")
print(f"  - {len(qubits)} qubits")
print(f"  - {len(list(hamiltonian_cirq))} terms")

print("\n[2/3] Running sorted insertion grouping...")
start_time = time.time()

measurement_groups = sorted_insertion_grouping(hamiltonian_cirq)

grouping_time = time.time() - start_time
print(f"Completed in {grouping_time:.2f} seconds")
print(f"  - Number of groups: {measurement_groups.num_groups()}")

print("\n[3/3] Creating measurement setups (circuits)...")
start_time = time.time()

measurement_setups = create_measurement_setups(measurement_groups.groups, qubits)

circuit_time = time.time() - start_time
print(f"Completed in {circuit_time:.2f} seconds")
print(f"  - {len(measurement_setups)} circuits created")

print("\n" + "=" * 80)
print("Creating and saving cache...")
print("=" * 80)

cache = GroupingCache(
    hamiltonian_cirq=hamiltonian_cirq,
    qubits=qubits,
    measurement_groups=measurement_groups,
    measurement_setups=measurement_setups
)

cache_filename = "data/caches/h2o_baseline_cache.pkl"
cache.save(cache_filename)

print(f"\nCache saved to: {cache_filename}")
print(f"  - {cache.num_groups()} groups")
print(f"  - {cache.num_terms()} Pauli terms")

print("\n" + "=" * 80)
print("Verifying cache by loading...")
print("=" * 80)

loaded_cache = GroupingCache.load(cache_filename, hamiltonian_cirq)

print(f"✓ Loaded cache successfully")
print(f"  - {loaded_cache.num_groups()} groups")
print(f"  - {loaded_cache.num_terms()} terms")

print("\nTesting access to group+circuit pairs...")
for i in [0, 10, loaded_cache.num_groups()-1]:
    group, circuit = loaded_cache.get_group_and_circuit(i)
    print(f"  Group {i}: {len(group)} terms, circuit depth {len(circuit)}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Total generation time: {load_time + grouping_time + circuit_time:.2f} s")
print(f"  - Load:     {load_time:6.2f} s")
print(f"  - Grouping: {grouping_time:6.2f} s")
print(f"  - Circuits: {circuit_time:6.2f} s")
print(f"\nCache saved to: {cache_filename}")
print("✓ Ready for baseline measurements and repacking experiments!")
print("=" * 80)
