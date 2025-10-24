import openfermion as of
import cirq
import numpy as np
import time
from cache import GroupingCache
from simulation import simulate_all_groups
from estimation import estimate_from_groups

print("=" * 80)
print("Testing Post-hoc Cache with Simulation")
print("=" * 80)

print("\n[1/4] Loading caches...")
hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(
        of.chem.MolecularData(filename="monomer_eqb.hdf5").get_molecular_hamiltonian()
    )
)
hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
baseline_cache = GroupingCache.load("h2o_baseline_cache.pkl", hamiltonian_cirq)
posthoc_cache = GroupingCache.load("h2o_posthoc_cache.pkl", hamiltonian_cirq)

print(f"  Baseline: {baseline_cache.num_groups()} groups, {baseline_cache.num_terms()} terms")
print(f"  Post-hoc: {posthoc_cache.num_groups()} groups, {posthoc_cache.num_terms()} terms")

# Compute exact energy for |0⟩ state
exact_energy = 0.0
for pauli_string in hamiltonian_cirq:
    coeff = pauli_string.coefficient
    # For |0⟩ state, only identity and Z terms contribute
    exp_value = 1.0
    for qubit, gate in pauli_string.items():
        if gate in [cirq.X, cirq.Y]:
            exp_value = 0.0
            break
    exact_energy += np.real(coeff * exp_value)

print(f"\nExact energy for |0⟩: {exact_energy:.6f}")

shots_per_group = 10000
simulator = cirq.Simulator()
state_prep = cirq.Circuit()  # |0⟩ state

print(f"\n[2/4] Simulating baseline (shots={shots_per_group})...")
start = time.time()
baseline_counts, baseline_shots = simulate_all_groups(
    state_prep, baseline_cache.measurement_setups, shots_per_group, simulator
)
baseline_time = time.time() - start

baseline_results = estimate_from_groups(
    baseline_cache.measurement_groups,
    baseline_cache.measurement_setups,
    baseline_counts,
    baseline_shots,
    baseline_cache.qubits
)

print(f"  Time: {baseline_time:.2f}s")
print(f"  Energy: {baseline_results.energy:.6f} ± {baseline_results.energy_std():.6f}")
baseline_error = abs(baseline_results.energy - exact_energy)
baseline_error_sigma = baseline_error / baseline_results.energy_std()
print(f"  Error: {baseline_error:.6f} ({baseline_error_sigma:.2f}σ)")

print(f"\n[3/4] Simulating post-hoc (shots={shots_per_group})...")
start = time.time()
posthoc_counts, posthoc_shots = simulate_all_groups(
    state_prep, posthoc_cache.measurement_setups, shots_per_group, simulator
)
posthoc_time = time.time() - start

posthoc_results = estimate_from_groups(
    posthoc_cache.measurement_groups,
    posthoc_cache.measurement_setups,
    posthoc_counts,
    posthoc_shots,
    posthoc_cache.qubits
)

print(f"  Time: {posthoc_time:.2f}s")
print(f"  Energy: {posthoc_results.energy:.6f} ± {posthoc_results.energy_std():.6f}")
posthoc_error = abs(posthoc_results.energy - exact_energy)
posthoc_error_sigma = posthoc_error / posthoc_results.energy_std()
print(f"  Error: {posthoc_error:.6f} ({posthoc_error_sigma:.2f}σ)")

print("\n[4/4] Comparison...")

print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)

print(f"\nWith {shots_per_group:,} shots per group:")
total_baseline_shots = sum(baseline_shots)
total_posthoc_shots = sum(posthoc_shots)
print(f"  Baseline total shots: {total_baseline_shots:,}")
print(f"  Post-hoc total shots: {total_posthoc_shots:,}")

print(f"\nEnergy estimates:")
print(f"  Exact:        {exact_energy:12.6f}")
print(f"  Baseline:     {baseline_results.energy:12.6f} ± {baseline_results.energy_std():.6f}")
print(f"  Post-hoc:     {posthoc_results.energy:12.6f} ± {posthoc_results.energy_std():.6f}")

variance_reduction = baseline_results.energy_variance / posthoc_results.energy_variance
std_reduction = baseline_results.energy_std() / posthoc_results.energy_std()

print(f"\nVariance:")
print(f"  Baseline:  {baseline_results.energy_variance:.6e}")
print(f"  Post-hoc:  {posthoc_results.energy_variance:.6e}")
print(f"  Reduction: {variance_reduction:.2f}x ({(1-1/variance_reduction)*100:.1f}% reduction)")

print(f"\nStandard deviation:")
print(f"  Baseline:  {baseline_results.energy_std():.6f}")
print(f"  Post-hoc:  {posthoc_results.energy_std():.6f}")
print(f"  Reduction: {std_reduction:.2f}x")

print(f"\nSimulation time:")
print(f"  Baseline:  {baseline_time:.2f}s")
print(f"  Post-hoc:  {posthoc_time:.2f}s")
print(f"  Slowdown:  {posthoc_time/baseline_time:.2f}x")

print(f"\nStatistical significance:")
print(f"  Baseline error: {baseline_error:.6f} ({baseline_error_sigma:.2f}σ)")
print(f"  Post-hoc error: {posthoc_error:.6f} ({posthoc_error_sigma:.2f}σ)")

if baseline_error_sigma < 3 and posthoc_error_sigma < 3:
    print("\n  ✓ Both estimates within 3σ")
else:
    print("\n  ✗ Some estimates outside 3σ (may need more shots)")

if variance_reduction > 1.0:
    print(f"  ✓ Post-hoc repacking improves variance by {variance_reduction:.2f}x")
else:
    print(f"  ✗ Post-hoc repacking does not improve variance")

print("=" * 80)
