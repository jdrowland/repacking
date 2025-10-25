import pickle
import openfermion as of
import cirq
import numpy as np
from src.grouping.cache import GroupingCache
from src.measurement.estimation import estimate_from_groups

print("=" * 80)
print("Analyzing IBM Hardware Data with Post-hoc Repacking")
print("=" * 80)

print("\n[1/4] Loading hardware counts and caches...")
with open('data/hardware_results/all_counts_fez_oct22_3.pkl', 'rb') as f:
    hardware_counts = pickle.load(f)

hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(
        of.chem.MolecularData(filename="data/hamiltonians/monomer_eqb.hdf5").get_molecular_hamiltonian()
    )
)
hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
baseline_cache = GroupingCache.load("data/caches/h2o_baseline_cache.pkl", hamiltonian_cirq)
posthoc_cache = GroupingCache.load("data/caches/h2o_posthoc_cache.pkl", hamiltonian_cirq)

shots_per_group = [sum(counts.values()) for counts in hardware_counts]
total_shots = sum(shots_per_group)

print(f"  Loaded {len(hardware_counts)} groups from hardware")
print(f"  Total shots: {total_shots:,} ({shots_per_group[0]:,} per group)")
print(f"  Baseline cache: {baseline_cache.num_groups()} groups")
print(f"  Post-hoc cache: {posthoc_cache.num_groups()} groups")

# Compute exact HF energy
mol_data = of.chem.MolecularData(filename="data/hamiltonians/monomer_eqb.hdf5")
n_electrons = mol_data.n_electrons

hf_energy_exact = 0.0
for pauli_string in hamiltonian_cirq:
    coeff = pauli_string.coefficient
    exp_value = 1.0
    for qubit in baseline_cache.qubits:
        if qubit in pauli_string.qubits:
            gate = pauli_string[qubit]
            qubit_idx = qubit.x
            if qubit_idx in range(n_electrons):
                if gate == cirq.Z:
                    exp_value *= -1
                elif gate in [cirq.X, cirq.Y]:
                    exp_value = 0.0
                    break
            else:
                if gate in [cirq.X, cirq.Y]:
                    exp_value = 0.0
                    break
    hf_energy_exact += np.real(coeff * exp_value)

print(f"\nExact HF energy: {hf_energy_exact:.6f} Ha")

print("\n[2/4] Analyzing baseline with hardware data...")
baseline_results = estimate_from_groups(
    baseline_cache.measurement_groups,
    baseline_cache.measurement_setups,
    hardware_counts,
    shots_per_group,
    baseline_cache.qubits
)

print(f"  Energy: {baseline_results.energy:.6f} ± {baseline_results.energy_std():.6f} Ha")
baseline_error = abs(baseline_results.energy - hf_energy_exact)
baseline_error_sigma = baseline_error / baseline_results.energy_std()
print(f"  Error: {baseline_error:.6f} Ha ({baseline_error_sigma:.2f}σ)")

print("\n[3/4] Analyzing post-hoc with hardware data...")
posthoc_results = estimate_from_groups(
    posthoc_cache.measurement_groups,
    posthoc_cache.measurement_setups,
    hardware_counts,
    shots_per_group,
    posthoc_cache.qubits
)

print(f"  Energy: {posthoc_results.energy:.6f} ± {posthoc_results.energy_std():.6f} Ha")
posthoc_error = abs(posthoc_results.energy - hf_energy_exact)
posthoc_error_sigma = posthoc_error / posthoc_results.energy_std()
print(f"  Error: {posthoc_error:.6f} Ha ({posthoc_error_sigma:.2f}σ)")

print("\n[4/4] Comparison...")

variance_reduction = baseline_results.energy_variance / posthoc_results.energy_variance
std_reduction = baseline_results.energy_std() / posthoc_results.energy_std()

print("\n" + "=" * 80)
print("Hardware Results Summary")
print("=" * 80)

print(f"\nTotal shots: {total_shots:,} ({shots_per_group[0]:,} per group)")

print(f"\nEnergy estimates:")
print(f"  Exact:        {hf_energy_exact:12.6f} Ha")
print(f"  Baseline:     {baseline_results.energy:12.6f} ± {baseline_results.energy_std():.6f} Ha")
print(f"  Post-hoc:     {posthoc_results.energy:12.6f} ± {posthoc_results.energy_std():.6f} Ha")

print(f"\nVariance:")
print(f"  Baseline:  {baseline_results.energy_variance:.6e}")
print(f"  Post-hoc:  {posthoc_results.energy_variance:.6e}")
print(f"  Reduction: {variance_reduction:.2f}x ({(1-1/variance_reduction)*100:.1f}% reduction)")

print(f"\nStandard deviation:")
print(f"  Baseline:  {baseline_results.energy_std():.6f} Ha")
print(f"  Post-hoc:  {posthoc_results.energy_std():.6f} Ha")
print(f"  Reduction: {std_reduction:.2f}x")

print(f"\nStatistical significance:")
print(f"  Baseline error: {baseline_error:.6f} Ha ({baseline_error_sigma:.2f}σ)")
print(f"  Post-hoc error: {posthoc_error:.6f} Ha ({posthoc_error_sigma:.2f}σ)")

if variance_reduction > 1.0:
    print(f"\n  ✓ Post-hoc repacking improves variance by {variance_reduction:.2f}x on hardware!")
else:
    print(f"\n  Note: Hardware noise may affect variance reduction")

print("=" * 80)
