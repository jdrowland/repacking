import openfermion as of
import cirq
import numpy as np
from src.grouping.cache import GroupingCache
from src.measurement.simulation import simulate_all_groups
from src.measurement.estimation import estimate_from_groups

hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(
        of.chem.MolecularData(filename="data/hamiltonians/monomer_eqb.hdf5").get_molecular_hamiltonian()
    )
)
hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
baseline_cache = GroupingCache.load("data/caches/h2o_baseline_cache.pkl", hamiltonian_cirq)
posthoc_cache = GroupingCache.load("data/caches/h2o_posthoc_cache.pkl", hamiltonian_cirq)

mol_data = of.chem.MolecularData(filename="data/hamiltonians/monomer_eqb.hdf5")
n_electrons = mol_data.n_electrons

hf_circuit = cirq.Circuit()
for i in range(n_electrons):
    hf_circuit.append(cirq.X(baseline_cache.qubits[i]))

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

shots_per_group = 10000
simulator = cirq.Simulator()

baseline_counts, baseline_shots = simulate_all_groups(hf_circuit, baseline_cache.measurement_setups, shots_per_group, simulator)
posthoc_counts, posthoc_shots = simulate_all_groups(hf_circuit, posthoc_cache.measurement_setups, shots_per_group, simulator)

baseline_results = estimate_from_groups(baseline_cache.measurement_groups, baseline_cache.measurement_setups, baseline_counts, baseline_shots, baseline_cache.qubits)
posthoc_results = estimate_from_groups(posthoc_cache.measurement_groups, posthoc_cache.measurement_setups, posthoc_counts, posthoc_shots, posthoc_cache.qubits)

variance_reduction = baseline_results.energy_variance / posthoc_results.energy_variance
std_reduction = baseline_results.energy_std() / posthoc_results.energy_std()

print(f"Exact HF energy: {hf_energy_exact:.6f}")
print(f"Baseline: {baseline_results.energy:.6f} ± {baseline_results.energy_std():.6f}")
print(f"Post-hoc: {posthoc_results.energy:.6f} ± {posthoc_results.energy_std():.6f}")
print(f"Variance reduction: {variance_reduction:.2f}x ({(1-1/variance_reduction)*100:.1f}%)")
print(f"Std dev reduction: {std_reduction:.2f}x")

baseline_error_sigma = abs(baseline_results.energy - hf_energy_exact) / baseline_results.energy_std()
posthoc_error_sigma = abs(posthoc_results.energy - hf_energy_exact) / posthoc_results.energy_std()

print(f"\nBaseline error: {abs(baseline_results.energy - hf_energy_exact):.6f} ({baseline_error_sigma:.2f}σ)")
print(f"Post-hoc error: {abs(posthoc_results.energy - hf_energy_exact):.6f} ({posthoc_error_sigma:.2f}σ)")

if baseline_error_sigma < 5 and posthoc_error_sigma < 5 and variance_reduction > 1.0:
    print("✓ All checks passed")
else:
    print("✗ Checks failed")
    exit(1)
