#!/usr/bin/env python3
"""Test simulation with CORRECTED post-hoc approach."""
import sys
sys.path.insert(0, '.')

import numpy as np
import openfermion as of
import cirq
from collections import Counter

from src.grouping import sorted_insertion_grouping
from src.repacking import greedy_repacking, posthoc_repacking
from src.measurement import create_measurement_setups, create_posthoc_setups
from src.measurement.estimation import estimate_from_groups

print("="*80)
print("CORRECTED Simulation Test (1000 shots/group)")
print("="*80)

# Load H2O Hamiltonian
print("\n[1/4] Loading H2O Hamiltonian...")
mol_data = of.chem.MolecularData(filename='data/hamiltonians/monomer_eqb.hdf5')
n_electrons = mol_data.n_electrons

hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(mol_data.get_molecular_hamiltonian())
)
hamiltonian = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
qubits = cirq.LineQubit.range(14)

# Compute exact HF energy
hf_energy_exact = 0.0
for pauli_string in hamiltonian:
    coeff = pauli_string.coefficient
    exp_value = 1.0
    for qubit in qubits:
        if qubit in pauli_string.qubits:
            gate = pauli_string[qubit]
            qubit_idx = qubit.x
            if qubit_idx < n_electrons:
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

print(f"✓ Exact HF energy: {hf_energy_exact:.6f} Ha")

# Generate all three groupings
print("\n[2/4] Generating groupings...")
baseline_groups = sorted_insertion_grouping(hamiltonian)
baseline_setups = create_measurement_setups(baseline_groups.groups, qubits)

greedy_groups = greedy_repacking(baseline_groups, verbose=False)
greedy_setups = create_measurement_setups(greedy_groups.groups, qubits)

posthoc_groups = posthoc_repacking(baseline_groups, baseline_setups, qubits, verbose=False)
# CORRECTED: Use baseline circuits for post-hoc
posthoc_setups = create_posthoc_setups(baseline_groups, baseline_setups, posthoc_groups, qubits)

print(f"✓ Baseline: {len(baseline_groups.groups)} groups, {sum(len(g) for g in baseline_groups.groups)} paulis")
print(f"✓ Greedy: {len(greedy_groups.groups)} groups, {sum(len(g) for g in greedy_groups.groups)} paulis")
print(f"✓ Post-hoc: {len(posthoc_groups.groups)} groups, {sum(len(g) for g in posthoc_groups.groups)} paulis")

# Verify post-hoc circuits match baseline
circuits_match = sum(1 for i in range(len(baseline_setups))
                     if baseline_setups[i].basis_rotation == posthoc_setups[i].basis_rotation)
print(f"✓ Post-hoc circuits matching baseline: {circuits_match}/{len(baseline_setups)}")

# Simulate measurements
print("\n[3/4] Running simulations (1000 shots/group)...")
hf_state = [1] * n_electrons + [0] * (len(qubits) - n_electrons)
shots_per_group = 1000

def simulate_measurements(groups, setups, state, shots):
    """Simulate measurements for all groups."""
    all_counts = []
    for group_idx, (group, setup) in enumerate(zip(groups.groups, setups)):
        circuit = setup.basis_rotation
        simulator = cirq.Simulator()
        initial_state_cirq = cirq.Circuit([cirq.X(qubits[i]) for i, bit in enumerate(state) if bit == 1])
        full_circuit = initial_state_cirq + circuit
        sampler = cirq.Simulator()
        samples = sampler.run(full_circuit + cirq.measure(*qubits), repetitions=shots)

        counts = Counter()
        for measurement in samples.measurements.values():
            for bitstring in measurement:
                key = ''.join(str(int(b)) for b in bitstring)
                counts[key] += 1

        all_counts.append(counts)
    return all_counts

print("  Simulating baseline...")
baseline_counts = simulate_measurements(baseline_groups, baseline_setups, hf_state, shots_per_group)

print("  Simulating greedy...")
greedy_counts = simulate_measurements(greedy_groups, greedy_setups, hf_state, shots_per_group)

print("  Simulating post-hoc (using BASELINE circuits)...")
# CRITICAL: For post-hoc, we already collected baseline_counts with baseline circuits
# We just analyze them differently! So we reuse baseline_counts for post-hoc
posthoc_counts = baseline_counts  # Same measurements as baseline!

# Estimate energies
print("\n[4/4] Computing energy estimates...")
shots_list = [shots_per_group] * len(baseline_groups.groups)

baseline_results = estimate_from_groups(baseline_groups, baseline_setups, baseline_counts, shots_list, qubits)
greedy_results = estimate_from_groups(greedy_groups, greedy_setups, greedy_counts, shots_list, qubits)
posthoc_results = estimate_from_groups(posthoc_groups, posthoc_setups, posthoc_counts, shots_list, qubits)

# Display results
print("\n" + "="*80)
print("CORRECTED RESULTS")
print("="*80)
print(f"Exact HF energy: {hf_energy_exact:.6f} Ha\n")

print(f"Baseline:")
print(f"  Energy: {baseline_results.energy:.6f} ± {baseline_results.energy_std():.6f} Ha")
print(f"  Error:  {abs(baseline_results.energy - hf_energy_exact):.6f} Ha")
print(f"  Variance: {baseline_results.energy_variance:.6e}")

print(f"\nGreedy Repacking (different circuits):")
print(f"  Energy: {greedy_results.energy:.6f} ± {greedy_results.energy_std():.6f} Ha")
print(f"  Error:  {abs(greedy_results.energy - hf_energy_exact):.6f} Ha")
print(f"  Variance: {greedy_results.energy_variance:.6e}")
print(f"  Variance reduction: {baseline_results.energy_variance / greedy_results.energy_variance:.2f}x")

print(f"\nPost-hoc Repacking (CORRECTED - same circuits as baseline):")
print(f"  Energy: {posthoc_results.energy:.6f} ± {posthoc_results.energy_std():.6f} Ha")
print(f"  Error:  {abs(posthoc_results.energy - hf_energy_exact):.6f} Ha")
print(f"  Variance: {posthoc_results.energy_variance:.6e}")
print(f"  Variance reduction: {baseline_results.energy_variance / posthoc_results.energy_variance:.2f}x")

# Statistical consistency check
energy_diff = abs(baseline_results.energy - posthoc_results.energy)
combined_std = np.sqrt(baseline_results.energy_variance + posthoc_results.energy_variance)
sigma_diff = energy_diff / combined_std

print(f"\nStatistical Consistency (Baseline vs Post-hoc):")
print(f"  Energy difference: {energy_diff:.6f} Ha")
print(f"  Combined std dev: {combined_std:.6f} Ha")
print(f"  Difference in sigmas: {sigma_diff:.2f}σ")

if sigma_diff < 3:
    print(f"  ✓ Statistically consistent (< 3σ)")
else:
    print(f"  ⚠ Large discrepancy (> 3σ)")

print("\n" + "="*80)
print("NOTE: Post-hoc analyzed the SAME measurements as baseline,")
print("just extracting additional Pauli expectations from them.")
print("This is what actually happens on hardware!")
print("="*80)
