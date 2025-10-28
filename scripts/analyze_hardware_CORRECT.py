import pickle
import sys
import openfermion as of
import cirq
import numpy as np
from src.grouping.cache import GroupingCache
from collections import defaultdict

print("=" * 80)
print("Analyzing IBM Hardware Data - Properly separating baseline and post-hoc")
print("=" * 80)

# Apply cirq pickle patch FIRST
print("\n[0/5] Applying cirq pickle patch...")
def patched_new(cls, x=0):
    obj = object.__new__(cls)
    obj._x = x
    return obj

def patched_hash(self):
    if not hasattr(self, '_x'):
        return id(self)
    return hash((self._x, self._dimension))

def patched_setstate(self, state):
    if not state or state == {}:
        return
    self.__dict__.update(state)

cirq.LineQubit.__new__ = staticmethod(patched_new)
cirq.LineQubit.__hash__ = patched_hash
cirq.LineQubit.__setstate__ = patched_setstate
print("  ✓ Cirq patch applied")

# Import compute_expectation from calibrate-ibm
sys.path.insert(0, '/tmp/calibrate-ibm')
from expectation import compute_expectation

print("\n[1/5] Loading hardware counts and caches...")
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

baseline_total = sum(len(g) for g in baseline_cache.measurement_groups.groups)
posthoc_total = sum(len(g) for g in posthoc_cache.measurement_groups.groups)
print(f"  Baseline measurements: {baseline_total}")
print(f"  Post-hoc measurements: {posthoc_total} (+{posthoc_total - baseline_total})")

# Get identity term
identity_term = list(hamiltonian_cirq)[0].coefficient

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

def analyze_grouping(cache, counts, name):
    """Analyze a specific grouping (baseline or post-hoc)."""
    print(f"\n[{name}] Computing energy...")

    # For each Hamiltonian term, collect expectation values from all groups that measure it
    term_measurements = defaultdict(list)

    for pauli, coeff in cache.measurement_groups.coefficients.items():
        # Find which groups measure this Pauli
        group_indices = cache.measurement_groups.get_group_containing(pauli)

        if len(group_indices) == 0:
            # Identity or unmeasured term
            if len(pauli) == 0:
                term_measurements[pauli].append((1.0, 1))
            continue

        for group_idx in group_indices:
            setup = cache.measurement_setups[group_idx]
            pauli_idx = setup.paulis.index(pauli)
            pauli_diag = setup.diagonalized_paulis[pauli_idx]

            # Compute expectation value using the correct bitstring convention
            exp = compute_expectation(cirq.PauliSum.from_pauli_strings([pauli_diag]),
                                    counts[group_idx], little_endian=False)
            term_measurements[pauli].append((exp, shots_per_group[group_idx]))

    # Pool measurements and compute energy/variance
    energy = 0.0
    variance = 0.0

    for pauli, measurements in term_measurements.items():
        coeff = cache.measurement_groups.coefficients[pauli]
        total_shots = sum(s for _, s in measurements)
        pooled_exp = sum(e * s for e, s in measurements) / total_shots if total_shots > 0 else 0.0

        energy += coeff * pooled_exp
        variance += (coeff**2) * ((1.0 - pooled_exp**2) / total_shots) if total_shots > 0 else 0.0

    return energy, variance, term_measurements

baseline_energy, baseline_variance, baseline_terms = analyze_grouping(baseline_cache, hardware_counts, "BASELINE")
posthoc_energy, posthoc_variance, posthoc_terms = analyze_grouping(posthoc_cache, hardware_counts, "POST-HOC")

# Extract real parts
baseline_energy = np.real(baseline_energy)
baseline_variance = np.real(baseline_variance)
posthoc_energy = np.real(posthoc_energy)
posthoc_variance = np.real(posthoc_variance)

print(f"\n  Baseline energy: {baseline_energy:.6f} ± {np.sqrt(baseline_variance):.6f} Ha")
print(f"  Post-hoc energy: {posthoc_energy:.6f} ± {np.sqrt(posthoc_variance):.6f} Ha")

baseline_error = abs(baseline_energy - hf_energy_exact)
posthoc_error = abs(posthoc_energy - hf_energy_exact)

print(f"\n  Baseline error: {baseline_error:.6f} Ha")
print(f"  Post-hoc error: {posthoc_error:.6f} Ha")

variance_reduction = baseline_variance / posthoc_variance if posthoc_variance > 0 else float('inf')
std_reduction = np.sqrt(baseline_variance) / np.sqrt(posthoc_variance) if posthoc_variance > 0 else float('inf')

print("\n" + "=" * 80)
print("Hardware Results Summary")
print("=" * 80)

print(f"\nTotal shots: {total_shots:,} ({shots_per_group[0]:,} per group)")
print(f"Baseline terms measured: {len(baseline_terms)}")
print(f"Post-hoc terms measured: {len(posthoc_terms)}")

print(f"\nEnergy estimates:")
print(f"  Exact:        {hf_energy_exact:12.6f} Ha")
print(f"  Baseline:     {baseline_energy:12.6f} ± {np.sqrt(baseline_variance):.6f} Ha")
print(f"  Post-hoc:     {posthoc_energy:12.6f} ± {np.sqrt(posthoc_variance):.6f} Ha")

print(f"\nVariance:")
print(f"  Baseline:  {baseline_variance:.6e}")
print(f"  Post-hoc:  {posthoc_variance:.6e}")
if posthoc_variance > 0:
    print(f"  Reduction: {variance_reduction:.2f}x ({(1-1/variance_reduction)*100:.1f}% reduction)")

print(f"\nStandard deviation:")
print(f"  Baseline:  {np.sqrt(baseline_variance):.6f} Ha")
print(f"  Post-hoc:  {np.sqrt(posthoc_variance):.6f} Ha")
if posthoc_variance > 0:
    print(f"  Reduction: {std_reduction:.2f}x")

if variance_reduction > 1.0:
    print(f"\n✓ Post-hoc repacking improves variance by {variance_reduction:.2f}x on hardware!")
elif variance_reduction < 1.0:
    print(f"\n✗ Warning: Post-hoc variance is HIGHER ({1/variance_reduction:.2f}x)")
else:
    print(f"\nNote: Variances are equal (may indicate same grouping used)")

print("=" * 80)
