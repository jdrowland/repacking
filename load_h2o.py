import openfermion as of
import cirq
import numpy as np

print("=" * 80)
print("Loading H2O Hamiltonian")
print("=" * 80)

print("\nLoading from monomer_eqb.hdf5...")
hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(
        of.chem.MolecularData(filename="monomer_eqb.hdf5").get_molecular_hamiltonian()
    )
)

print(f"Hamiltonian type: {type(hamiltonian_fermion)}")
print(f"Number of terms: {len(hamiltonian_fermion.terms)}")

qubits_used = set()
for term in hamiltonian_fermion.terms:
    for qubit_idx, _ in term:
        qubits_used.add(qubit_idx)

print(f"Number of qubits: {len(qubits_used)}")
print(f"Qubit indices: {sorted(qubits_used)}")

print("\nConverting to Cirq PauliSum...")
hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)

print(f"Cirq PauliSum type: {type(hamiltonian_cirq)}")
print(f"Number of terms: {len(list(hamiltonian_cirq))}")

print("\n" + "=" * 80)
print("Analyzing coefficient distribution")
print("=" * 80)

coefficients = []
for pauli_string in hamiltonian_cirq:
    coefficients.append(abs(pauli_string.coefficient))

coefficients = np.array(coefficients)
print(f"\nMin coefficient: {coefficients.min():.6e}")
print(f"Max coefficient: {coefficients.max():.6e}")
print(f"Mean coefficient: {coefficients.mean():.6e}")
print(f"Median coefficient: {np.median(coefficients):.6e}")

percentiles = [10, 25, 50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    print(f"  {p:2d}th: {np.percentile(coefficients, p):.6e}")

print("\n" + "=" * 80)
print("Analyzing Pauli string structure")
print("=" * 80)

string_lengths = []
for pauli_string in hamiltonian_cirq:
    string_lengths.append(len(pauli_string))

string_lengths = np.array(string_lengths)
print(f"\nMin length (qubit support): {string_lengths.min()}")
print(f"Max length (qubit support): {string_lengths.max()}")
print(f"Mean length: {string_lengths.mean():.2f}")

print(f"\nLength distribution:")
unique, counts = np.unique(string_lengths, return_counts=True)
for length, count in zip(unique, counts):
    print(f"  Length {length:2d}: {count:4d} terms ({count/len(string_lengths)*100:5.1f}%)")

print("\n" + "=" * 80)
print("Sample terms (top 10 by coefficient)")
print("=" * 80)

terms_sorted = sorted(hamiltonian_cirq, key=lambda ps: abs(ps.coefficient), reverse=True)
for i, pauli_string in enumerate(terms_sorted[:10]):
    print(f"{i+1:2d}. {abs(pauli_string.coefficient):10.6f} * {pauli_string}")

print("\n" + "=" * 80)
print("Saving Hamiltonian as pickle")
print("=" * 80)

import pickle
with open('h2o_hamiltonian.pkl', 'wb') as f:
    pickle.dump(hamiltonian_cirq, f)

print("Saved to h2o_hamiltonian.pkl")
print("\n" + "=" * 80)
