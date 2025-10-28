# Working Configuration for Energy Calculation

## Result
**Energy: -72.992977 Ha** ✓

This matches the expected result from the water_ibm.ipynb notebook (cells 86-87).

## Data Files

The magic combination uses:

1. **Conjugated Paulis**: `data/hardware_results/all_conjugated_paulis_fez_oct22_circuit_index_15.pkl`
   - 65 groups of conjugated Pauli strings
   - Total: 1619 Pauli terms

2. **Hardware Counts**: `data/hardware_results/all_counts_fez_oct22_3.pkl`
   - 65 count dictionaries from IBM hardware
   - ~50,000 shots per circuit
   - File size: 9.7 MB

3. **Hamiltonian**: `data/hamiltonians/monomer_eqb.hdf5` or `/tmp/calibrate-ibm/monomer_eqb.hdf5`
   - Water molecule molecular data
   - Used to extract identity term

## Critical: Cirq Pickle Patch

The old cirq pickle files are incompatible with cirq 1.3.0. You **must** apply this patch before loading:

```python
import cirq

def patched_new(cls, x=0):
    """Create LineQubit with _x initialized"""
    obj = object.__new__(cls)
    obj._x = x  # Directly set _x without calling __init__
    return obj

def patched_hash(self):
    """Handle hash being called before _x is set during unpickling"""
    if not hasattr(self, '_x'):
        return id(self)
    return hash((self._x, self._dimension))

def patched_setstate(self, state):
    """Handle old pickle format where __setstate__ gets empty dict"""
    if not state or state == {}:
        # Empty dict means object was initialized via __new__
        return
    self.__dict__.update(state)

# Apply patches
cirq.LineQubit.__new__ = staticmethod(patched_new)
cirq.LineQubit.__hash__ = patched_hash
cirq.LineQubit.__setstate__ = patched_setstate
```

## Calculation Workflow

```python
import pickle
import sys
import openfermion as of

# 1. Apply the cirq patch (see above)

# 2. Load the data
with open('data/hardware_results/all_conjugated_paulis_fez_oct22_circuit_index_15.pkl', 'rb') as f:
    conjugated_paulis = pickle.load(f)

with open('data/hardware_results/all_counts_fez_oct22_3.pkl', 'rb') as f:
    all_counts_fez = pickle.load(f)

# 3. Get identity term from Hamiltonian
mol_data = of.chem.MolecularData(filename="/tmp/calibrate-ibm/monomer_eqb.hdf5")
hamiltonian_fermion = of.jordan_wigner(
    of.get_fermion_operator(mol_data.get_molecular_hamiltonian())
)
hamiltonian_cirq = of.qubit_operator_to_pauli_sum(hamiltonian_fermion)
identity_term = list(hamiltonian_cirq)[0].coefficient

# 4. Compute expectation value
sys.path.insert(0, '/tmp/calibrate-ibm')
from expectation import compute_expectation

expval_fez = 0.0
for conjugated_group, counts in zip(conjugated_paulis, all_counts_fez):
    expval_fez += compute_expectation(sum(conjugated_group), counts, little_endian=False)

# 5. Add identity term to get energy
energy_fez = expval_fez + identity_term

print(f"expval = {expval_fez.real:.6f} Ha")
print(f"energy = {energy_fez.real:.6f} Ha")
```

## Output

```
expval = -26.132984 Ha
energy = -72.992977 Ha
```

## Why This Combination Works

- **circuit_index_15**: Contains the correct grouping of Pauli terms
- **oct22_3**: Contains the matching hardware measurement counts for those groups
- **Pickle patch**: Allows cirq 1.3.0 to load old pickle files that were created with an earlier cirq version

## Common Mistakes to Avoid

1. **Wrong file pairing**: Using circuit_index_15 with oct22_1 or oct22_2 gives wrong energy
2. **No patch**: Loading pickles without the patch causes `AttributeError: 'LineQubit' object has no attribute '_x'`
3. **Wrong patch**: Early attempts that called `__init__` in `__new__` corrupted the data
4. **Missing identity term**: Must add identity_term to expval to get total energy

## Working Test Script

See: `heisenberg_tests/test_minimal_patch.py`

This script demonstrates the complete working configuration.
