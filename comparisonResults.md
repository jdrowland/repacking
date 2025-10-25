# Repacking Method Comparison Results

## H2O Molecule - Hartree-Fock State

Comparison of baseline (sorted insertion), post-hoc repacking, and greedy repacking methods on the H2O molecule (1620 Pauli terms, 65 groups).

### Corrected Simulation Results (1000 shots/group)

**Important**: Previous results contained a bug where post-hoc analysis used regenerated circuits instead of reusing baseline circuits. This has been corrected.

**Exact HF energy**: -75.679017 Ha

| Method | Energy (Ha) | Error (Ha) | Variance | Var. Reduction |
|--------|-------------|------------|----------|----------------|
| Baseline | -75.706501 ± 0.040264 | 0.027484 | 1.621×10⁻³ | 1.00x |
| Post-hoc | -75.698851 ± 0.019408 | 0.019834 | 3.767×10⁻⁴ | **4.30x** |
| Greedy | -75.706371 ± 0.017680 | 0.027354 | 3.126×10⁻⁴ | **5.19x** |

**Statistical Consistency (Baseline vs Post-hoc)**:
- Energy difference: 0.007650 Ha
- Combined std dev: 0.044698 Ha
- Difference: **0.17σ** ✓ (statistically consistent)

**Circuit Verification**:
- Post-hoc circuits matching baseline: **65/65 (100%)**
- Post-hoc analyzes the SAME measurements as baseline, just extracting additional Pauli expectations

### Group Statistics

- **Baseline**: 65 groups, 1,619 Paulis
- **Post-hoc**: 65 groups, 4,825 Paulis (+198% measurements from same data)
- **Greedy**: 65 groups, 5,057 Paulis (+212% measurements, different circuits)

### Key Findings

1. **Post-hoc repacking** achieves 83% of greedy's variance reduction (4.30x vs 5.19x) while maintaining **100% circuit reuse**
2. **Greedy repacking** provides the best variance reduction but requires regenerating circuits
3. **Statistical consistency confirmed**: Baseline and post-hoc produce statistically identical energies (0.17σ difference) because they analyze the same measurements
4. Both methods significantly outperform baseline (4-5x variance reduction)

### Bugfix Summary

**Previous Issue**: Post-hoc was incorrectly regenerating diagonalization circuits instead of reusing baseline circuits, causing:
- Large statistical discrepancies (400+ σ on hardware data)
- Incorrect comparison (different circuits = different measurements)

**Fix**: Created `create_posthoc_setups()` function that:
- Reuses baseline circuits for all groups
- For existing Paulis: reuses their diagonalized forms
- For added Paulis: computes diagonalization under baseline circuit
- Ensures post-hoc extracts more information from the SAME measurements

**Validation**:
- Hardware data (Sep 25): 427σ → 0.30σ discrepancy after fix
- Simulation data: 0.17σ discrepancy (expected statistical noise)

### Use Case Recommendations

- **Post-hoc repacking**: Ideal when measurement data has already been collected. Provides ~4x variance reduction through better post-processing without re-running any circuits. Perfect for hardware data analysis.
- **Greedy repacking**: Best when you can afford circuit regeneration and need maximum variance reduction (~5x). Useful for planning new experiments.
- **Baseline**: Acceptable for initial measurements with minimal overhead.

### Implementation Notes

**Post-hoc repacking** (CORRECTED approach):
1. Takes existing baseline measurement circuits (no regeneration!)
2. Adds Paulis from previous groups if they commute AND are diagonal under the baseline circuit
3. For existing Paulis: reuses their diagonalized forms from baseline
4. For added Paulis: computes their diagonalization under baseline circuit via conjugation
5. Analyzes the same measurement data, just extracting more Pauli expectations from it

This approach provides substantial variance reduction with **zero circuit overhead**, making it particularly valuable for post-processing existing quantum measurement data from real hardware.
