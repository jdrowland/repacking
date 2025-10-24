import cirq
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from measurement import MeasurementSetup


def create_state_preparation_circuit(
    qubits: List[cirq.Qid],
    state_vector: Optional[np.ndarray] = None
) -> cirq.Circuit:
    if state_vector is None:
        return cirq.Circuit()

    state_vector = state_vector / np.linalg.norm(state_vector)
    circuit = cirq.Circuit()

    if len(qubits) == 1:
        a, b = state_vector[0], state_vector[1]
        theta = 2 * np.arccos(np.abs(a))
        phi = np.angle(b) - np.angle(a) if np.abs(b) > 1e-10 else 0

        if not np.isclose(theta, 0):
            circuit.append(cirq.ry(theta)(qubits[0]))
        if not np.isclose(phi, 0):
            circuit.append(cirq.rz(phi)(qubits[0]))

    return circuit


def simulate_measurement_group(
    state_prep_circuit: cirq.Circuit,
    measurement_setup: MeasurementSetup,
    num_shots: int,
    simulator: Optional[cirq.Simulator] = None
) -> Dict[str, int]:
    if simulator is None:
        simulator = cirq.Simulator()

    circuit = cirq.Circuit()
    for q in measurement_setup.qubits:
        circuit.append(cirq.I(q))

    circuit.append(state_prep_circuit)
    circuit.append(measurement_setup.basis_rotation)
    circuit.append(cirq.measure(*measurement_setup.qubits, key='result'))

    result = simulator.run(circuit, repetitions=num_shots)
    measurements = result.measurements['result']

    counts = {}
    for measurement in measurements:
        bitstring = ''.join(str(bit) for bit in measurement)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


def simulate_all_groups(
    state_prep_circuit: cirq.Circuit,
    measurement_setups: List[MeasurementSetup],
    shots_per_group: Union[int, List[int]],
    simulator: Optional[cirq.Simulator] = None
) -> Tuple[List[Dict[str, int]], List[int]]:
    group_counts = []
    shots_list = []

    if isinstance(shots_per_group, int):
        shots_per_group = [shots_per_group] * len(measurement_setups)

    for setup, num_shots in zip(measurement_setups, shots_per_group):
        counts = simulate_measurement_group(state_prep_circuit, setup, num_shots, simulator)
        group_counts.append(counts)
        shots_list.append(num_shots)

    return group_counts, shots_list
