import pickle
import cirq
from typing import List, Dict
from dataclasses import dataclass
from src.grouping.measurement_groups import MeasurementGroups
from src.measurement.setup import MeasurementSetup


@dataclass
class GroupingCache:
    hamiltonian_cirq: cirq.PauliSum
    qubits: List[cirq.Qid]
    measurement_groups: MeasurementGroups
    measurement_setups: List[MeasurementSetup]

    def num_groups(self) -> int:
        return len(self.measurement_setups)

    def num_terms(self) -> int:
        return len(self.measurement_groups.coefficients)

    def get_group_and_circuit(self, group_idx: int):
        return (
            self.measurement_setups[group_idx].paulis,
            self.measurement_setups[group_idx].basis_rotation
        )

    def save(self, filename: str):
        cache_data = {
            'qubits': self.qubits,
            'groups': self.measurement_groups.groups,
            'coefficients': self.measurement_groups.coefficients,
            'setups': [
                {
                    'group_index': setup.group_index,
                    'paulis': setup.paulis,
                    'diagonalized_paulis': setup.diagonalized_paulis,
                    'circuit': setup.basis_rotation
                }
                for setup in self.measurement_setups
            ]
        }

        with open(filename, 'wb') as f:
            pickle.dump(cache_data, f)

    @staticmethod
    def load(filename: str, hamiltonian_cirq: cirq.PauliSum) -> 'GroupingCache':
        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)

        qubits = cache_data['qubits']
        groups = cache_data['groups']
        coefficients = cache_data['coefficients']

        measurement_groups = MeasurementGroups(groups, coefficients)

        measurement_setups = [
            MeasurementSetup(
                group_index=setup_data['group_index'],
                paulis=setup_data['paulis'],
                diagonalized_paulis=setup_data['diagonalized_paulis'],
                qubits=qubits,
                basis_rotation=setup_data['circuit']
            )
            for setup_data in cache_data['setups']
        ]

        return GroupingCache(
            hamiltonian_cirq=hamiltonian_cirq,
            qubits=qubits,
            measurement_groups=measurement_groups,
            measurement_setups=measurement_setups
        )
