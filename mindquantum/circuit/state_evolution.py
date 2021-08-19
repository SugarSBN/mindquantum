# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evaluate a quantum circuit."""

import numpy as np
from mindquantum.parameterresolver import ParameterResolver as PR
from mindquantum.utils import ket_string
from mindquantum.circuit import Circuit
from mindquantum.simulator import Simulator


def _generate_n_qubits_index(n_qubits):
    out = []
    for i in range(1 << n_qubits):
        out.append(bin(i)[2:].zfill(n_qubits))
    return out


class StateEvolution:
    """
    Calculate the final state of a parameterized or non parameterized quantum circuit.

    Args:
        circuit (Circuit): The circuit that you want to do evolution.
        backend (str): The simulation backend.

    Examples:
        >>> from mindquantum.circuit import StateEvolution
        >>> from mindquantum.circuit import qft
        >>> print(StateEvolution(qft([0, 1])).final_state(ket=True))
        0.5¦00⟩
        0.5¦01⟩
        0.5¦10⟩
        0.5¦11⟩
    """
    def __init__(self, circuit, backend='projectq'):
        if not isinstance(circuit, Circuit):
            raise TypeError(
                f'Input circuit should be a quantum circuit, but get {type(circuit)}'
            )
        self.circuit = circuit
        self.sim = Simulator(backend, circuit.n_qubits)
        self.index = _generate_n_qubits_index(self.circuit.n_qubits)

    def final_state(self, param=None, ket=False):
        """
        Get the final state of the input quantum circuit.

        Args:
            param (Union[numpy.ndarray, ParameterResolver, dict]): The
                parameter for the parameterized quantum circuit. If None, the
                quantum circuit should be a non parameterized quantum circuit.
                Default: None.
            ket (bool): Whether to print the final state in ket format. Default: False.

        Returns:
            numpy.ndarray, the final state in numpy array format.
        """
        if param is None:
            if self.circuit.params_name:
                raise ValueError(
                    "Require a non parameterized quantum circuit, since not parameters specified."
                )
            self.sim.apply_circuit(self.circuit)
            state = self.sim.get_qs()
            return state if not ket else '\n'.join(ket_string(state))
        if isinstance(param, (np.ndarray, PR, dict)):
            if isinstance(param, np.ndarray):
                if len(param.shape) != 1 and param.shape[0] != len(
                        self.circuit.params_name):
                    raise ValueError(
                        f"size of ndarray ({param.shape}) does not match with\
circuit parameters ({len(self.circuit.params_name)}, )")
                param = PR(dict(zip(self.circuit.params_name, param)))
            else:
                param = PR(param)
            self.sim.apply_circuit(self.circuit, param)
            state = self.sim.get_qs()
            return state if not ket else '\n'.join(ket_string(state))
        raise TypeError(
            f"parameter requires a numpy array or a ParameterResolver or a dict, ut get {type(param)}"
        )
