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
"""Test nn."""

from mindquantum.nn.operations import MQAnsatzOnlyOps
from mindquantum.parameterresolver.parameterresolver import ParameterResolver
from mindquantum.nn.layer import MQLayer
from mindquantum.simulator.simulator import Simulator
import numpy as np
import mindspore as ms
from mindquantum.ops import QubitOperator
import mindquantum.gate as G
from mindquantum import Hamiltonian
from mindquantum.circuit import Circuit


def test_mindquantumlayer():
    """Test mindquantumlayer forward and backward."""
    encoder = Circuit()
    ansatz = Circuit()
    encoder += G.RX('e1').on(0)
    encoder += G.RY('e2').on(1)
    ansatz += G.X.on(1, 0)
    ansatz += G.RY('p1').on(0)
    ham = Hamiltonian(QubitOperator('Z0'))
    ms.set_seed(55)
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    circ = encoder + ansatz
    sim = Simulator('projectq', circ.n_qubits)
    f_g_ops = sim.get_expectation_with_grad(ham,
                                            circ,
                                            encoder_params_name=['e1', 'e2'],
                                            ansatz_params_name=['p1'])
    net = MQLayer(f_g_ops)
    encoder_data = ms.Tensor(np.array([[0.1, 0.2]]).astype(np.float32))
    res = net(encoder_data)
    assert round(float(res.asnumpy()[0, 0]), 6) == round(float(0.9949919), 6)


def test_evolution_state():
    a, b = 0.3, 0.5
    circ = Circuit([G.RX('a').on(0), G.RX('b').on(1)])
    data = ms.Tensor(np.array([a, b]).astype(np.float32))
    s = Simulator('projectq', circ.n_qubits)
    s.apply_circuit(circ, ParameterResolver({'a': a, 'b': b}))
    state = s.get_qs()
    state_exp = [
        0.9580325796404553, -0.14479246283091116j, -0.2446258794777393j,
        -0.036971585637570345
    ]
    assert np.allclose(state, state_exp)


def test_mindquantum_ansatz_only_ops():
    circ = Circuit(G.RX('a').on(0))
    data = ms.Tensor(np.array([0.5]).astype(np.float32))
    ham = Hamiltonian(QubitOperator('Z0'))
    sim = Simulator('projectq', circ.n_qubits)

    evol = MQAnsatzOnlyOps(sim.get_expectation_with_grad(ham, circ))
    output = evol(data)
    assert np.allclose(output.asnumpy(), [[8.77582550e-01]])
