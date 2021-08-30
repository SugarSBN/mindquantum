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
"""Test max_cut"""

import os

os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import mindspore as ms
from mindquantum.ansatz import MaxCutAnsatz
from mindquantum.nn import MQAnsatzOnlyLayer
from mindquantum.gate import Hamiltonian
from mindquantum.simulator import Simulator

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def test_max_cut():
    graph = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]
    depth = 3
    maxcut = MaxCutAnsatz(graph, depth)
    sim = Simulator('projectq', maxcut.circuit.n_qubits)
    ham = maxcut.hamiltonian
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(-ham), maxcut.circuit)
    ms.set_seed(42)
    net = MQAnsatzOnlyLayer(f_g_ops, len(maxcut.circuit.params_name))
    opti = ms.nn.Adagrad(net.trainable_params(), learning_rate=4e-1)
    train_net = ms.nn.TrainOneStepCell(net, opti)
    for i in range(50):
        cut = -train_net().asnumpy()[0]
    assert np.allclose(round(cut, 3), 4.831)
