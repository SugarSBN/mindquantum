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
"""Test hardware efficient ansatz"""

import os
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import mindspore as ms
from mindquantum.ansatz import HardwareEfficientAnsatz
from mindquantum.nn import MindQuantumAnsatzOnlyLayer as MAL
from mindquantum.gate import Hamiltonian, RX, RY, X
from mindquantum.ops import QubitOperator

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")


def test_hardware_efficient():
    depth = 3
    n_qubits = 3
    hea = HardwareEfficientAnsatz(n_qubits, [RX, RY, RX], X, 'all', depth)
    ham = QubitOperator('Z0 Z1 Z2')
    ms.set_seed(42)
    net = MAL(hea.circuit.para_name, hea.circuit, Hamiltonian(ham))
    opti = ms.nn.Adagrad(net.trainable_params(), learning_rate=4e-1)
    train_net = ms.nn.TrainOneStepCell(net, opti)
    for i in range(3):
        res = train_net().asnumpy()[0, 0]
    assert np.allclose(round(res, 4), -0.7588)
