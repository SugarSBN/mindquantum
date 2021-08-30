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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test simulator."""
import numpy as np
from mindquantum import Circuit
from mindquantum import Simulator
from mindquantum import ParameterResolver as PR
import mindquantum.gate as G


def test_init_reset():
    s1 = Simulator('projectq', 2)
    s2 = Simulator('quest', 2)
    circ = Circuit().h(0).h(1)
    v1 = s1.get_qs()
    v2 = s2.get_qs()
    s1.apply_circuit(circ)
    s2.apply_circuit(circ)
    s1.reset()
    s2.reset()
    v3 = s1.get_qs()
    v4 = s1.get_qs()
    v = np.array([1, 0, 0, 0], dtype=np.complex128)
    assert np.allclose(v1, v)
    assert np.allclose(v1, v2)
    assert np.allclose(v1, v3)
    assert np.allclose(v1, v4)


def test_apply_circuit_and_hermitian():
    sv0 = np.array([[1, 0], [0, 0]])
    sv1 = np.array([[0, 0], [0, 1]])
    circ = Circuit()
    circ.ry(1.2, 0).ry(3.4, 1)
    circ.h(0).h(1)
    circ.x(1, 0)
    circ.rx('a', 0).ry('b', 1)
    circ.zz('c', (0, 1)).z(1, 0)
    s1 = Simulator('projectq', circ.n_qubits)
    s2 = Simulator('quest', circ.n_qubits)
    pr = PR({'a': 1, 'b': 3, 'c': 5})
    s1.apply_circuit(circ, pr)
    s2.apply_circuit(circ, pr)
    v1 = s1.get_qs()
    v2 = s2.get_qs()
    m = np.kron(G.RY(3.4).matrix(), G.RY(1.2).matrix())
    m = np.kron(G.H.matrix(), G.H.matrix()) @ m
    m = (np.kron(G.I.matrix(), sv0) + np.kron(G.X.matrix(), sv1)) @ m
    m = np.kron(G.RY(3).matrix(), G.RX(1).matrix()) @ m
    m = G.ZZ(5).matrix() @ m
    m = (np.kron(G.I.matrix(), sv0) + np.kron(G.Z.matrix(), sv1)) @ m
    v = m[:, 0]
    assert np.allclose(v, v1)
    assert np.allclose(v, v2)

    circ2 = circ.hermitian
    s1.reset()
    s2.reset()
    s1.apply_circuit(circ2, pr)
    s2.apply_circuit(circ2, pr)
    m = np.conj(m.T)
    v1 = s1.get_qs()
    v2 = s2.get_qs()
    v = m[:, 0]
    assert np.allclose(v, v1)
    assert np.allclose(v, v2)
