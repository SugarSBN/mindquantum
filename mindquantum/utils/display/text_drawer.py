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
"""Text draw a circuit"""
import numpy as np

_text_drawer_config = {
    'ctrl_mask': '*',  #⨉
    'circ_line': '─',
    'ctrl_line': '|',
    'v_n': 1,
    'swap_mask': ['*', '*'],
    'edge_num': 2,
}

_text_drawer_config['edge'] = _text_drawer_config[
    'circ_line'] * _text_drawer_config['edge_num']


def _get_qubit_range(gate):
    """_get_qubit_range"""
    out = []
    out.extend(gate.obj_qubits)
    out.extend(gate.ctrl_qubits)
    return out


def brick_model(circ):
    """Split a circuit into layers."""
    n = circ.n_qubits
    v_n = _text_drawer_config['v_n']
    blocks = []
    qubit_hight = np.zeros(n, dtype=int)
    for gate in circ:
        qrange = _get_qubit_range(gate)
        max_hight = np.max(qubit_hight[qrange])
        if len(blocks) <= max_hight:
            blocks.append([])
        blocks[max_hight].append(gate)
        qubit_hight[range(min(qrange), max(qrange) + 1)] = max_hight + 1
    blocks = [_single_block_drawer(i, n) for i in blocks]
    res = {}
    max_q = 0
    for i in range(n):
        res[i * (v_n + 1)] = f'q{i}: '
        max_q = max(max_q, len(res[i * (v_n + 1)]))
    for i in range(n):
        res[i * (v_n + 1)] = res[i * (v_n + 1)].ljust(max_q, ' ')
        if i != n - 1:
            for j in range(v_n):
                res[i * (v_n + 1) + j + 1] = ' ' * max_q
    for block in blocks:
        for k, v in block.items():
            res[k] += v
    return '\n'.join([res[i] for i in range((n - 1) * (v_n + 1) + 1)])


def _single_gate_drawer(gate):
    """_single_gate_drawer"""
    import mindquantum.gate as G
    main_text = gate.name
    if issubclass(gate.__class__, G.ParameterGate):
        main_text = str(gate.__class__(gate.coeff))
    main_text = _text_drawer_config['edge'] + main_text + _text_drawer_config[
        'edge']
    res = {}
    for i in gate.obj_qubits:
        res[i] = main_text
    for i in gate.ctrl_qubits:
        res[i] = _text_drawer_config['ctrl_mask']
        res[i] = res[i].center(len(main_text),
                               _text_drawer_config['circ_line'])
    res['len'] = len(main_text)
    return res


def _single_block_drawer(block, n_qubits):
    """single block drawer"""
    v_n = _text_drawer_config['v_n']
    text_gates = {}
    for gate in block:
        text_gate = _single_gate_drawer(gate)
        qrange = _get_qubit_range(gate)
        for q in range(min(qrange), max(qrange) + 1):
            ind = q * (v_n + 1)
            if q in qrange:
                text_gates[ind] = text_gate[q]
            else:
                text_gates[
                    ind] = _text_drawer_config['circ_line'] * text_gate['len']
        for q in range(min(qrange), max(qrange)):
            for i in range(v_n):
                ind = q * (v_n + 1) + i + 1
                text_gates[ind] = _text_drawer_config['ctrl_line']
                text_gates[ind] = text_gates[ind].center(text_gate['len'], ' ')
    max_l = max([len(j) for j in text_gates.values()])
    for k, v in text_gates.items():
        if len(v) != max_l:
            if k % (v_n + 1) == 0:
                text_gates[k] = text_gates[k].center(
                    max_l, _text_drawer_config['circ_line'])
            else:
                text_gates[k] = text_gates[k].center(max_l, ' ')
    for i in range((n_qubits - 1) * v_n + n_qubits):
        if i not in text_gates:
            if i % (v_n + 1) == 0:
                text_gates[i] = _text_drawer_config['circ_line'] * max_l
            else:
                text_gates[i] = ' ' * max_l
    return text_gates
