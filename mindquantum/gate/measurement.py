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
"""Basic module for quantum gate."""
from collections.abc import Iterable
import numpy as np
from .basic import NoneParameterGate
from .. import mqbackend as mb


#TODO:  🔥🔥🔥🔥🔥《测量相关文档开发》↪️1.编写Measure门的文档
class Measure(NoneParameterGate):
    """Measurement gate"""
    def __init__(self, name=""):
        self.key = name
        NoneParameterGate.__init__(self, name)

    def get_cpp_obj(self):
        out = mb.get_measure_gate(self.key)
        out.obj_qubits = self.obj_qubits
        return out

    def __str__(self):
        info = ""
        if self.key and self.obj_qubits:
            info = f'({self.obj_qubits[0]}, key={self.key})'
        elif self.key:
            info = f'(key={self.key})'
        elif self.obj_qubits:
            info = f'({self.obj_qubits[0]})'
        return f"Measure{info}"

    def __repr__(self):
        return self.__str__()

    def on(self, obj_qubits, ctrl_qubits=None):
        """apply this measurement gate on which qubit"""
        #TODO: 🔥🔥🔥🔥🔥《测量相关接口校验》↪️1.对比特位进行校验
        if ctrl_qubits is not None:
            raise ValueError("Measure gate can not have control qubit")
        self.obj_qubits = [obj_qubits]
        if not self.key:
            self.key = str(obj_qubits)
        return self

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if self.key == other.key:
            return True
        return False

    def hermitian(self):
        """Hermitian gate of measure return its self"""
        return self.__class__(self.name).on(self.obj_qubits[0])

    def check_obj_qubits(self):
        if not self.obj_qubits:
            raise ValueError("Empty measure obj qubit")
        if len(self.obj_qubits) > 1:
            raise ValueError("Measure gate only apply on a single qubit")

    def define_projectq_gate(self):
        raise NotImplementedError


##TODO:  🔥🔥🔥🔥🔥《测量相关文档开发》↪️1.编写MeasureResult的文档
class MeasureResult:
    """Measurement result container"""
    def __init__(self):
        self.measures = []
        self.keys = []
        self.samples = np.array([])
        self.bit_string_data = {}

    def add_measure(self, measure):
        """add a measurement gate into this measurement result container."""
        #TODO: 🔥🔥🔥🔥🔥《测量相关接口校验》↪️2.对输入测量门进行校验
        if not isinstance(measure, Iterable):
            measure = [measure]
        for m in measure:
            if m.key in self.keys:
                raise ValueError(f"Measure key {m.key} already defined.")
            self.measures.append(m)
            self.keys.append(m.key)

    @property
    def keys_map(self):
        return {i: j for j, i in enumerate(self.keys)}

    def collect_data(self):
        """collect the measured bit string"""
        out = {}
        res = np.fliplr(self.samples)
        for s in res:
            s = ''.join([str(i) for i in s])
            if s in out:
                out[s] += 1
            else:
                out[s] = 1
        keys = sorted(list(out.keys()))
        self.bit_string_data = {key: out[key] for key in keys}

    @property
    def data(self):
        return self.bit_string_data
