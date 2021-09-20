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
"""Utils"""

from .beauty_print import bprint
from .f import mod
from .f import normalize
from .f import random_state
from .f import ket_string
from .f import random_circuit
from . import display
from . import qasm
from .display import brick_model
from .qasm import OpenQASM
from .utils_operator import (number_operator, normal_ordered, count_qubits,
                             commutator, get_fermion_operator,
                             hermitian_conjugated, up_index, down_index,
                             sz_operator)

__all__ = [
    'bprint', 'mod', 'normalize', 'random_state', 'number_operator',
    'normal_ordered', 'commutator', 'up_index', 'down_index', 'sz_operator',
    'hermitian_conjugated', 'ket_string', 'get_fermion_operator',
    'count_qubits', 'OpenQASM', 'random_circuit', 'brick_model'
]

__all__.sort()