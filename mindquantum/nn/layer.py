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
"""Mindspore quantum simulator layer."""
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from .operations import MQOps
from .operations import MQN2Ops
from .operations import MQAnsatzOnlyOps
from .operations import MQN2AnsatzOnlyOps


class MQLayer(nn.Cell):
    """MindQuantum trainable layer"""
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQLayer, self).__init__()
        self.evolution = MQOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32),
                                name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)


class MQN2Layer(nn.Cell):
    """MindQuantum norm square trainable layer"""
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQN2Layer, self).__init__()
        self.evolution = MQN2Ops(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32),
                                name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)


class MQAnsatzOnlyLayer(nn.Cell):
    """MindQuantum ansatz only trainable layer"""
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQAnsatzOnlyLayer, self).__init__()
        self.evolution = MQAnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32),
                                name='ansatz_weight')

    def construct(self):
        return self.evolution(self.weight)


class MQN2AnsatzOnlyLayer(nn.Cell):
    """MindQuantum norm square ansatz only trainable layer"""
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQN2AnsatzOnlyLayer, self).__init__()
        self.evolution = MQN2AnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32),
                                name='ansatz_weight')

    def construct(self):
        return self.evolution(self.weight)
