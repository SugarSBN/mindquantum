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
"""Mindspore quantum simulator operator."""
import numpy as np
import mindspore as ms
import mindspore.nn as nn


class MQOps(nn.Cell):
    """MindQuantum operator"""
    def __init__(self, expectation_with_grad):
        super(MQOps, self).__init__()
        self.expectation_with_grad = expectation_with_grad

    def construct(self, enc_data, ans_data):
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        f = ms.Tensor(np.real(f), dtype=ms.float32)
        self.g_enc = np.real(g_enc)
        self.g_ans = np.real(g_ans)
        return f

    def bprop(self, enc_data, ans_data, out, dout):
        dout = dout.asnumpy()
        enc_grad = np.einsum('smp,sm->sp', self.g_enc, dout)
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return ms.Tensor(enc_grad,
                         dtype=ms.float32), ms.Tensor(ans_grad,
                                                      dtype=ms.float32)


class MQN2Ops(nn.Cell):
    """MindQuantum norm square operator"""
    def __init__(self, expectation_with_grad):
        super(MQN2Ops, self).__init__()
        self.expectation_with_grad = expectation_with_grad

    def construct(self, enc_data, ans_data):
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        self.f = f
        f = ms.Tensor(np.abs(f)**2, dtype=ms.float32)
        self.g_enc = g_enc
        self.g_ans = g_ans
        return f

    def bprop(self, enc_data, ans_data, out, dout):
        dout = dout.asnumpy()
        enc_grad = 2 * np.real(
            np.einsum('smp,sm,sm->sp', self.g_enc, dout, np.conj(self.f)))
        ans_grad = 2 * np.real(
            np.einsum('smp,sm,sm->p', self.g_ans, dout, np.conj(self.f)))
        return ms.Tensor(enc_grad,
                         dtype=ms.float32), ms.Tensor(ans_grad,
                                                      dtype=ms.float32)


class MQAnsatzOnlyOps(nn.Cell):
    """MindQuantum ansatz only operator"""
    def __init__(self, expectation_with_grad):
        super(MQAnsatzOnlyOps, self).__init__()
        self.expectation_with_grad = expectation_with_grad

    def construct(self, x):
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        f = ms.Tensor(np.real(f[0]), dtype=ms.float32)
        self.g = np.real(g[0])
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = dout @ self.g
        return ms.Tensor(grad, dtype=ms.float32)


class MQN2AnsatzOnlyOps(nn.Cell):
    """MindQuantum norm square ansatz only operator"""
    def __init__(self, expectation_with_grad):
        super(MQN2AnsatzOnlyOps, self).__init__()
        self.expectation_with_grad = expectation_with_grad

    def construct(self, x):
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        self.f = f[0]
        f = ms.Tensor(np.abs(f[0])**2, dtype=ms.float32)
        self.g = g[0]
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = 2 * np.real(
            np.einsum('m,m,mp->p', np.conj(self.f), dout, self.g))
        return ms.Tensor(grad, dtype=ms.float32)


class MQEncoderOnlyOps(nn.Cell):
    """MindQuantum encoder only operator"""
    def __init__(self, expectation_with_grad):
        super(MQEncoderOnlyOps, self).__init__()
        self.expectation_with_grad = expectation_with_grad

    def construct(self, x):
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        f = ms.Tensor(np.real(f), dtype=ms.float32)
        self.g = np.real(g)
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = np.einsum('smp,sm->sp', self.g, dout)
        return ms.Tensor(grad, dtype=ms.float32)


class MQN2EncoderOnlyOps(nn.Cell):
    """MindQuantum norm square encoder only operator"""
    def __init__(self, expectation_with_grad):
        super(MQN2EncoderOnlyOps, self).__init__()
        self.expectation_with_grad = expectation_with_grad

    def construct(self, x):
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        self.f = f
        f = ms.Tensor(np.abs(f)**2, dtype=ms.float32)
        self.g = g
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = 2 * np.real(
            np.einsum('smp,sm,sm->sp', self.g, dout, np.conj(self.f)))
        return ms.Tensor(grad, dtype=ms.float32)
