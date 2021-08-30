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
"""Simulator."""
import numpy as np
from mindquantum.circuit import Circuit
from mindquantum.gate import Hamiltonian
from mindquantum.parameterresolver import ParameterResolver
from mindquantum.gate import MeasureResult
from mindquantum.gate import Measure
from .. import mqbackend as mb

SUPPORTED_SIMULATOR = ['projectq', 'quest']


def get_supported_simulator():
    """Get simulator name that supported by MindQuantum """
    return SUPPORTED_SIMULATOR


#TODO: 🔥🔥🔥🔥🔥《模拟器相关文档开发》↪️编写模拟器的文档
#TODO: 🔥🔥🔥《模拟器模块测试》↪️充分测试模拟器相关的接口
class Simulator:
    """Simulator"""
    def __init__(self, backend: str, n_qubits: int, seed: int = 1):
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️1.对后端进行校验
        self.backend = backend
        self.seed = seed
        if backend == 'projectq':
            self.sim = mb.projectq(seed, n_qubits)
        elif backend == 'quest':
            self.sim = mb.quest(n_qubits)

    def reset(self):
        """reset simulator"""
        self.sim.reset()

    def flush(self):
        """flush"""
        if self.backend == 'projectq':
            self.sim.run()

    def apply_measure(self, gate: Measure):
        """apply measure gate"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️2.对量子门进行校验
        return self.sim.apply_measure(gate.get_cpp_obj())

    def apply_gate(self, gate, parameter_resolver=None):
        """apply gate"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️3.对量子门和参数解析器进行校验
        if parameter_resolver is None:
            if gate.parameterized:
                raise ValueError(
                    "apply a parameterized gate needs a parameter_resolver")
            self.sim.apply_gate(gate.get_cpp_obj())
        else:
            self.sim.apply_gate(gate.get_cpp_obj(),
                                parameter_resolver.get_cpp_obj(), False)

    def apply_circuit(self, circuit: Circuit, parameter_resolver=None):
        """apply circuit"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️3.对量子线路和参数解析器进行校验，如果线路含有测量门，返回测量结果（参考sampling）
        if not isinstance(circuit, Circuit):
            raise TypeError(
                f"circuit must be Circuit, but get {type(Circuit)}")
        if parameter_resolver is None:
            if circuit.params_name:
                raise ValueError(
                    "Applying a parameterized circuit needs a parameter_resolver"
                )
            self.sim.apply_circuit(circuit.get_cpp_obj())
        else:
            if not isinstance(parameter_resolver,
                              (ParameterResolver, dict, np.ndarray)):
                raise TypeError(
                    f"parameter_resolver requires a ParameterResolver, but get {type(parameter_resolver)}"
                )
            if isinstance(parameter_resolver, dict):
                parameter_resolver = ParameterResolver(parameter_resolver)
            if isinstance(parameter_resolver, np.ndarray):
                if len(parameter_resolver.shape
                       ) != 1 or parameter_resolver.shape[0] != len(
                           circuit.params_name):
                    raise ValueError(
                        f"size of parameters input ({parameter_resolver.shape}) not\
match with circuit parameters ({len(circuit.params_name)}, )")
                parameter_resolver = ParameterResolver(
                    dict(zip(circuit.params_name, parameter_resolver)))
            self.sim.apply_circuit(circuit.get_cpp_obj(),
                                   parameter_resolver.get_cpp_obj())

    def sampling(self,
                 circuit: Circuit,
                 parameter_resolver=None,
                 shots: int = 1,
                 seed: int = None):
        """samping the measurement"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️4.对量子线路和参数解析器等进行校验
        if circuit.parameterized:
            if parameter_resolver is None:
                raise ValueError(
                    "Sampling a parameterized circuit need a ParameterResolver"
                )
            parameter_resolver = ParameterResolver(parameter_resolver)
        else:
            parameter_resolver = ParameterResolver()
        if seed is None:
            seed = self.seed
        res = MeasureResult()
        res.add_measure(circuit.all_measures.keys())
        res.samples = np.array(
            self.sim.sampling(circuit.get_cpp_obj(),
                              parameter_resolver.get_cpp_obj(), shots,
                              res.keys_map, seed)).reshape((shots, -1))
        res.collect_data()
        return res

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """apply hamiltonian"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️5.对哈密顿量进行校验，检查哈密顿量比特数跟模拟器比特数时候一致
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError(
                f"hamiltonian requires a Hamiltonian, but got {type(hamiltonian)}"
            )
        self.sim.apply_hamiltonian(hamiltonian.get_cpp_obj())

    def get_expectation(self, hamiltonian):
        """get expectation"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️6.对哈密顿量进行校验，检查哈密顿量比特数跟模拟器比特数时候一致
        return self.sim.get_expectation(hamiltonian.get_cpp_obj())

    def get_qs(self):
        """get quantum state"""
        return np.array(self.sim.get_qs())

    def get_expectation_with_grad(self,
                                  hams: Hamiltonian,
                                  circ_right: Circuit,
                                  circ_left: Circuit = None,
                                  encoder_params_name=None,
                                  ansatz_params_name=None,
                                  parallel_worker: int = None):
        """hermitian measure with grad"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️0.对相关接口进行校验（下面已经做完了，可作为参考）
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        elif not isinstance(hams, list):
            raise ValueError(
                f"hams requires a Hamiltonian or a list of Hamiltonian, but get {type(hams)}"
            )
        if not isinstance(circ_right, Circuit):
            raise ValueError(
                f"Quantum circuit need a Circuit, but get {type(circ_right)}")
        if circ_left is not None and not isinstance(circ_left, Circuit):
            raise ValueError(
                f"Quantum circuit need a Circuit, but get {type(circ_left)}")
        if circ_left is None:
            circ_left = Circuit()
        if circ_left.has_measure or circ_right.has_measure:
            raise ValueError(
                "circuit for variational algorithm cannot have measure gate")
        if parallel_worker is not None and not isinstance(
                parallel_worker, int):
            raise ValueError(
                f"parallel_worker need a integer, but get {type(parallel_worker)}"
            )
        if encoder_params_name is None and ansatz_params_name is None:
            encoder_params_name = []
            ansatz_params_name = [i for i in circ_right.params_name]
            for i in circ_left.params_name:
                if i not in ansatz_params_name:
                    ansatz_params_name.append(i)
        if encoder_params_name is not None and not isinstance(
                encoder_params_name, list):
            raise ValueError(
                f"encoder_params_name requires a list of str, but get {type(encoder_params_name)}"
            )
        if ansatz_params_name is not None and not isinstance(
                ansatz_params_name, list):
            raise ValueError(
                f"ansatz_params_name requires a list of str, but get {type(ansatz_params_name)}"
            )
        if encoder_params_name is None:
            encoder_params_name = []
        if ansatz_params_name is None:
            ansatz_params_name = []
        s1 = set(circ_right.params_name) | set(circ_left.params_name)
        s2 = set(encoder_params_name) | set(ansatz_params_name)
        if s1 - s2 or s2 - s1:
            raise ValueError(
                "encoder_params_name and ansatz_params_name are different with circuit parameters"
            )
        version = "both"
        if not ansatz_params_name:
            version = "encoder"
        if not encoder_params_name:
            version = "ansatz"

        def grad_ops(*inputs):
            if version == "both" and len(inputs) != 2:
                raise ValueError("Need two inputs!")
            if version in ("encoder", "ansatz") and len(inputs) != 1:
                raise ValueError("Need one input!")
            if version == "both":
                _check_encoder(inputs[0], len(encoder_params_name))
                _check_ansatz(inputs[1], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(
                    inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = inputs[1]
            if version == "encoder":
                _check_encoder(inputs[0], len(encoder_params_name))
                batch_threads, mea_threads = _thread_balance(
                    inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = np.array([])
            if version == "ansatz":
                _check_ansatz(inputs[0], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(
                    1, len(hams), parallel_worker)
                inputs0 = np.array([[]])
                inputs1 = inputs[0]
            if circ_left:
                f_g1_g2 = self.sim.non_hermitian_measure_with_grad(
                    [i.get_cpp_obj() for i in hams],
                    [i.get_cpp_obj(hermitian=True) for i in hams],
                    circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True),
                    circ_left.get_cpp_obj(),
                    circ_left.get_cpp_obj(hermitian=True), inputs0, inputs1,
                    encoder_params_name, ansatz_params_name, batch_threads,
                    mea_threads)
            else:
                f_g1_g2 = self.sim.hermitian_measure_with_grad(
                    [i.get_cpp_obj() for i in hams], circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True), inputs0, inputs1,
                    encoder_params_name, ansatz_params_name, batch_threads,
                    mea_threads)
            res = np.array(f_g1_g2)
            if version == 'both':
                f = res[:, :, 0]
                g1 = res[:, :, 1:1 + len(encoder_params_name)]
                g2 = res[:, :, 1 + len(encoder_params_name):]
                return f, g1, g2
            f = res[:, :, 0]
            g = res[:, :, 1:]
            return f, g

        return grad_ops


def _check_encoder(data, encoder_params_size):
    if not isinstance(data, np.ndarray):
        raise ValueError(
            f"encoder parameters need numpy array, but get {type(data)}")
    data_shape = data.shape
    if len(data_shape) != 2:
        raise ValueError("encoder data requires a two dimension numpy array")
    if data_shape[1] != encoder_params_size:
        raise ValueError(
            f"encoder parameters size do not match with encoder parameters name,\
need {encoder_params_size} but get {data_shape[1]}.")


def _check_ansatz(data, ansatz_params_size):
    """check ansatz"""
    if not isinstance(data, np.ndarray):
        raise ValueError(
            f"ansatz parameters need numpy array, but get {type(data)}")
    data_shape = data.shape
    if len(data_shape) != 1:
        raise ValueError("ansatz data requires a one dimension numpy array")
    if data_shape[0] != ansatz_params_size:
        raise ValueError(
            f"ansatz parameters size do not match with ansatz parameters name,\
need {ansatz_params_size} but get {data_shape[0]}")


def _thread_balance(n_prs, n_meas, parallel_worker):
    """threa balance"""
    if parallel_worker is None:
        parallel_worker = n_meas * n_prs
    if n_meas * n_prs <= parallel_worker:
        batch_threads = n_prs
        mea_threads = n_meas
    else:
        if n_meas < n_prs:
            batch_threads = min(n_prs, parallel_worker)
            mea_threads = min(n_meas, max(1, parallel_worker // batch_threads))
        else:
            mea_threads = min(n_meas, parallel_worker)
            batch_threads = min(n_prs, max(1, parallel_worker // mea_threads))
    return batch_threads, mea_threads
