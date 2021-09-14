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
from mindquantum.gate.hamiltonian import MODE
from mindquantum.parameterresolver import ParameterResolver
from mindquantum.gate import MeasureResult
from mindquantum.gate import Measure
from mindquantum.gate import BarrierGate
from .. import mqbackend as mb

SUPPORTED_SIMULATOR = ['projectq', 'quest']


def get_supported_simulator():
    """Get simulator name that supported by MindQuantum """
    return SUPPORTED_SIMULATOR


class Simulator:
    """
    Quantum simulator that simulate quantum circuit.

    Args:
        backend (str): which backend you want. The supported backend can be found
            in SUPPORTED_SIMULATOR
        n_qubits (int): number of quantum simulator.
        seed (int): the random seed for this simulator. Default: 42.

    Examples:
        >>> from mindquantum import Simulator
        >>> from mindquantum import qft
        >>> sim = Simulator('projectq', 2)
        >>> sim.apply_circuit(qft(range(2)))
        >>> sim.get_qs()
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
    """
    def __init__(self, backend, n_qubits, seed=42):
        if not isinstance(backend, str):
            raise TypeError(f"backend need a string, but get {type(backend)}")
        if backend not in SUPPORTED_SIMULATOR:
            raise ValueError(f"backend {backend} not supported!")
        if not isinstance(n_qubits, int) or n_qubits < 0:
            raise ValueError(
                f"n_qubits of simulator should be a non negative int, but get {n_qubits}"
            )
        if not isinstance(seed, int) or seed < 0 or seed > 2**32 - 1:
            raise ValueError(f"seed must be between 0 and 2**32 - 1")
        self.backend = backend
        self.seed = seed
        self.n_qubits = n_qubits
        if backend == 'projectq':
            self.sim = mb.projectq(seed, n_qubits)
        elif backend == 'quest':
            self.sim = mb.quest(n_qubits)

    def reset(self):
        """
        Reset simulator to zero state.

        Examples:
            >>> from mindquantum import Simulator
            >>> from mindquantum import qft
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.reset()
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])        
        """
        self.sim.reset()

    def flush(self):
        """
        Flush gate that works for projectq simulator. The projectq simulator
        will cache several gate and fushion these gate into a bigger gate, and
        than act on the quantum state. The flush command will ask the simulator
        to fushion currently stored gate and act on the quantum state.

        Examples:
            >>> from mindquantum import Simulator
            >>> from mindquantum import H
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(H.on(0))
            >>> sim.flush()
        """
        if self.backend == 'projectq':
            self.sim.run()

    def apply_measure(self, gate):
        """
        Apply a measure gate.

        Args:
            gate (Measure): a measure gate.

        Returns:
            int, the qubit measurement result.

        Examples:
            >>> from mindquantum import Simulator
            >>> from mindquantum import Measure, H
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(H.on(0))
            >>> sim.apply_measure(Measure().on(0))
            1
            >>> sim.get_qs()
            array([0.+0.j, 1.+0.j])
        """
        if not isinstance(gate, Measure):
            raise TypeError(
                f"simulator apply_measure requires a Measure, but get {type(gate)}"
            )
        return self.sim.apply_measure(gate.get_cpp_obj())

    def apply_gate(self, gate, parameter_resolver=None):
        """apply gate"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️3.对量子门和参数解析器进行校验
        if not isinstance(gate, BarrierGate):
            if parameter_resolver is None:
                if gate.parameterized:
                    raise ValueError(
                        "apply a parameterized gate needs a parameter_resolver"
                    )
                self.sim.apply_gate(gate.get_cpp_obj())
            else:
                self.sim.apply_gate(gate.get_cpp_obj(),
                                    parameter_resolver.get_cpp_obj(), False)

    def apply_circuit(self, circuit: Circuit, parameter_resolver=None):
        """apply circuit"""
        #TODO: 🔥🔥🔥🔥🔥《模拟器接口校验》↪️4.对量子线路和参数解析器进行校验，如果线路含有测量门，返回测量结果（参考sampling）
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

    def sampling(self, circuit, parameter_resolver=None, shots=1, seed=None):
        """
        Samping the measure qubit in circuit.

        Args:
            circuit (Circuit): The circuit that you want to evolution and do sampling.
            Parameter_resolver (Union[None, dict, ParameterResolver]): The parameter
                resolver for this circuit, if this circuit is a parameterized circuit.
                Default: None.
            shots (int): How many shots you want to sampling this circuit. Default: 1
            seed (int): Random seed for random sampling. Default: None.

        Returns:
            MeasureResult, the measure result of sampling.

        Examples:
            >>> from mindquantum import Circuit, Measure
            >>> from mindquantum import Simulator
            >>> circ = Circuit().ry('a', 0).ry('b', 1)
            >>> circ += Measure('q0_0').on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ += Measure('q1').on(1)
            >>> sim = Simulator('projectq', circ.n_qubits)
            >>> res = sim.sampling(circ, {'a': 1.1, 'b': 2.2}, shots=100, seed=42)
            >>> res
            {'000': 17, '011': 8, '100': 49, '111': 26}
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(
                f"sampling circuit need a quantum circuit but get {type(circuit)}"
            )
        if not isinstance(shots, int) or shots < 0:
            raise ValueError(
                f"sampling shot should be non negative int, but get {shots}")
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
        elif not isinstance(seed, int) or seed < 0 or seed > 2**23 - 1:
            raise ValueError(f"seed must be between 0 and 2**23 - 1")
        res = MeasureResult()
        res.add_measure(circuit.all_measures.keys())
        res.samples = np.array(
            self.sim.sampling(circuit.get_cpp_obj(),
                              parameter_resolver.get_cpp_obj(), shots,
                              res.keys_map, seed)).reshape((shots, -1))
        res.collect_data()
        return res

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """
        Apply hamiltonian to a simulator, this hamiltonian can be
        hermitian or non hermitian.

        Notes:
            The quantum state may be not a normalized quantum state after apply hamiltonian.

        Args:
            hamiltonian (Hamiltonian): the hamiltonian you want to apply.

        Examples:
            >>> from mindquantum import Simulator
            >>> from mindquantum import Circuit, Hamiltonian
            >>> from mindquantum.ops import QubitOperator
            >>> import scipy.sparse as sp
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_circuit(Circuit().h(0))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
            >>> ham1 = Hamiltonian(QubitOperator('Z0'))
            sim.apply_hamiltonian(ham1)
            >>> sim.get_qs()
            array([ 0.70710678+0.j, -0.70710678+0.j])

            >>> sim.reset()
            >>> ham2 = Hamiltonian(sp.csr_matrix([[1, 2], [3, 4]]))
            >>> sim.apply_hamiltonian(ham2)
            >>> sim.get_qs()
            array([1.+0.j, 3.+0.j])
        """

        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError(
                f"hamiltonian requires a Hamiltonian, but got {type(hamiltonian)}"
            )
        if hamiltonian.how_to != MODE['origin']:
            if hamiltonian.n_qubits != self.n_qubits:
                raise ValueError(
                    f"Hamiltonian qubits is {hamiltonian.n_qubits}, not match \
with simulator qubits number {self.n_qubits}")
        self.sim.apply_hamiltonian(hamiltonian.get_cpp_obj())

    def get_expectation(self, hamiltonian):
        """
        Get expectation of the given hamiltonian. The hamiltonian could be non hermitian.

        .. math::

            E = \left<\psi\right|H\left|\psi\right>

        Args:
            hamiltonian (Hamiltonian): The hamiltonian you want to get expectation.

        Examples:
            >>> from mindquantum.ops import QubitOperator
            >>> from mindquantum import Circuit, Simulator
            >>> from mindquantum import Hamiltonian
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_circuit(Circuit().ry(1.2, 0))
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim.get_expectation(ham)
            (0.36235775447667357+0j)
        """
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError(
                f"hamiltonian requires a Hamiltonian, but got {type(hamiltonian)}"
            )
        if hamiltonian.how_to != MODE['origin']:
            if hamiltonian.n_qubits != self.n_qubits:
                raise ValueError(
                    f"Hamiltonian qubits is {hamiltonian.n_qubits}, not match \
with simulator qubits number {self.n_qubits}")
        return self.sim.get_expectation(hamiltonian.get_cpp_obj())

    def get_qs(self):
        """
        Get current quantum state of this simulator.

        Returns:
            numpy.ndarray, the current quantum state.

        Examples:
            >>> from mindquantum import qft, Simulator
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.get_qs()
            array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
        """
        return np.array(self.sim.get_qs())

    def set_qs(self, vec):
        """
        Set quantum state for this simulation.

        Args:
            vec (numpy.ndarray): the quantum state that you want.

        Examples:
            >>> from mindquantum import Simulator
            >>> import numpy as np
            >>> sim = Simulator('projectq', 1)
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j])
            >>> sim.set_qs(np.array([1, 1]))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
        """
        if not isinstance(vec, np.ndarray):
            raise TypeError(
                f"quantum state must be a ndarray, but get {type(vec)}")
        if len(vec.shape) != 1:
            raise ValueError(
                f"vec requires a 1-dimensional array, but get {vec.shape}")
        n_qubits = np.log2(vec.shape[0])
        if n_qubits % 1 != 0:
            raise ValueError(f"vec size {vec.shape[0]} is not power of 2")
        n_qubits = int(n_qubits)
        if self.n_qubits != n_qubits:
            raise ValueError(
                f"{n_qubits} qubits vec does not match with simulation qubits ({self.n_qubits})"
            )
        self.sim.set_qs(vec / np.sqrt(np.sum(vec**2)))

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

        grad_wrapper = GradOpsWrapper(grad_ops, hams, circ_right, circ_left,
                                      encoder_params_name, ansatz_params_name,
                                      parallel_worker)
        return grad_wrapper


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


class GradOpsWrapper:
    """GradOpsWrapper"""
    def __init__(self, grad_ops, hams, circ_right, circ_left,
                 encoder_params_name, ansatz_params_name, parallel_worker):
        self.grad_ops = grad_ops
        self.hams = hams
        self.circ_right = circ_right
        self.circ_left = circ_left
        self.encoder_params_name = encoder_params_name
        self.ansatz_params_name = ansatz_params_name
        self.parallel_worker = parallel_worker

    def __call__(self, *args):
        return self.grad_ops(*args)
