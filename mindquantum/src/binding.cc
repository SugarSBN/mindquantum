/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "backends/projectq/projectq.h"
#include "backends/quest/quest.h"
#include "gate/gates.h"
#include "hamiltonian/hamiltonian.h"
#include "matrix/two_dim_matrix.h"
#include "pr/parameter_resolver.h"
#include "sparse/algo.h"
#include "sparse/csrhdmatrix.h"
#include "sparse/paulimat.h"

namespace py = pybind11;
namespace mindquantum {
using mindquantum::sparse::Csr_Plus_Csr;
using mindquantum::sparse::GetPauliMat;
using mindquantum::sparse::PauliMat;
using mindquantum::sparse::PauliMatToCsrHdMatrix;
using mindquantum::sparse::SparseHamiltonian;
using mindquantum::sparse::TransposeCsrHdMatrix;

using mindquantum::projectq::Projectq;
using mindquantum::quest::Quest;

// Interface with python
PYBIND11_MODULE(mqbackend, m) {
  m.doc() = "MindQuantum c plugin";

  // matrix
  py::class_<Dim2Matrix<MT>, std::shared_ptr<Dim2Matrix<MT>>>(m, "dim2matrix")
      .def(py::init<>())
      .def(py::init<const VVT<CT<MT>> &>());

  // basic gate
  py::class_<BasicGate<MT>, std::shared_ptr<BasicGate<MT>>>(m, "basic_gate")
      .def(py::init<>())
      .def(py::init<bool, std::string, int64_t, Dim2Matrix<MT>>())
      .def(py::init<std::string, int64_t, py::object, py::object>())
      .def("PrintInfo", &BasicGate<MT>::PrintInfo)
      .def("apply_value", &BasicGate<MT>::ApplyValue)
      .def_readwrite("obj_qubits", &BasicGate<MT>::obj_qubits_)
      .def_readwrite("ctrl_qubits", &BasicGate<MT>::ctrl_qubits_)
      .def_readwrite("params", &BasicGate<MT>::params_)
      .def_readwrite("daggered", &BasicGate<MT>::daggered_)
      .def_readwrite("applied_value", &BasicGate<MT>::applied_value_)
      .def_readwrite("hermitian_prop", &BasicGate<MT>::hermitian_prop_);
  m.def("get_gate_by_name", &GetGateByName<MT>);
  m.def("get_measure_gate", &GetMeasureGate<MT>);
  // parameter resolver
  py::class_<ParameterResolver<MT>, std::shared_ptr<ParameterResolver<MT>>>(
      m, "parameter_resolver")
      .def(py::init<const MST<MT> &, const SS &, const SS &>())
      .def(py::init<>())
      .def(
          py::init<const VT<std::string> &, const VT<MT> &, const VT<bool> &>())
      .def_readonly("data", &ParameterResolver<MT>::data_)
      .def_readonly("no_grad_parameters",
                    &ParameterResolver<MT>::no_grad_parameters_)
      .def_readonly("requires_grad_parameters",
                    &ParameterResolver<MT>::requires_grad_parameters_);

  m.def("linear_combine", &LinearCombine<MT>);

  // pauli mat
  py::class_<PauliMat<MT>, std::shared_ptr<PauliMat<MT>>>(m, "pauli_mat")
      .def(py::init<>())
      .def(py::init<const PauliTerm<MT> &, Index>())
      .def_readonly("n_qubits", &PauliMat<MT>::n_qubits_)
      .def_readonly("dim", &PauliMat<MT>::dim_)
      .def_readwrite("coeff", &PauliMat<MT>::p_)
      .def("PrintInfo", &PauliMat<MT>::PrintInfo);

  m.def("get_pauli_mat", &GetPauliMat<MT>);

  // csr_hd_matrix
  py::class_<CsrHdMatrix<MT>, std::shared_ptr<CsrHdMatrix<MT>>>(m,
                                                                "csr_hd_matrix")
      .def(py::init<>())
      .def(py::init<Index, Index, py::array_t<Index>, py::array_t<Index>,
                    py::array_t<CT<MT>>>())
      .def("PrintInfo", &CsrHdMatrix<MT>::PrintInfo);
  m.def("csr_plus_csr", &Csr_Plus_Csr<MT>);
  m.def("transpose_csr_hd_matrix", &TransposeCsrHdMatrix<MT>);
  m.def("pauli_mat_to_csr_hd_matrix", &PauliMatToCsrHdMatrix<MT>);

  // hamiltonian
  py::class_<Hamiltonian<MT>, std::shared_ptr<Hamiltonian<MT>>>(m,
                                                                "hamiltonian")
      .def(py::init<>())
      .def(py::init<const VT<PauliTerm<MT>> &>())
      .def(py::init<const VT<PauliTerm<MT>> &, Index>())
      .def(py::init<std::shared_ptr<CsrHdMatrix<MT>>, Index>())
      .def_readwrite("how_to", &Hamiltonian<MT>::how_to_)
      .def_readwrite("n_qubits", &Hamiltonian<MT>::n_qubits_)
      .def_readwrite("ham", &Hamiltonian<MT>::ham_)
      .def_readwrite("ham_sparse_main", &Hamiltonian<MT>::ham_sparse_main_)
      .def_readwrite("ham_sparse_second", &Hamiltonian<MT>::ham_sparse_second_);
  m.def("sparse_hamiltonian", &SparseHamiltonian<MT>);

  // projectq simulator
  py::class_<Projectq<MT>, std::shared_ptr<Projectq<MT>>>(m, "projectq")
      .def(py::init<>())
      .def(py::init<unsigned, unsigned>())
      .def("reset", py::overload_cast<>(&Projectq<MT>::InitializeSimulator))
      .def("apply_measure", &Projectq<MT>::ApplyMeasure)
      .def("apply_gate",
           py::overload_cast<const BasicGate<MT> &>(&Projectq<MT>::ApplyGate))
      .def("apply_gate", py::overload_cast<const BasicGate<MT> &,
                                           const ParameterResolver<MT> &, bool>(
                             &Projectq<MT>::ApplyGate))
      .def("apply_circuit", py::overload_cast<const VT<BasicGate<MT>> &>(
                                &Projectq<MT>::ApplyCircuit))
      .def("apply_circuit", py::overload_cast<const VT<BasicGate<MT>> &,
                                              const ParameterResolver<MT> &>(
                                &Projectq<MT>::ApplyCircuit))
      .def("sampling", &Projectq<MT>::Sampling)
      .def("apply_hamiltonian", &Projectq<MT>::ApplyHamiltonian)
      .def("get_expectation", &Projectq<MT>::GetExpectation)
      .def("PrintInfo", &Projectq<MT>::PrintInfo)
      .def("run", &Projectq<MT>::run)
      .def("get_qs", &Projectq<MT>::cheat)
      .def("set_qs", &Projectq<MT>::SetState)
      .def("hermitian_measure_with_grad",
           py::overload_cast<
               const VT<Hamiltonian<MT>> &, const VT<BasicGate<MT>> &,
               const VT<BasicGate<MT>> &, const VVT<MT> &, const VT<MT> &,
               const VS &, const VS &, size_t, size_t>(
               &Projectq<MT>::HermitianMeasureWithGrad))
      .def("non_hermitian_measure_with_grad",
           py::overload_cast<
               const VT<Hamiltonian<MT>> &, const VT<Hamiltonian<MT>> &,
               const VT<BasicGate<MT>> &, const VT<BasicGate<MT>> &,
               const VT<BasicGate<MT>> &, const VT<BasicGate<MT>> &,
               const VVT<MT> &, const VT<MT> &, const VS &, const VS &, size_t,
               size_t>(&Projectq<MT>::NonHermitianMeasureWithGrad));

  // quest simulator
  py::class_<Quest<MT>, std::shared_ptr<Quest<MT>>>(m, "quest")
      .def(py::init<int>())
      .def(py::init<>())
      .def("reset", &Quest<MT>::InitializeSimulator)
      .def("PrintInfo", &Quest<MT>::PrintInfo)
      .def("get_qs", &Quest<MT>::GetVec)
      .def("apply_gate",
           py::overload_cast<const BasicGate<MT> &>(&Quest<MT>::ApplyGate))
      .def("apply_gate", py::overload_cast<const BasicGate<MT> &,
                                           const ParameterResolver<MT> &, bool>(
                             &Quest<MT>::ApplyGate))
      .def("apply_circuit", py::overload_cast<const VT<BasicGate<MT>> &>(
                                &Quest<MT>::ApplyCircuit))
      .def("apply_circuit", py::overload_cast<const VT<BasicGate<MT>> &,
                                              const ParameterResolver<MT> &>(
                                &Quest<MT>::ApplyCircuit))
      .def("apply_hamiltonian", &Quest<MT>::ApplyHamiltonian)
      .def("get_expectation", &Quest<MT>::GetExpectation)
      .def("hermitian_measure_with_grad",
           py::overload_cast<
               const VT<Hamiltonian<MT>> &, const VT<BasicGate<MT>> &,
               const VT<BasicGate<MT>> &, const VT<ParameterResolver<MT>> &,
               const VT<std::string> &, size_t, size_t>(
               &Quest<MT>::HermitianMeasureWithGrad));
}
}  // namespace mindquantum