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

#ifndef MINDQUANTUM_GATE_basic_gate_H_
#define MINDQUANTUM_GATE_basic_gate_H_
#include <functional>
#include <string>

#include "matrix/two_dim_matrix.h"
#include "pr/parameter_resolver.h"
#include "src/utils.h"

namespace mindquantum {
namespace py = pybind11;

template <typename T>
inline VVT<CT<T>> CastArray(const py::object &fun, T theta) {
  py::array_t<CT<T>> a = fun(theta);
  py::buffer_info buf = a.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Gate matrix must be two dimension!");
  }
  if (buf.shape[0] != buf.shape[1]) {
    throw std::runtime_error("Gate matrix need a square matrix!");
  }
  CTP<T> ptr = static_cast<CTP<T>>(buf.ptr);

  VVT<CT<T>> m;
  for (size_t i = 0; i < buf.shape[0]; i++) {
    m.push_back({});
    for (size_t j = 0; j < buf.shape[1]; j++) {
      m[i].push_back(ptr[i * buf.shape[1] + j]);
    }
  }
  return m;
}
template <typename T>
struct BasicGate {
  bool parameterized_ = false;
  std::string name_;
  VT<Index> obj_qubits_;
  VT<Index> ctrl_qubits_;
  ParameterResolver<T> params_;
  int64_t hermitian_prop_ = SELFHERMITIAN;
  bool daggered_ = false;
  T applied_value_ = 0;
  bool is_measure_ = false;
  Dim2Matrix<T> base_matrix_;
  std::function<Dim2Matrix<T>(T)> param_matrix_;
  std::function<Dim2Matrix<T>(T)> param_diff_matrix_;
  // Dim2Matrix<T> (*param_matrix_)(T para);
  // Dim2Matrix<T> (*param_diff_matrix_)(T para);
  void PrintInfo() {
    if (!daggered_) {
      std::cout << "Gate name: " << name_ << std::endl;
    } else {
      std::cout << "Gate name: " << name_ << " (daggered version)" << std::endl;
    }
    std::cout << "Parameterized: " << parameterized_ << std::endl;
    if (!parameterized_) {
      base_matrix_.PrintInfo();
    }
    if (obj_qubits_.size() != 0) {
      std::cout << "Obj qubits: ";
      for (auto o : obj_qubits_) {
        std::cout << o << " ";
      }
      std::cout << std::endl;
    }
    if (ctrl_qubits_.size() != 0) {
      std::cout << "Control qubits: ";
      for (auto o : ctrl_qubits_) {
        std::cout << o << " ";
      }
      std::cout << std::endl;
    }
  }
  void ApplyValue(T theta) {
    if (parameterized_) {
      parameterized_ = false;
      applied_value_ = theta;
      base_matrix_ = param_matrix_(theta);
    }
  }
  BasicGate() {}
  BasicGate(bool parameterized, std::string name, int64_t hermitian_prop,
            Dim2Matrix<T> base_matrix)
      : parameterized_(parameterized),
        name_(name),
        hermitian_prop_(hermitian_prop),
        base_matrix_(base_matrix) {}
  BasicGate(bool parameterized, std::string name, int64_t hermitian_prop,
            Dim2Matrix<T> (*param_matrix)(T para),
            Dim2Matrix<T> (*param_diff_matrix)(T para))
      : parameterized_(parameterized),
        name_(name),
        hermitian_prop_(hermitian_prop),
        param_matrix_(param_matrix),
        param_diff_matrix_(param_diff_matrix) {}
  BasicGate(std::string name, int64_t hermitian_prop, py::object matrix_fun,
            py::object diff_matrix_fun)
      : parameterized_(true), name_(name), hermitian_prop_(hermitian_prop) {
    param_matrix_ = [matrix_fun](T theta) {
      auto matrix = CastArray<T>(matrix_fun, theta);
      Dim2Matrix<T> res = Dim2Matrix<T>(matrix);
      return res;
    };
    param_diff_matrix_ = [diff_matrix_fun](T theta) {
      auto matirx = CastArray<T>(diff_matrix_fun, theta);
      Dim2Matrix<T> res = Dim2Matrix<T>(matirx);
      return res;
    };
  }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_basic_gate_H_
