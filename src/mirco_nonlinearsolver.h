#pragma once

#include "mirco_kokkos_types.hpp"

namespace MIRCO
{
  template<typename T>
  void NonlinearSolve(const view_dmat& matrix,
      const T& b0, const T& y0,
      T& w, T& y, T& s0,
      double* mem_matrix, view_ivec& P);
}  // namespace MIRCO
