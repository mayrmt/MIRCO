#pragma once

#include "mirco_kokkos_types.hpp"

namespace MIRCO
{
  // important! adelus needs layout left
  // [TODO] do not gamble where the memory and the execution space is
  typedef Kokkos::View<double**, Kokkos::LayoutLeft, MIRCO::device_space> adelus_view_mat;

  void LinearSolve(const size_t N, adelus_view_mat& matrix);
}  // namespace MIRCO

