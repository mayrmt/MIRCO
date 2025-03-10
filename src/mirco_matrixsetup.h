#pragma once

#include "mirco_kokkos_types.hpp"

namespace MIRCO
{
  void SetupMatrix(view_dmat& A,
      const subview_dvec& xv0, const subview_dvec& yv0,
      const double GridSize, const double CompositeYoungs,
      const double CompositePoissonsRatio, const size_t systemsize,
      const bool PressureGreenFunFlag);
}  // namespace MIRCO
