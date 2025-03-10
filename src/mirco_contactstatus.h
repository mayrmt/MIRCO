#pragma once

#include <cmath>
#include <vector>

#include "mirco_kokkos_types.hpp"

namespace MIRCO
{
  /**
   * @brief The aim of this function is to compute to nodes in contact in the current iteration
   *
   * @param xvf x-coordinates of the points in contact in the previous iteration.
   * @param yvf y-coordinates of the points in contact in the previous iteration.
   * @param pf Contact force at (xvf,yvf) predicted in the previous iteration.
   * @param y Solution containing force
   * @param xv0 x-coordinates of the points in contact in the previous iteration.
   * @param yv0 y-coordinates of the points in contact in the previous iteration.
   */
  void ComputeContactNodes(subview_dvec& xvf, subview_dvec& yvf, subview_dvec& pf,
      const subview_dvec& y, const subview_dvec& xv0, const subview_dvec& yv0);

  /**
   * @brief The aim of this function is to calulate the contact force and contact area for the
   * current iteration
   *
   * @param force0 Force vector; Each element contating contact force calculated at every iteraion
   * @param area0 Force vector; Each element contating contact area calculated at every iteraion
   * @param w_el Elastic correction
   * @param nf Number of nodes in contact in the previous iteration
   * @param pf Contact force at (xvf,yvf) predicted in the previous iteration.
   * @param k Iteration number
   * @param GridSize Grid size (length of each cell)
   * @param LateralLength Lateral side of the surface [micrometers]
   * @param ElasticComplianceCorrection Elastic compliance correction
   * @param PressureGreenFunFlag Flag to use Green function based on uniform pressure instead of
   * point force
   */
  void ComputeContactForceAndArea(std::vector<double> &force0, std::vector<double> &area0,
      double &w_el, const size_t nf, const subview_dvec& pf,
      const double GridSize, const double LateralLength,
      const double ElasticComplianceCorrection, const bool PressureGreenFunFlag);
}  // namespace MIRCO
