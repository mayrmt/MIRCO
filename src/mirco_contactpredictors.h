#pragma once

#include "mirco_kokkos_types.hpp"

namespace MIRCO
{
  void ContactSetPredictorSize(size_t& n0,
      const double zmax, const double Delta, const double w_el,
      const view_dmat& topology_view, view_bits& topology_mask);

  /**
   * @brief The aim of this function is to determine all the points, for which the gap is bigger
   * than the displacement of the rigid indenter, cannot be in contact and thus are not checked in
   * nonlinear solve
   *
   * @param n0 Number of nodes predicted to be in contact
   * @param xv0 x-coordinates of the points in contact in the previous iteration.
   * @param yv0 y-coordinates of the points in contact in the previous iteration.
   * @param b0 Indentation value of the half space at the predicted points of contact.
   * @param zmax Maximum height of the topology
   * @param Delta Far-field displacement (Gap)
   * @param w_el Elastic correction
   * @param meshgrid Meshgrid
   * @param topology Topology matrix containing heights
   */
  void ContactSetPredictor(subview_dvec& xv0, subview_dvec& yv0, subview_dvec& b0,
      const double zmax, const double Delta, const double w_el,
      const view_dvec& meshgrid_view, const view_dmat& topology_view,
      const view_bits& topology_mask);

  /**
   * @brief The aim of this function is to guess the set of nodes in contact among the nodes
   * predicted in the ContactSetPredictor function. It uses Warmstart to make an initial guess of
   * the nodes incontact in this iteration based on the previous iteration.
   *
   * @param WarmStartingFlag Warm-Starter flag
   * @param k Iteration number
   * @param n0 Number of nodes predicted to be in contact
   * @param xv0 x-coordinates of the points in contact in the previous iteration.
   * @param yv0 y-coordinates of the points in contact in the previous iteration.
   * @param pf Contact force at (xvf,yvf) predicted in the previous iteration.
   * @param x0 contact forces at (xvf,yvf) predicted in the previous iteration but are a part of
   * currect predicted contact set.
   * @param b0 Indentation value of the half space at the predicted points of contact.
   * @param xvf x-coordinates of the points in contact in the previous iteration.
   * @param yvf y-coordinates of the points in contact in the previous iteration.
   */
  void InitialGuessPredictor(const bool WarmStartingFlag, const int k,
      const subview_dvec& xv0, const subview_dvec& yv0, const subview_dvec& pf,
      subview_dvec& x0, const subview_dvec& xvf, const subview_dvec& yvf);
}  // namespace MIRCO
