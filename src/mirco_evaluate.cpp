#include "mirco_evaluate.h"

#include <omp.h>
#include <unistd.h>

#include <Teuchos_Assert.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "mirco_contactpredictors.h"
#include "mirco_contactstatus.h"
#include "mirco_matrixsetup.h"
#include "mirco_nonlinearsolver.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Bitset.hpp>
#include "mirco_kokkos_types.hpp"

void MIRCO::Evaluate(double& pressure, double Delta, double LateralLength, double GridSize,
    double Tolerance, int MaxIteration, double CompositeYoungs, double CompositePoissonsRatio,
    bool WarmStartingFlag, double ElasticComplianceCorrection,
    Teuchos::SerialDenseMatrix<int, double>& topology, double zmax, std::vector<double>& meshgrid,
    bool PressureGreenFunFlag)
{
  // Initialise the area vector and force vector. Each element containing the
  // area and force calculated at every iteration.
  std::vector<double> area0;
  std::vector<double> force0;
  double w_el = 0.0;

  // Initialise number of iteration, k, and initial number of predicted contact
  // nodes, n0.
  size_t k = 0, n0 = 0;

  // get max memory size
  const int ni = topology.numCols();
  const int nj = topology.numCols();

  // copy topology to kokkos view
  MIRCO::view_dmat topology_view("topology", ni, nj);
  {
    // [TODO] could be done with unmanaged view on the raw address of topology (if contiguous) (dangerous due to kokkos padding)
    MIRCO::view_dmat::HostMirror topology_view_h("topology host", ni, nj);
    for (int i = 0; i < ni; ++i){ for (int j = 0; j < nj; ++j){
      try { topology_view_h(i,j) = topology(i,j); } catch (const std::exception &e) { } // who uses structures which need try catch in HPC?
    } }
    Kokkos::deep_copy(topology_view, topology_view_h);
  }
  MIRCO::view_bits topology_mask(ni * nj);

  // copy meshgrid to kokkos view
  MIRCO::view_dvec meshgrid_view("meshgrid", ni);
  {
    // [TODO] could be done with unmanaged view on the raw address of meshgrid (if contiguous)
    MIRCO::view_dvec::HostMirror meshgrid_view_h("meshgrid host", ni);
    for (int i = 0; i < ni; ++i){
      try { meshgrid_view_h(i) = meshgrid[i]; } catch (const std::exception &e) { } // who uses structures which need try catch in HPC?
    }
    Kokkos::deep_copy(meshgrid_view, meshgrid_view_h);
  }

  // Coordinates of the points predicted to be in contact.
  MIRCO::view_dvec xv0_fview("full xv0", ni * nj);
  MIRCO::view_dvec yv0_fview("full yv0", ni * nj);
  // Coordinates of the points in contact in the previous iteration.
  MIRCO::view_dvec xvf_fview("full xvf", ni * nj);
  MIRCO::view_dvec yvf_fview("full yvf", ni * nj);
  // Indentation value of the half space at the predicted points of contact.
  MIRCO::view_dvec b0_fview("full b0", ni * nj);
  // Contact force at (xvf,yvf) predicted in the previous iteration.
  MIRCO::view_dvec pf_fview("full bf", ni * nj);

  // x0 --> contact forces at (xvf,yvf) predicted in the previous iteration but
  // are a part of currect predicted contact set. x0 is calculated in the
  // Warmstart function to be used in the NNLS to accelerate the simulation.
  MIRCO::view_dvec x0_fview("full x0", ni * nj);

  // Defined as (u - u(bar)) in (Bemporad & Paggi, 2015)
  // Gap between the point on the topology and the half space
  MIRCO::view_dvec w_fview("full w", ni * nj);

  // The number of nodes in contact in the previous iteration.
  size_t nf = 0;
  MIRCO::subview_dvec xvf = Kokkos::subview(xvf_fview, Kokkos::pair<size_t,size_t>(0,nf)); // [TODO] possibly nf instead of n0?
  MIRCO::subview_dvec yvf = Kokkos::subview(yvf_fview, Kokkos::pair<size_t,size_t>(0,nf)); // [TODO] possibly nf instead of n0?
  MIRCO::subview_dvec pf = Kokkos::subview(yvf_fview, Kokkos::pair<size_t,size_t>(0,nf)); // [TODO] possibly nf instead of n0?

  // The influence coefficient matrix (Discrete version of Green Function)
  double* mem_A = (double*) Kokkos::kokkos_malloc<>("memory A", ni*ni * nj*nj * sizeof(double));
  // Solution containing force
  MIRCO::view_dvec y_fview("full y", ni * nj);

  // solver memory
  double* mem_matrix = (double*) Kokkos::kokkos_malloc<>("memory solverMatrix", (ni*ni * nj*nj + ni * nj) * sizeof(double));
  MIRCO::view_ivec mem_filter("memory index filter", ni * nj);
  MIRCO::view_dvec s0_fview("full s0", ni * nj);

  // Initialise the error in force
  double ErrorForce = std::numeric_limits<double>::max();
  while (ErrorForce > Tolerance && k < (size_t)MaxIteration)
  {
    // First predictor for contact set

    MIRCO::ContactSetPredictorSize(n0, zmax, Delta, w_el, topology_view, topology_mask);
    MIRCO::subview_dvec xv0 = Kokkos::subview(xv0_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::subview_dvec yv0 = Kokkos::subview(yv0_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::subview_dvec b0 = Kokkos::subview(b0_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::subview_dvec x0 = Kokkos::subview(x0_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::subview_dvec w = Kokkos::subview(w_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::subview_dvec y = Kokkos::subview(y_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::subview_dvec s0 = Kokkos::subview(s0_fview, Kokkos::pair<size_t,size_t>(0,n0));
    MIRCO::ContactSetPredictor(xv0, yv0, b0, zmax, Delta, w_el, meshgrid_view, topology_view, topology_mask);

    MIRCO::view_dmat A(mem_A, n0, n0);

    // Construction of the Matrix A
    MIRCO::SetupMatrix(A, xv0, yv0, GridSize, CompositeYoungs, CompositePoissonsRatio, n0, PressureGreenFunFlag);

    // Second predictor for contact set
    // @{
    MIRCO::InitialGuessPredictor(WarmStartingFlag, k, xv0, yv0, pf, x0, xvf, yvf);
    // }

    // use Nonlinear solver --> Non-Negative Least Squares (NNLS) as in
    // (Bemporad & Paggi, 2015)
    MIRCO::NonlinearSolve(A, b0, x0, w, y, s0, mem_matrix, mem_filter);

    // Compute number of contact node
    // @{
    Kokkos::parallel_reduce("nf", n0, KOKKOS_LAMBDA (const size_t& i, size_t& k) { k += y(i) != 0 ? 1 : 0; }, Kokkos::Sum<size_t>(nf));
    xvf = Kokkos::subview(xvf_fview, std::pair<size_t,size_t>(0,nf)); // [TODO] possibly nf instead of n0?
    yvf = Kokkos::subview(yvf_fview, std::pair<size_t,size_t>(0,nf)); // [TODO] possibly nf instead of n0?
    pf = Kokkos::subview(yvf_fview, std::pair<size_t,size_t>(0,nf)); // [TODO] possibly nf instead of n0?
    MIRCO::ComputeContactNodes(xvf, yvf, pf, y, xv0, yv0);
    // }

    // Compute contact force and contact area
    // @{
    MIRCO::ComputeContactForceAndArea(force0, area0, w_el, nf, pf, GridSize, LateralLength, ElasticComplianceCorrection, PressureGreenFunFlag);
    // }

    // Compute error due to nonlinear correction
    // @{
    if (k > 0) ErrorForce = abs(force0[k] - force0[k - 1]) / force0[k];
    k += 1;
    // }
  }

  Kokkos::kokkos_free(mem_A);
  Kokkos::kokkos_free(mem_matrix);

  TEUCHOS_TEST_FOR_EXCEPTION(ErrorForce > Tolerance, std::out_of_range, "The solution did not converge in the maximum number of iternations defined");
  // @{

  // Calculate the final force value at the end of the iteration.
  const double force = force0[k - 1];

  // Mean pressure
  double sigmaz = force / pow(LateralLength, 2);
  pressure = sigmaz;
}
