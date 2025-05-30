#ifndef SRC_EVALUATE_H_
#define SRC_EVALUATE_H_

#include <Teuchos_SerialDenseMatrix.hpp>
#include <string>

namespace MIRCO
{
  /**
   * @brief Relate the far-field displacement with pressure
   *
   * @param pressure Pressure
   * @param Delta Far-field displacement (Gap)
   * @param LateralLength Lateral side of the surface [micrometers]
   * @param GridSize Grid size (length of each cell)
   * @param Tolerance Tolerance for the convergence of force
   * @param MaxIteration Maximum number of iterations for the force to converge
   * @param CompositeYoungs Composite Young's modulus
   * @param WarmStartingFlag Warm-Starter Flag
   * @param ElasticComplianceCorrection Elastic compliance correction
   * @param topology Topology matrix containing heights
   * @param zmax Maximum height
   * @param meshgrid Meshgrid vector
   * @param PressureGreenFunFlag Flag to use Green function based on uniform pressure instead of
   * point force
   */
  void Evaluate(double& pressure, double Delta, double LateralLength, double GridSize,
      double Tolerance, int MaxIteration, double CompositeYoungs, bool WarmStartingFlag,
      double ElasticComplianceCorrection, Teuchos::SerialDenseMatrix<int, double>& topology,
      double zmax, std::vector<double>& meshgrid, bool PressureGreenFunFlag);
}  // namespace MIRCO


#endif  // SRC_EVALUATE_H_
