#include "mirco_contactstatus.h"

#include <cmath>

void MIRCO::ComputeContactNodes(MIRCO::subview_dvec& xvf, MIRCO::subview_dvec& yvf, MIRCO::subview_dvec& pf,
    const MIRCO::subview_dvec& y, const MIRCO::subview_dvec& xv0, const MIRCO::subview_dvec& yv0)
{
  Kokkos::parallel_scan("pxyvf <- pxyv0",
      Kokkos::RangePolicy<MIRCO::device_space>(0,y.extent(0)),
      KOKKOS_LAMBDA (const size_t& i, size_t& k, const bool final)
      {
        const size_t n = y(i) != 0 ? 1 : 0;
        if (final)
        {
          if ( n == 1)
          {
            xvf(k) = xv0(i);
            yvf(k) = yv0(i);
            pf(k) = y(i);
          }
        }
        k += n;
      });
}

void MIRCO::ComputeContactForceAndArea(std::vector<double> &force0, std::vector<double> &area0,
    double &w_el, const size_t nf, const MIRCO::subview_dvec& pf,
    const double GridSize, const double LateralLength,
    const double ElasticComplianceCorrection, const bool PressureGreenFunFlag)
{
  double force_sum = 0;
  const double GridSize2 = GridSize * GridSize;
  Kokkos::parallel_reduce("force",
      nf,
      KOKKOS_LAMBDA (const size_t& i, double& sum)
      {
        if (PressureGreenFunFlag)
        {
          sum += pf(i) * GridSize2; // [TODO] possibly make two different kernels
        }
        else
        {
          sum += pf(i);
        }
      },
      Kokkos::Sum<double>(force_sum));
  force0.push_back(force_sum);
  area0.push_back((double)nf * (GridSize2 / pow(LateralLength, 2)));
  w_el = force_sum / ElasticComplianceCorrection;
}
