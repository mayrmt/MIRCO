#include "mirco_warmstart.h"

template<typename T>
void MIRCO::Warmstart(T& x0, const T& xv0, const T& yv0, const T& xvf, const T& yvf, const T& pf)
{
  const size_t invalid_j = xvf.extent(0);
  size_t j;
  // [TODO] there should be a nicer way
  for (size_t i = 0; i < xv0.extent(0); ++i)
  {
    double curr_xv0 = xv0(i);
    double curr_yv0 = yv0(i);
    Kokkos::parallel_reduce("find same xv0 yv0",
        Kokkos::RangePolicy<MIRCO::device_space>(0,xvf.extent(0)),
        KOKKOS_LAMBDA (const size_t& i, size_t& k)
        {
          k = xvf(i) == curr_xv0 && yvf(i) == curr_yv0 ? i : invalid_j;
        },
        Kokkos::Min<size_t>(j));
    if ( j < invalid_j ) x0(i) = pf(j);
  }
}

template void MIRCO::Warmstart(MIRCO::view_dvec&, const MIRCO::view_dvec&, const MIRCO::view_dvec&, const MIRCO::view_dvec&, const MIRCO::view_dvec&, const MIRCO::view_dvec&);
template void MIRCO::Warmstart(MIRCO::subview_dvec&, const MIRCO::subview_dvec&, const MIRCO::subview_dvec&, const MIRCO::subview_dvec&, const MIRCO::subview_dvec&, const MIRCO::subview_dvec&);
