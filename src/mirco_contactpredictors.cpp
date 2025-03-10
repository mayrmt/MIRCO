#include "mirco_contactpredictors.h"

#include "mirco_warmstart.h"

void MIRCO::ContactSetPredictorSize(size_t &n0,
    const double zmax, const double Delta, const double w_el,
    const MIRCO::view_dmat& topology_view, MIRCO::view_bits& topology_mask) // [TODO] bitset shouldn't be necessary, just recompute
{
  const double value = zmax - Delta - w_el;
  const size_t ni = topology_view.extent(0);
  const size_t nj = topology_view.extent(1);
  Kokkos::parallel_for("mask topology",
      Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{ni,nj}),
      KOKKOS_LAMBDA (const size_t& i, const size_t& j)
      {
        const size_t k = i * nj + j;
        if ( topology_view(i,j) >= value )
        {
          topology_mask.set(k);
        }
        else
        {
          topology_mask.reset(k);
        }
      });
  n0 = topology_mask.count();
}

void MIRCO::ContactSetPredictor(MIRCO::subview_dvec& xv0, MIRCO::subview_dvec& yv0, MIRCO::subview_dvec& b0,
    const double zmax, const double Delta, const double w_el,
    const MIRCO::view_dvec& meshgrid_view, const MIRCO::view_dmat& topology_view,
    const MIRCO::view_bits& topology_mask)
{
  //const size_t ni = topology_view.extent(0);
  const size_t nj = topology_view.extent(1);

  Kokkos::parallel_scan("pack topology",
      Kokkos::RangePolicy<MIRCO::device_space>(0,topology_mask.size()),
      KOKKOS_LAMBDA (const size_t& k, size_t& I, const bool final)
      {
        const bool t = topology_mask.test(k);
        if (final)
        {
          if ( t )
          {
            const size_t j = k % nj;
            const size_t i = k / nj;
            xv0(I) = meshgrid_view(j);
            yv0(I) = meshgrid_view(i);
            b0(I) = Delta + w_el - (zmax - topology_view(i,j));
          }
        }
        I += t ? 1 : 0;
      });
}

void MIRCO::InitialGuessPredictor(const bool WarmStartingFlag, const int k,
    const MIRCO::subview_dvec& xv0, const MIRCO::subview_dvec& yv0, const MIRCO::subview_dvec& pf,
    MIRCO::subview_dvec& x0, const MIRCO::subview_dvec& xvf, const MIRCO::subview_dvec& yvf)
{
  if (WarmStartingFlag == 1 && k > 0)
  {
    MIRCO::Warmstart(x0, xv0, yv0, xvf, yvf, pf);
  }
}
