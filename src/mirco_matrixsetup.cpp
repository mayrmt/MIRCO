#include "mirco_matrixsetup.h"

#include <math.h>

void MIRCO::SetupMatrix(MIRCO::view_dmat& A,
    const MIRCO::subview_dvec& xv0, const MIRCO::subview_dvec& yv0,
    const double GridSize, const double CompositeYoungs,
    const double CompositePoissonsRatio, const size_t systemsize,
    const bool PressureGreenFunFlag)
{
  const double pi = M_PI;
  if (PressureGreenFunFlag)
  {
    const double scale = (1 - pow(CompositePoissonsRatio, 2)) / (pi * CompositeYoungs);
    Kokkos::parallel_for("fill A",
        Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{systemsize,systemsize}),
        KOKKOS_LAMBDA (const size_t& i, const size_t& j)
        {
          double k = xv0(i) - xv0(j) + GridSize / 2;
          double l = xv0(i) - xv0(j) - GridSize / 2;
          double m = yv0(i) - yv0(j) + GridSize / 2;
          double n = yv0(i) - yv0(j) - GridSize / 2;
          A(i,j) = (k * log((sqrt(k * k + m * m) + m) / (sqrt(k * k + n * n) + n)) +
                    l * log((sqrt(l * l + n * n) + n) / (sqrt(l * l + m * m) + m)) +
                    m * log((sqrt(m * m + k * k) + k) / (sqrt(m * m + l * l) + l)) +
                    n * log((sqrt(n * n + l * l) + l) / (sqrt(n * n + k * k) + k))) * scale;
        });
  }
  else
  {
    const double raggio = GridSize / 2;
    const double C = 1 / (CompositeYoungs * pi * raggio);

    Kokkos::parallel_for("fill A",
        Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{systemsize,systemsize}),
        KOKKOS_LAMBDA (const size_t& i, const size_t& j)
        {
          const double dx = xv0(j) - xv0(i);
          const double dy = yv0(j) - yv0(i);
          const double r2 = dx * dx + dy * dy;
          const double factor = i != j ? asin(raggio / sqrt(r2)) : 1; // [TODO] possibly move asin and div sqrt to own var and mask it with `i != j` to enforce full SIMD (careful div_by_0)?
          A(i,j) = C * factor;
        });
  }
}
