#include "mirco_nonlinearsolver.h"
#include "mirco_linearsolver.h"
#include <Kokkos_ScatterView.hpp>

typedef Kokkos::MinLoc<double,size_t>::value_type KK_minloc_type;

template<typename T>
void MIRCO::NonlinearSolve(const MIRCO::view_dmat& matrix,
    const T& b0, const T& y0,
    T& w, T& y, T& s0,
    double* mem_matrix, MIRCO::view_ivec& P)
{
  const double nnlstol = 1.0000e-08;
  const size_t maxiter = 10000;
  const double eps = 2.2204e-16;
  double alpha = 100000000;
  size_t iter = 0;
  bool init = false;
  size_t n0 = b0.extent(0);

  Kokkos::parallel_for("zero y", n0, KOKKOS_LAMBDA (const size_t& i) { y(i) = 0.0; });
  Kokkos::parallel_for("zero w", n0, KOKKOS_LAMBDA (const size_t& i) { w(i) = 0.0; });

  // Initialize active set
  size_t counter = 0;
  Kokkos::parallel_reduce("counter",
      Kokkos::RangePolicy<MIRCO::device_space>(0,y0.extent(0)),
      KOKKOS_LAMBDA (const size_t& i, size_t& k) { k += y0(i) >= nnlstol ? 1 : 0; },
      Kokkos::Sum<size_t>(counter));
  Kokkos::parallel_scan("positons (P)",
      Kokkos::RangePolicy<MIRCO::device_space>(0,y0.extent(0)),
      KOKKOS_LAMBDA (const size_t& i, size_t& k, const bool final)
      {
        const size_t n = y0(i) >= nnlstol ? 1 : 0;
        if (final) { if ( n == 1) P(k) = i; }
        k += n;
      });

  if (counter == 0)
  {
    Kokkos::parallel_for("w = -b0", n0, KOKKOS_LAMBDA (const size_t& i) { w(i) = -b0(i); });
    init = false;
  }
  else
  {
    init = true;
  }

  bool aux1 = true, aux2 = true;

  // New searching algorithm
  double minValue;
  int minPosition;

  KK_minloc_type w_minloc;
  KK_minloc_type alpha_minloc;

  while (aux1 == true)
  {
    Kokkos::parallel_reduce("min pos w",
        w.extent(0),
        KOKKOS_LAMBDA (const size_t& i, KK_minloc_type& minloc)
        {
          const double val = w(i);
          if( val < minloc.val )
          {
            minloc.val = val; minloc.loc = i;
          }
        },
        Kokkos::MinLoc<double,size_t>(w_minloc));

    minValue = w_minloc.val;
    minPosition = w_minloc.loc;

    if (((counter == n0) || (minValue > -nnlstol) || (iter >= maxiter)) && (init == false))
    {
      aux1 = false;
    }
    else
    {
      if (init == false) { P(counter) = minPosition; counter += 1; } else { init = false; }
    }

    size_t j = 0;
    aux2 = true;
    while (aux2 == true)
    {
      iter++;
      // important! adelus needs layout left
      MIRCO::adelus_view_mat M(mem_matrix, counter, counter + 1); // [TODO] force device space if available
      auto vector_x = Kokkos::subview(M, Kokkos::ALL(), counter);

      Kokkos::parallel_for("mask topology",
          Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{counter,counter}),
          KOKKOS_LAMBDA (const size_t& i, const size_t& j)
          {
            M(i,j) = matrix(P(i), P(j));
          });
      Kokkos::parallel_for("vector_b",
          counter,
          KOKKOS_LAMBDA (const size_t& i) { vector_x(i) = b0(P(i)); });

      MIRCO::LinearSolve(counter, M);

      Kokkos::parallel_for("s0(P(i)) = vector_x(i)", counter, KOKKOS_LAMBDA (const size_t& i) { s0(P(i)) = vector_x(i); });

      bool allBigger = true;
      Kokkos::parallel_reduce("allBigger",
          counter, // [TODO] why do you operate on indirect s0 instead of vector_x ?
          KOKKOS_LAMBDA (const size_t& i, bool& r)
          {
            const bool t = s0(P(i)) < nnlstol;
            r = r && !t;
          },
          Kokkos::LAnd<bool>(allBigger));

      if (allBigger == true)
      {
        aux2 = false;
        Kokkos::parallel_for("y <- s0", counter, KOKKOS_LAMBDA (const size_t& i) { y(P(i)) = s0(P(i)); }); // [TODO] why P indirection?
        Kokkos::parallel_for("w = -b0", matrix.extent(0), KOKKOS_LAMBDA (const size_t& i) { w(i) = -b0(i); });
        {
          auto w_scatter = Kokkos::Experimental::create_scatter_view(w);
          Kokkos::parallel_for("w <- matrix * y | filtered P",
              Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{matrix.extent(0),counter}),
              KOKKOS_LAMBDA (const size_t i, const size_t j)
              {
                auto access = w_scatter.access();
                // [TODO] there should be a much nicer way to achieve this, but the indirection is fucking us here
                access(i) += (matrix(i, P(j)) * y(P(j)));
              });
          Kokkos::Experimental::contribute(w, w_scatter);
        }
      }
      else
      {
        // Searching for minimum value with index position
        Kokkos::parallel_reduce("min pos alpha",
            counter,
            KOKKOS_LAMBDA (const size_t& i, KK_minloc_type& minloc)
            {
              if ( s0(P(i)) < nnlstol ) // [TODO] why indirection?
              {
                const double alphai = y(P(i)) / (eps + y(P(i)) - s0(P(i)));
                if( alphai < minloc.val )
                {
                  minloc.val = alphai; minloc.loc = i;
                }
              }
            },
            Kokkos::MinLoc<double,size_t>(alpha_minloc));

        alpha = alpha_minloc.val;
        j = alpha_minloc.loc;

        // TODO: WHAT BELONGS TO THIS LOOP????????????????????? // [TODO] don't know, now it is gone
        Kokkos::parallel_for("y <- y + alpha * (s0 - y)",
            counter, // [TODO] why P indirection?
            KOKKOS_LAMBDA (const size_t& i)
            {
              y(P(i)) = y(P(i)) + alpha * (s0(P(i)) - y(P(i)));
            });

        if ( j > 0 )
        {
          MIRCO::view_dvec temp(mem_matrix, counter); // [TODO] force device space if available
          Kokkos::deep_copy(temp, Kokkos::subview(P, Kokkos::pair<size_t,size_t>(0,counter))); // use part of matrix for buffer in order to possibly avoid race conditions
          s0(P(j)) = 0;
          Kokkos::parallel_scan("remove P(j)",
              Kokkos::RangePolicy<MIRCO::device_space>(0,counter),
              KOKKOS_LAMBDA (const size_t& i, size_t& k, const bool final)
              {
                const size_t n = i == j ? 0 : 1;
                if (final) { if ( n == 1 ) P(k) = temp(i); }
                k += n;
              });
          counter -= 1;
        }
      }
    }
  }
}

template void MIRCO::NonlinearSolve(const MIRCO::view_dmat&,
    const MIRCO::subview_dvec&, const MIRCO::subview_dvec&,
    MIRCO::subview_dvec&, MIRCO::subview_dvec&, MIRCO::subview_dvec&,
    double*, MIRCO::view_ivec&);
template void MIRCO::NonlinearSolve(const MIRCO::view_dmat&,
    const MIRCO::view_dvec&, const MIRCO::view_dvec&,
    MIRCO::view_dvec&, MIRCO::view_dvec&, MIRCO::view_dvec&,
    double*, MIRCO::view_ivec&);
