#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Bitset.hpp>

namespace MIRCO
{

typedef Kokkos::View<size_t*> view_ivec;
typedef Kokkos::View<double*> view_dvec;
typedef Kokkos::View<double**> view_dmat;

typedef typename Kokkos::Subview<view_ivec, Kokkos::pair<size_t, size_t>> subview_ivec;
typedef typename Kokkos::Subview<view_dvec, Kokkos::pair<size_t, size_t>> subview_dvec;

// [TODO] do not gamble where the memory and the execution space is
typedef typename view_dmat::device_type::execution_space device_space;
typedef typename view_dmat::device_type::memory_space memory_space;

typedef Kokkos::Bitset<device_space> view_bits;

}  // namespace MIRCO
