#include "mirco_linearsolver.h"

#include <Adelus.hpp>
#include <mpi.h>

void MIRCO::LinearSolve(const size_t N, MIRCO::adelus_view_mat& matrix)
{
  // [TODO] do not gamble where the memory and the execution space is
  Adelus::AdelusHandle<typename adelus_view_mat::value_type,
                       typename adelus_view_mat::execution_space,
                       typename adelus_view_mat::memory_space>ahandle(0, MPI_COMM_WORLD, N, 1, 1); // [TODO] tune for threads/ranks etc.

  double secs; // seconds the solver took
  // [TODO] add preconditioner? (former version had 'factorWithEquilibration')
  Adelus::FactorSolve(ahandle, matrix, &secs);
}
