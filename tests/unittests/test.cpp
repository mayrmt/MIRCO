#include <gtest/gtest.h>
#include <stdlib.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <iostream>
/*
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialSymDenseMatrix.hpp>
*/
#include <vector>

#include "../../src/mirco_filesystem_utils.h"
#include "../../src/mirco_linearsolver.h"
#include "../../src/mirco_nonlinearsolver.h"
#include "../../src/mirco_topology.h"
#include "../../src/mirco_warmstart.h"
#include "nonlinear_solver_test.h"

TEST(linearsolver, solves)
{
  size_t systemsize = 2;

  // Build the matrix
  MIRCO::adelus_view_mat topology("topology", systemsize, systemsize + 1);
  // [TODO] host/device separation
  for (size_t i = 0; i < systemsize; i++)
  {
    topology(i, i) = 2;
    for (size_t j = 0; j < i; j++)
    {
      topology(i, j) = 1;
      topology(j, i) = 1;
    }
  }

  // Build the vector(s)
  auto vector_x = Kokkos::subview(topology, Kokkos::ALL(), systemsize);

  // Build right hand side
  for (size_t i = 0; i < systemsize; i++)
  {
    vector_x(i) = 1;
  }

  // Call linear solver
  MIRCO::LinearSolve(systemsize, topology);

  EXPECT_NEAR(vector_x(0), 0.333333333333333, 1e-06);
  EXPECT_NEAR(vector_x(1), 0.333333333333333, 1e-06);
}

TEST_F(NonlinearSolverTest, primalvariable)
{
  const size_t ni = matrix_.numCols();
  const size_t nj = matrix_.numCols();

  // [TODO] host/device separation
  double* mem_matrix = (double*) Kokkos::kokkos_malloc<>("memory solverMatrix", (ni * (nj + 1)) * sizeof(double));
  MIRCO::view_ivec mem_filter("index filter", ni);
  MIRCO::view_dvec s0("s0", ni);

  MIRCO::view_dmat matrix("matrix", ni, nj);
  {
    auto matrix_h = Kokkos::create_mirror(matrix);
    for (size_t i = 0; i < ni; ++i){ for (size_t j = 0; j < nj; ++j){ matrix_h(i,j) = matrix_(i,j); } }
    Kokkos::deep_copy(matrix, matrix_h);
  }

  MIRCO::view_dvec b_vector("b_vector", ni);
  {
    auto b_vector_h = Kokkos::create_mirror(b_vector);
    for (size_t i = 0; i < ni; ++i){ b_vector_h(i) = b_vector_[i]; }
    Kokkos::deep_copy(b_vector, b_vector_h);
  }

  MIRCO::view_dvec x_vector("x_vector", ni);
  {
    auto x_vector_h = Kokkos::create_mirror(x_vector);
    for (size_t i = 0; i < ni; ++i){ x_vector_h(i) = x_vector_[i]; }
    Kokkos::deep_copy(x_vector, x_vector_h);
  }

  MIRCO::view_dvec w("w", ni);
  MIRCO::view_dvec y("y", ni);

  MIRCO::NonlinearSolve(matrix, b_vector, x_vector, w, y, s0, mem_matrix, mem_filter);

  auto y_h = Kokkos::create_mirror(y);
  Kokkos::deep_copy(y_h, y);

  EXPECT_NEAR(y_h(0), 163213.374921086, 1e-06);
  EXPECT_NEAR(y_h(1), 43877.9231473546, 1e-06);
  EXPECT_NEAR(y_h(2), 163702.923578063, 1e-06);
  EXPECT_NEAR(y_h(3), 55159.5440853170, 1e-06);
  EXPECT_NEAR(y_h(4), 10542.1713862417, 1e-06);
  EXPECT_NEAR(y_h(5), 53809.0897795325, 1e-06);
  EXPECT_NEAR(y_h(6), 148773.412150208, 1e-06);
  EXPECT_NEAR(y_h(7), 83711.5732276221, 1e-06);
  EXPECT_NEAR(y_h(8), 149262.960807186, 1e-06);
}

TEST_F(NonlinearSolverTest, dualvariable)
{
  const size_t ni = matrix_.numCols();
  const size_t nj = matrix_.numCols();

  // [TODO] host/device separation
  double* mem_matrix = (double*) Kokkos::kokkos_malloc<>("memory solverMatrix", (ni * (nj + 1)) * sizeof(double));
  MIRCO::view_ivec mem_filter("index filter", ni);
  MIRCO::view_dvec s0("s0", ni);

  MIRCO::view_dmat matrix("matrix", ni, nj);
  {
    auto matrix_h = Kokkos::create_mirror(matrix);
    for (size_t i = 0; i < ni; ++i){ for (size_t j = 0; j < nj; ++j){ matrix_h(i,j) = matrix_(i,j); } }
    Kokkos::deep_copy(matrix, matrix_h);
  }

  MIRCO::view_dvec b_vector("b_vector", ni);
  {
    auto b_vector_h = Kokkos::create_mirror(b_vector);
    for (size_t i = 0; i < ni; ++i){ b_vector_h(i) = b_vector_[i]; }
    Kokkos::deep_copy(b_vector, b_vector_h);
  }

  MIRCO::view_dvec x_vector("x_vector", ni);
  {
    auto x_vector_h = Kokkos::create_mirror(x_vector);
    for (size_t i = 0; i < ni; ++i){ x_vector_h(i) = x_vector_[i]; }
    Kokkos::deep_copy(x_vector, x_vector_h);
  }

  MIRCO::view_dvec w("w", ni);
  MIRCO::view_dvec y("y", ni);

  MIRCO::NonlinearSolve(matrix, b_vector, x_vector, w, y, s0, mem_matrix, mem_filter);

  auto w_h = Kokkos::create_mirror(w);
  Kokkos::deep_copy(w_h, w);

  EXPECT_NEAR(w_h(0), 0, 1e-06);
  EXPECT_NEAR(w_h(1), 0, 1e-06);
  EXPECT_NEAR(w_h(2), 0, 1e-06);
  EXPECT_NEAR(w_h(3), 0, 1e-06);
  EXPECT_NEAR(w_h(4), 0, 1e-06);
  EXPECT_NEAR(w_h(5), 0, 1e-06);
  EXPECT_NEAR(w_h(6), 0, 1e-06);
  EXPECT_NEAR(w_h(7), 0, 1e-06);
  EXPECT_NEAR(w_h(8), 0, 1e-06);
}

TEST(FilesystemUtils, createrelativepath)
{
  std::string targetfilename = "input.dat";
  std::string sourcefilename = "../inputfiles/sourceinput.json";
  UTILS::ChangeRelativePath(targetfilename, sourcefilename);
  EXPECT_EQ(targetfilename, "../inputfiles/input.dat");
}

TEST(FilesystemUtils, keepabsolutpath)
{
  std::string targetfilename = "/root_dir/home/user/Input/input.dat";
  std::string sourcefilename = "../inputfiles/sourceinput.json";
  UTILS::ChangeRelativePath(targetfilename, sourcefilename);
  EXPECT_EQ(targetfilename, "/root_dir/home/user/Input/input.dat");
}

TEST(readtopology, RMG)
{
  int Resolution = 2;
  float HurstExponent = 0.1;
  bool RandomSeedFlag = false;
  int RandomGeneratorSeed = 95;
  Teuchos::SerialDenseMatrix<int, double> outsurf;
  int N = pow(2, Resolution);
  outsurf.shape(N + 1, N + 1);
  double InitialTopologyStdDeviation = 20.0;

  MIRCO::Rmg surface(
      Resolution, InitialTopologyStdDeviation, HurstExponent, RandomSeedFlag, RandomGeneratorSeed);
  surface.GetSurface(outsurf);

  EXPECT_NEAR(outsurf(0, 0), 23.5435469989256, 1e-06);
  EXPECT_NEAR(outsurf(0, 1), 30.2624522170979, 1e-06);
  EXPECT_NEAR(outsurf(0, 2), 69.5813622417479, 1e-06);
  EXPECT_NEAR(outsurf(0, 3), 43.5026425381265, 1e-06);
  EXPECT_NEAR(outsurf(0, 4), 23.5435469989256, 1e-06);
  EXPECT_NEAR(outsurf(1, 0), 68.8507553267314, 1e-06);
  EXPECT_NEAR(outsurf(1, 1), 73.8350740079714, 1e-06);
  EXPECT_NEAR(outsurf(1, 2), 77.9927972851754, 1e-06);
  EXPECT_NEAR(outsurf(1, 3), 35.2927793006724, 1e-06);
  EXPECT_NEAR(outsurf(1, 4), 22.6620325442329, 1e-06);
  EXPECT_NEAR(outsurf(2, 0), 39.1583562054882, 1e-06);
  EXPECT_NEAR(outsurf(2, 1), 19.2247183888878, 1e-06);
  EXPECT_NEAR(outsurf(2, 2), 79.1711886771701, 1e-06);
  EXPECT_NEAR(outsurf(2, 3), 5.66729306836534, 1e-06);
  EXPECT_NEAR(outsurf(2, 4), 41.3691438722521, 1e-06);
  EXPECT_NEAR(outsurf(3, 0), 59.1811726494348, 1e-06);
  EXPECT_NEAR(outsurf(3, 1), 21.2400598989696, 1e-06);
  EXPECT_NEAR(outsurf(3, 2), 54.6656122080671, 1e-06);
  EXPECT_NEAR(outsurf(3, 3), 28.0246974768169, 1e-06);
  EXPECT_NEAR(outsurf(3, 4), 6.72730409669533, 1e-06);
  EXPECT_NEAR(outsurf(4, 0), 23.5435469989256, 1e-06);
  EXPECT_NEAR(outsurf(4, 1), 0, 1e-03);
  EXPECT_NEAR(outsurf(4, 2), 30.6777944575233, 1e-06);
  EXPECT_NEAR(outsurf(4, 3), 35.2191824993355, 1e-06);
  EXPECT_NEAR(outsurf(4, 4), 23.5435469989256, 1e-06);
}


TEST(warmstarting, warmstart)
{
  MIRCO::view_dvec x0("x0", 3);
  MIRCO::view_dvec xv0("xv0", 3);
  MIRCO::view_dvec yv0("yv0", 3);
  MIRCO::view_dvec xvf("xvf", 2);
  MIRCO::view_dvec yvf("yvf", 2);
  MIRCO::view_dvec pf("pf", 2);

  // [TODO] host/device separation
  xv0(0) = 1;
  xv0(1) = 3;
  xv0(2) = 5;

  yv0(0) = 2;
  yv0(1) = 4;
  yv0(2) = 6;

  xvf(0) = 1;
  xvf(1) = 5;

  yvf(0) = 2;
  yvf(1) = 6;

  pf(0) = 10;
  pf(1) = 30;

  MIRCO::Warmstart(x0, xv0, yv0, xvf, yvf, pf);

  EXPECT_EQ(x0(0), 10);
  EXPECT_EQ(x0(1), 0);
  EXPECT_EQ(x0(2), 30);
}

TEST(warmstarting, warmstart2)
{
  MIRCO::view_dvec x0("x0", 3);
  MIRCO::view_dvec xv0("xv0", 3);
  MIRCO::view_dvec yv0("yv0", 3);
  MIRCO::view_dvec xvf("xvf", 4);
  MIRCO::view_dvec yvf("yvf", 4);
  MIRCO::view_dvec pf("pf", 4);

  xv0(0) = 1;
  xv0(1) = 3;
  xv0(2) = 5;

  yv0(0) = 2;
  yv0(1) = 4;
  yv0(2) = 6;

  xvf(0) = 1;
  xvf(1) = 7;
  xvf(2) = 9;
  xvf(3) = 5;


  yvf(0) = 2;
  yvf(1) = 8;
  yvf(2) = 10;
  yvf(3) = 6;

  pf(0) = 10;
  pf(1) = 50;
  pf(2) = 70;
  pf(3) = 30;

  MIRCO::Warmstart(x0, xv0, yv0, xvf, yvf, pf);

  EXPECT_EQ(x0(0), 10);
  EXPECT_EQ(x0(1), 0);
  EXPECT_EQ(x0(2), 30);
}

int main(int argc, char **argv)
{
  int res;

  MPI_Init(&argc, &argv);
  Kokkos::initialize (argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  res = RUN_ALL_TESTS();

  Kokkos::finalize();

  MPI_Finalize();

  return res;
}
