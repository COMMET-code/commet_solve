#ifndef INCLUDE_COMMET_SOLVE_TENSOR_UTILS_HPP_
#define INCLUDE_COMMET_SOLVE_TENSOR_UTILS_HPP_

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>


namespace commet_solve{
using namespace dealii;

template <int dim, typename Number = double>
inline SymmetricTensor<4, dim, Number>
IoA_and_AoI(const SymmetricTensor<2, dim, Number> &A) {
  SymmetricTensor<4, dim, Number> out;
  out = 0;

  for (unsigned int ij = 0; ij < dim; ij++)
    for (unsigned int k = 0; k < dim; k++)
      for (unsigned int l = k; l < dim; l++)
        out[ij][ij][k][l] = A[k][l];

  for (unsigned int kl = 0; kl < dim; kl++)
    for (unsigned int i = 0; i < dim; i++)
      for (unsigned int j = i; j < dim; j++)
        out[i][j][kl][kl] += A[i][j];

  return out;
}

template <int dim, typename Number = double>
inline SymmetricTensor<4, dim, Number>
odot(const SymmetricTensor<2, dim, Number> &A) {
  SymmetricTensor<4, dim, Number> out;

  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = i; j < dim; j++)
      for (unsigned int k = 0; k < dim; k++)
        for (unsigned int l = k; l < dim; l++)
          // out[i][j][k][l] += A[i][k] * A[l][j];
          out[i][j][k][l] += 0.5 * (A[i][k] * A[l][j] + A[i][l] * A[j][k]);

  return out;
}

template <int dim, typename Number = double>
inline SymmetricTensor<4, dim, Number>
odot(const SymmetricTensor<2, dim, Number> &A,
     const SymmetricTensor<2, dim, Number> &B) {
  SymmetricTensor<4, dim, Number> out;

  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = i; j < dim; j++)
      for (unsigned int k = 0; k < dim; k++)
        for (unsigned int l = k; l < dim; l++)
          out[i][j][k][l] += A[i][k] * B[j][l] + B[i][l] * A[j][k];

  out /= 2;

  return out;
}



}

#endif  // INCLUDE_COMMET_SOLVE_TENSOR_UTILS_HPP_
