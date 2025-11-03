#ifndef INCLUDE_NCM_DOMAIN_TENSOR_CONVERSION_UTILS_HPP_
#define INCLUDE_NCM_DOMAIN_TENSOR_CONVERSION_UTILS_HPP_

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <span>

#undef Assert
#include <torch/torch.h>

namespace commet_solve {
using namespace dealii;

enum class TensorLayout {
  STANDARD,
  VOIGT,
};

const std::vector<std::pair<unsigned int, unsigned int>> VOIGT_NUMBERING{
    {0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

inline TensorLayout determine_tensor_layout(const torch::Tensor &tensor) {
  if (tensor.size(1) == 6)
    return TensorLayout::VOIGT;
  else if (tensor.size(1) == 3)
    return TensorLayout::STANDARD;
  else
    throw std::logic_error(
        "Got unexpected tensor shape in 'determine_tensor_layout'");
}

template <int dim, typename Number>
inline void
torch_to_deal_tensor(const unsigned int &entry, const torch::Tensor &src,
                     Tensor<2, dim> &target, const TensorLayout &layout) {
  switch (layout) {
  case TensorLayout::STANDARD: {
    const int start = entry * dim * dim;
    for (int i = 0; i < dim; i++) {
      const int start_row = start + dim * i;
      for (int j = 0; j < dim; j++)
        target[i][j] = *(src.data_ptr<Number>() + start_row + j);
    }
    break;
  }
  case TensorLayout::VOIGT: {
    const unsigned int start = entry * 6;

    target[0][0] = *(src.data_ptr<Number>() + start);
    target[1][1] = *(src.data_ptr<Number>() + start + 1);
    target[2][2] = *(src.data_ptr<Number>() + start + 2);
    target[1][2] = *(src.data_ptr<Number>() + start + 3);
    target[0][2] = *(src.data_ptr<Number>() + start + 4);
    target[0][1] = *(src.data_ptr<Number>() + start + 5);

    target[1][0] = target[0][1];
    target[2][0] = target[0][2];
    target[1][2] = target[1][2];

    break;
  }
  }
}

template <int dim, typename Number>
inline void
deal_to_torch_tensor(const unsigned int &entry, const Tensor<2, dim> &src,
                     torch::Tensor &target, const TensorLayout &layout) {
  switch (layout) {
  case TensorLayout::STANDARD: {
    const unsigned int start = entry * dim * dim;
    for (unsigned int i = 0; i < dim; i++) {
      const unsigned int start_row = start + dim * i;
      for (unsigned int j = 0; j < dim; j++)
        *(target.data_ptr<Number>() + start_row + j) = src[i][j];
    }

    break;
  }
  case TensorLayout::VOIGT: {
    const unsigned int start = entry * 6;

    *(target.data_ptr<Number>() + start) = src[0][0];
    *(target.data_ptr<Number>() + start + 1) = src[1][1];
    *(target.data_ptr<Number>() + start + 2) = src[2][2];
    *(target.data_ptr<Number>() + start + 3) = src[1][2];
    *(target.data_ptr<Number>() + start + 4) = src[0][2];
    *(target.data_ptr<Number>() + start + 5) = src[0][1];
    break;
  }
  }
}

template <int dim, typename Number>
inline void
mat_point_structural_vector_to_torch_tensor(const unsigned int &entry,
                     const std::span<Tensor<1, dim, Number>> src,
                     const unsigned int & n_vecs_per_point,
                     torch::Tensor &target) {
    // assert(n_vecs_per_point<=N_ORIENTATION_VECS);

    const unsigned int start = entry * n_vecs_per_point * dim;
    for(unsigned int i_vec =0; i_vec < n_vecs_per_point; i_vec++) { 
        const unsigned int start_row = start + dim * i_vec; 
        for(unsigned int i_comp=0; i_comp<dim; i_comp++){
            *(target.data_ptr<Number>() + start_row + i_comp) = src[i_vec][i_comp];
        }
    }

}

template <int dim, typename Number>
inline void torch_to_deal_tensor(const unsigned int &entry,
                                 const torch::Tensor &src,
                                 SymmetricTensor<2, dim> &target,
                                 const TensorLayout &layout) {
  switch (layout) {
  case TensorLayout::STANDARD: {
    const unsigned int start = entry * dim * dim;
    for (unsigned int i = 0; i < dim; i++) {
      const unsigned int start_row = start + dim * i;
      for (unsigned int j = i; j < dim; j++)
        target[i][j] = *(src.data_ptr<Number>() + start_row + j);
    }
    break;
  }
  case TensorLayout::VOIGT: {
    const unsigned int start = entry * 6;

    target[0][0] = *(src.data_ptr<Number>() + start);
    target[1][1] = *(src.data_ptr<Number>() + start + 1);
    target[2][2] = *(src.data_ptr<Number>() + start + 2);
    target[1][2] = *(src.data_ptr<Number>() + start + 3);
    target[0][2] = *(src.data_ptr<Number>() + start + 4);
    target[0][1] = *(src.data_ptr<Number>() + start + 5);
    break;
  }
  }
}

template <int dim, typename Number>
inline void deal_to_torch_tensor(const unsigned int &entry,
                                 const SymmetricTensor<2, dim> &src,
                                 torch::Tensor &target,
                                 const TensorLayout &layout) {
  switch (layout) {
  case TensorLayout::STANDARD: {
    const unsigned int start = entry * dim * dim;
    for (unsigned int i = 0; i < dim; i++) {
      const unsigned int start_row = start + dim * i;
      for (unsigned int j = 0; j < dim; j++)
        *(target.data_ptr<Number>() + start_row + j) = src[i][j];
    }
    break;
  }
  case TensorLayout::VOIGT: {
    const unsigned int start = entry * 6;

    *(target.data_ptr<Number>() + start) = src[0][0];
    *(target.data_ptr<Number>() + start + 1) = src[1][1];
    *(target.data_ptr<Number>() + start + 2) = src[2][2];
    *(target.data_ptr<Number>() + start + 3) = src[1][2];
    *(target.data_ptr<Number>() + start + 4) = src[0][2];
    *(target.data_ptr<Number>() + start + 5) = src[0][1];
    break;
  }
  }
}

template <int dim, typename Number>
inline void torch_to_deal_tensor(const unsigned int &entry,
                                 const torch::Tensor &src,
                                 SymmetricTensor<4, dim> &target,
                                 const TensorLayout &layout) {
  switch (layout) {
  case TensorLayout::STANDARD: {

    const unsigned int start = entry * dim * dim * dim * dim;
    for (unsigned int i = 0; i < dim; i++) {
      const int start_row_i = start + dim * dim * dim * i;
      // for (unsigned int j = 0; j < dim; j++)
      for (unsigned int j = i; j < dim; j++) {
        const int start_row_j = start_row_i + dim * dim * j;
        for (unsigned int k = 0; k < dim; k++) {
          const int start_row_k = start_row_j + dim * k;
          // for (unsigned int l = 0; l < dim; l++)
          for (unsigned int l = k; l < dim; l++) {
            target[i][j][k][l] = *(src.data_ptr<Number>() + start_row_k + l);
          }
        }
      }
    }
    break;
  }
  case TensorLayout::VOIGT: {
    const unsigned int n_vals = VOIGT_NUMBERING.size();
    const unsigned int start = entry * n_vals * n_vals;

    for (unsigned int i = 0; i < n_vals; i++) {

      const int start_row_i = start + n_vals * i;
      const unsigned int &iv = VOIGT_NUMBERING.at(i).first;
      const unsigned int &jv = VOIGT_NUMBERING.at(i).second;

      target[iv][jv][0][0] = *(src.data_ptr<Number>() + start_row_i);
      target[iv][jv][1][1] = *(src.data_ptr<Number>() + start_row_i + 1);
      target[iv][jv][2][2] = *(src.data_ptr<Number>() + start_row_i + 2);
      target[iv][jv][1][2] = *(src.data_ptr<Number>() + start_row_i + 3);
      target[iv][jv][0][2] = *(src.data_ptr<Number>() + start_row_i + 4);
      target[iv][jv][0][1] = *(src.data_ptr<Number>() + start_row_i + 5);
    }
    break;
  }
  }
}

} // namespace commet_solve

#endif // INCLUDE_NCM_DOMAIN_TENSOR_CONVERSION_UTILS_HPP_
