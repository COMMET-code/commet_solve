#ifndef INCLUDE_NCM_DOMAIN_VECTORIZED_DOMAIN_HPP_
#define INCLUDE_NCM_DOMAIN_VECTORIZED_DOMAIN_HPP_

#include "../material_domain.hpp"
#include "commet_solve/material_domain/material_points.hpp"
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <stdexcept>
#include <string>

// Deal.II defines an Assert marco
// This ends up clashing with the torch library
// Hence we undef Assert here.
// This also means that the order of includes is important (deal.II must come
// first, then torch)
#undef Assert
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include "tensor_conversion_utils.hpp"

namespace commet_solve {

using t_Tensor = torch::Tensor;
using t_Slice = torch::indexing::Slice;

enum class NCMEvaluationMethod {
  USING_F,
  USING_C,
  OPT_F,
};

const static std::map<NCMEvaluationMethod, std::string> //
    ncm_evaluation_method_names({
        {NCMEvaluationMethod::USING_F, "W_NN_from_F"},    //
        {NCMEvaluationMethod::USING_C, "W_NN_from_C"},    //
        {NCMEvaluationMethod::OPT_F, "psi_tau_cc_from_F"} //
    });

template <int dim, typename Number = double>
class VectorizedMaterialDomain : public MaterialDomain<dim, Number> {
public:
  VectorizedMaterialDomain(const unsigned int &n_structural_vectors = 0) 
        : n_structural_vectors(n_structural_vectors) 
        , module_loaded(false)
        , evaluation_method(NCMEvaluationMethod::USING_F)
        {};
  VectorizedMaterialDomain(VectorizedMaterialDomain &&) = delete;
  VectorizedMaterialDomain(const VectorizedMaterialDomain &) = delete;
  VectorizedMaterialDomain &operator=(VectorizedMaterialDomain &&) = delete;
  VectorizedMaterialDomain &
  operator=(const VectorizedMaterialDomain &) = delete;
  ~VectorizedMaterialDomain() = default;

  void set_evaluation_method(const NCMEvaluationMethod &method) {
    this->evaluation_method = method;
    if (module_loaded)
      this->check_module();
  };

  void load_model(const std::string &pth_to_model) {

    module = torch::jit::script::Module(torch::jit::load(pth_to_model));
    module_loaded = true;

    //         // Deving and debugging --
    //         // Can be helpful to see method names and args:
    // for (const auto &method : module.get_methods())
    // {
    // 	std::cout << "Method name: " << method.name() << std::endl;
    // 	const auto &schema = method.function().getSchema();
    // 	for (const auto &thing : schema.arguments())
    // 	{
    // 		std::cout << "Arg: " << thing.name() << "\t type: " <<
    // thing.type()->str() << std::endl;
    // 	}
    // 	std::cout << "Returns: " << std::endl;
    // 	for (const auto &thing : schema.returns())
    // 	{
    // 		std::cout << "type: " << thing.type()->str() << std::endl;
    // 	}
    // }

    module.to(torch::kDouble);
    this->check_module();
  };

  void add_entry(const dealii::types::global_cell_index &cell,
                 const unsigned int &qp) override {
    qp_data[{cell, qp}] = HyperelasticMaterialPointData<dim, Number>();
  };

  void add_entry(const dealii::types::global_cell_index &cell_idx,
                 const DoFCellAccessor<dim, dim, false> & /*cell*/,
                 const FEValues<dim, dim> &fe_values,
                 ScalarFieldManager<dim, Number> & /*scalar_fields*/,
                 VectorFieldManager<dim, Number> & /*vector_fields*/,
                 TensorFieldManager<dim, Number> & /*tensor_fields*/) override {
    for (unsigned int qp = 0; qp < fe_values.n_quadrature_points; qp++)
      qp_data[{cell_idx, qp}] = HyperelasticMaterialPointData<dim, Number>();
  };

  void update_F(const dealii::types::global_cell_index &cell,
                const unsigned int &qp,
                const Tensor<2, dim, Number> &F) override {
    qp_data.at({cell, qp}).F = F;
  };

  void get_vals(const dealii::types::global_cell_index &cell,
                const unsigned int &qp, Tensor<2, dim, Number> &F, Number &psi,
                SymmetricTensor<2, dim, Number> &tau,
                SymmetricTensor<4, dim, Number> &cc) override {
    const HyperelasticMaterialPointData<dim, Number> &point =
        qp_data.at({cell, qp});

    F = point.F;
    psi = point.psi;
    tau = point.tau;
    cc = point.cc;
  };

  //   virtual void compute_constitutive_behaviour() override {

  //   throw std::logic_error("Not implemented yet...");
  // };

  void evaluate_model(const torch::Tensor &F,
                      const torch::Tensor &structural_tensors,
                      torch::Tensor &strain_energy_out,
                      torch::Tensor &kirchhoff_stress_out,
                      torch::Tensor &spatial_stiffness_out);

  // virtual Number get_scalar_value(const dealii::types::global_cell_index &
  // /*cell*/, 								const unsigned int & /*qp*/, 								const scalar_output_flag & /*flag*/)
  // {
  // 	return NAN;
  // };

  // virtual Tensor<1, dim, Number> get_vector_value(const
  // dealii::types::global_cell_index & /*cell*/, 												const unsigned int & /*qp*/,
  // 												const
  // vector_output_flag & /*flag*/)
  // {
  // 	return Tensor<1, dim, Number>({NAN, NAN, NAN});
  // };

  // virtual Tensor<2, dim, Number> get_tensor_value(const
  // dealii::types::global_cell_index & /*cell*/, 												const unsigned int & /*qp*/,
  // 												const
  // tensor_output_flag & /*flag*/)
  // {
  // 	return Tensor<2, dim, Number>({{NAN, NAN, NAN}, {NAN, NAN, NAN}, {NAN,
  // NAN, NAN}});
  // };

  virtual Number get_scalar_value(const dealii::types::global_cell_index &cell,
                                  const unsigned int &qp,
                                  const scalar_output_flag &flag) override {
    switch (flag) {
    case scalar_output_flag::jacobian:
      return determinant(qp_data.at({cell, qp}).F);
    case scalar_output_flag::strain_energy:
      return qp_data.at({cell, qp}).psi;
    case scalar_output_flag::I1:
      return trace(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F));
    case scalar_output_flag::I2: {
      const Tensor<2, dim, Number> b =
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F);
      return 0.5 * (pow(trace(b), 2) - trace(b * b));
    }
    case scalar_output_flag::I3:
      return pow(determinant(qp_data.at({cell, qp}).F), 2);
    default:
      return MaterialDomain<dim, Number>::get_scalar_value(cell, qp, flag);
    }
  };

  virtual Tensor<1, dim, Number>
  get_vector_value(const dealii::types::global_cell_index &cell,
                   const unsigned int &qp,
                   const vector_output_flag &flag) override {
    switch (flag) {
    case vector_output_flag::principal_stress_0: {
      auto vecs_and_vals = eigenvectors(
          qp_data.at({cell, qp}).tau, SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[0].first * vecs_and_vals[0].second /
             vecs_and_vals[0].second.norm();
    }
    case vector_output_flag::principal_stress_1: {
      auto vecs_and_vals = eigenvectors(
          qp_data.at({cell, qp}).tau, SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[1].first * vecs_and_vals[1].second /
             vecs_and_vals[1].second.norm();
    }
    case vector_output_flag::principal_stress_2: {
      auto vecs_and_vals = eigenvectors(
          qp_data.at({cell, qp}).tau, SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[2].first * vecs_and_vals[2].second /
             vecs_and_vals[2].second.norm();
    }
    case vector_output_flag::principal_stretch_0: {
      auto vecs_and_vals = eigenvectors(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F),
          SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[0].first * vecs_and_vals[0].second /
             vecs_and_vals[0].second.norm();
    }
    case vector_output_flag::principal_stretch_1: {
      auto vecs_and_vals = eigenvectors(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F),
          SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[1].first * vecs_and_vals[1].second /
             vecs_and_vals[1].second.norm();
    }
    case vector_output_flag::principal_stretch_2: {
      auto vecs_and_vals = eigenvectors(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F),
          SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[2].first * vecs_and_vals[2].second /
             vecs_and_vals[2].second.norm();
    }
    default:
      return MaterialDomain<dim, Number>::get_vector_value(cell, qp, flag);
    }
  };

  virtual Tensor<2, dim, Number>
  get_tensor_value(const dealii::types::global_cell_index &cell,
                   const unsigned int &qp,
                   const tensor_output_flag &flag) override {
    switch (flag) {
    case tensor_output_flag::kirchhoff_stress: {
      return qp_data.at({cell, qp}).tau;
    }
    case tensor_output_flag::F: {
      return qp_data.at({cell, qp}).F;
    }
    case tensor_output_flag::B: {
      return Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F);
    }
    case tensor_output_flag::C: {
      return Physics::Elasticity::Kinematics::C(qp_data.at({cell, qp}).F);
    }
    case tensor_output_flag::E: {
      return Physics::Elasticity::Kinematics::E(qp_data.at({cell, qp}).F);
    }
    default:
      return MaterialDomain<dim, Number>::get_tensor_value(cell, qp, flag);
    }
  };

  const unsigned int n_structural_vectors;

protected:
  std::unordered_map<point_index, HyperelasticMaterialPointData<dim, Number>,
                     PointIndexHash>
      qp_data;
  TensorLayout get_return_tensor_layout() {
    switch (this->evaluation_method) {
    case NCMEvaluationMethod::USING_F:
    case NCMEvaluationMethod::USING_C:
      return TensorLayout::STANDARD;
    case NCMEvaluationMethod::OPT_F:
      return TensorLayout::VOIGT;
    }

    throw std::logic_error(
        "Couldn't determine return tensor layout in VectorizedMaterialDomain..."
        "This is a bug. Please contact developers.");
  }

private:
  torch::jit::script::Module module;
  bool module_loaded;
  NCMEvaluationMethod evaluation_method;

  void check_module();

  inline void evaluate_model_from_F(const torch::Tensor &F,
                                    const torch::Tensor &structural_tensors,
                                    torch::Tensor &strain_energy_out,
                                    torch::Tensor &kirchhoff_stress_out,
                                    torch::Tensor &spatial_stiffness_out);

  inline void evaluate_model_from_C(const torch::Tensor &F,
                                    const torch::Tensor &structural_tensors,
                                    torch::Tensor &strain_energy_out,
                                    torch::Tensor &kirchhoff_stress_out,
                                    torch::Tensor &spatial_stiffness_out);

  inline void evaluate_model_from_F_opt(const torch::Tensor &F,
                                        const torch::Tensor &structural_tensors,
                                        torch::Tensor &strain_energy_out,
                                        torch::Tensor &kirchhoff_stress_out,
                                        torch::Tensor &spatial_stiffness_out);
};

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::check_module() {

  bool found_method = false;
  std::string neccessary_method =
      ncm_evaluation_method_names.at(this->evaluation_method);

  for (const auto &method : module.get_methods()) {
    found_method = method.name() == neccessary_method;
    if (found_method) {
      const auto &schema = method.function().getSchema();
      // Note for later if you also want to check types you can do so using
      // arguments.at(i).type()->str();
      if (schema.arguments().size() != 3)
        throw std::runtime_error(
            "Found the neccessary method '" + neccessary_method +
            "', but it takes " + std::to_string(schema.arguments().size()) +
            " arguments, when it should take exactly 3 arguments: " +
            "self, F: torch.Tensor, structural_tensors: torch.Tensor.");
      if (schema.returns().size() != 1)
        throw std::runtime_error("Found the neccessary method '" +
                                 neccessary_method + "', but it returns " +
                                 std::to_string(schema.returns().size()) +
                                 " value(s), when it should return 1 value.");

      break;
    }
  }
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model(
    const torch::Tensor &F, const torch::Tensor &structural_tensors,
    torch::Tensor &strain_energy_out, torch::Tensor &kirchhoff_stress_out,
    torch::Tensor &spatial_stiffness_out) {
  switch (evaluation_method) {
  case NCMEvaluationMethod::USING_F:
    this->evaluate_model_from_F(F, structural_tensors, strain_energy_out,
                                kirchhoff_stress_out, spatial_stiffness_out);
    break;
  case NCMEvaluationMethod::USING_C:
    this->evaluate_model_from_C(F, structural_tensors, strain_energy_out,
                                kirchhoff_stress_out, spatial_stiffness_out);
    break;
  case NCMEvaluationMethod::OPT_F:
    this->evaluate_model_from_F_opt(F, structural_tensors, strain_energy_out,
                                    kirchhoff_stress_out,
                                    spatial_stiffness_out);
    break;
  }
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model_from_F(
    const torch::Tensor &F_mem_safe, const torch::Tensor &structural_tensors,
    torch::Tensor &strain_energy_out, torch::Tensor &kirchhoff_stress_out,
    torch::Tensor &spatial_stiffness_out) {

  torch::Tensor F = torch::zeros_like(F_mem_safe);
  // F.requires_grad_(true);
  // torch::Tensor Is = torch::zeros_like(F);
  F.index_put_({t_Slice(), t_Slice(), t_Slice()}, torch::eye(dim));
  F.requires_grad_(true);

  t_Tensor W_NN =
      module.run_method("W_NN_from_F", F, structural_tensors).toTensor();

  t_Tensor grad_output = torch::ones_like(W_NN);

  t_Tensor H = -torch::autograd::grad({W_NN}, {F},
                                      /*grad_outputs=*/{grad_output},
                                      /*retain_graph=*/true,
                                      /*create_graph=*/true)[0][0];
  t_Tensor W0 = -W_NN[0].clone();

  F = F_mem_safe.detach().clone();
  F.requires_grad_(true);

  strain_energy_out =
      module.run_method("W_NN_from_F", F, structural_tensors).toTensor();

  grad_output = torch::ones_like(strain_energy_out);

  kirchhoff_stress_out =
      torch::autograd::grad({strain_energy_out}, {F}, {grad_output},
                            /*retain_graph=*/true,
                            /*create_graph=*/true)[0];

  t_Tensor grad_grad_output = torch::ones({strain_energy_out.size(0)});

  t_Tensor dP_ij_dF = torch::zeros_like(kirchhoff_stress_out);

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) {
      const t_Tensor &grad_slice =
          kirchhoff_stress_out.index({t_Slice(), i, j});

      dP_ij_dF = torch::autograd::grad({grad_slice}, {F}, {grad_grad_output},
                                       /*retain_graph=*/true,
                                       /*create_graph=*/false)[0];

      for (int k = 0; k < dim; k++)
        for (int l = 0; l < dim; l++)
          spatial_stiffness_out.index({t_Slice(), i, j, k, l}) =
              dP_ij_dF.index({t_Slice(), k, l});
    }

  F = F_mem_safe.detach().clone();
  F.requires_grad_(false);
  torch::NoGradGuard no_grad;

  strain_energy_out = strain_energy_out + W0;

  kirchhoff_stress_out = kirchhoff_stress_out + H;
  kirchhoff_stress_out =
      torch::einsum("...il,...jl->...ij", {kirchhoff_stress_out, F});

  spatial_stiffness_out =
      torch::einsum("miJkL,mjJ,mlL->mijkl", {spatial_stiffness_out, F, F});
  spatial_stiffness_out =
      spatial_stiffness_out -
      torch::einsum("ik,mjl->mijkl", {torch::eye(dim), kirchhoff_stress_out});
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model_from_C(
    const torch::Tensor &F, const torch::Tensor &structural_tensors,
    torch::Tensor &strain_energy_out, torch::Tensor &kirchhoff_stress_out,
    torch::Tensor &spatial_stiffness_out) {

  torch::Tensor Cs = torch::zeros_like(F);

  Cs.index_put_({t_Slice(), t_Slice(), t_Slice()}, torch::eye(dim));
  Cs.requires_grad_(true);

  t_Tensor W_NN =
      module.run_method("W_NN_from_C", Cs, structural_tensors).toTensor();

  t_Tensor grad_output = torch::ones_like(W_NN);

  t_Tensor H = -torch::autograd::grad({W_NN}, {Cs},
                                      /*grad_outputs=*/{grad_output},
                                      /*retain_graph=*/true,
                                      /*create_graph=*/true)[0][0] *
               2;
  t_Tensor W0 = -W_NN[0].clone();

  Cs = torch::einsum("mki,mkj->mij", {F, F});
  Cs.requires_grad_(true);

  strain_energy_out =
      module.run_method("W_NN_from_C", Cs, structural_tensors).toTensor();

  grad_output = torch::ones_like(strain_energy_out);

  kirchhoff_stress_out =
      2 * torch::autograd::grad({strain_energy_out}, {Cs}, {grad_output},
                                /*retain_graph=*/true,
                                /*create_graph=*/true)[0];

  t_Tensor grad_grad_output = torch::ones({strain_energy_out.size(0)});

  t_Tensor dS_ij_dC = torch::zeros_like(kirchhoff_stress_out);

  for (int i = 0; i < dim; i++)
    for (int j = i; j < dim; j++) {

      const t_Tensor &grad_slice =
          // kirchhoff_stress_out.index({torch::indexing::Slice(torch::indexing::None),
          // i, j});
          kirchhoff_stress_out.index({t_Slice(), i, j});

      // std::cout << "5.1" << std::endl;
      dS_ij_dC =
          2 * torch::autograd::grad({grad_slice}, {Cs}, {grad_grad_output}, //
                                    /*retain_graph=*/true,
                                    /*create_graph=*/false)[0];

      // std::cout << "5.2" << std::endl;
      for (int k = 0; k < dim; k++)
        for (int l = k; l < dim; l++) {

          // std::cout << "5.3" << std::endl;
          spatial_stiffness_out.index({t_Slice(), i, j, k, l}) =
              dS_ij_dC.index({t_Slice(), k, l});
          spatial_stiffness_out.index({t_Slice(), j, i, k, l}) =
              dS_ij_dC.index({t_Slice(), k, l});
          spatial_stiffness_out.index({t_Slice(), j, i, l, k}) =
              dS_ij_dC.index({t_Slice(), k, l});
          spatial_stiffness_out.index({t_Slice(), i, j, l, k}) =
              dS_ij_dC.index({t_Slice(), k, l});
          // std::cout << "5.4" << std::endl;
        }
    }

  spatial_stiffness_out = torch::einsum("mIJKL,miI,mjJ,mkK,mlL->mijkl",
                                        {spatial_stiffness_out, F, F, F, F});

  kirchhoff_stress_out = kirchhoff_stress_out + H;
  kirchhoff_stress_out =
      torch::einsum("...ik,...kl,...jl->...ij", {F, kirchhoff_stress_out, F});

  strain_energy_out = strain_energy_out + W0;
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model_from_F_opt(
    const torch::Tensor &F, const torch::Tensor &structural_tensors,
    torch::Tensor &strain_energy_out, torch::Tensor &kirchhoff_stress_out,
    torch::Tensor &spatial_stiffness_out) {
  torch::NoGradGuard no_grad;

  at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
  auto result =
      module.run_method("psi_tau_cc_from_F", F, structural_tensors).toTuple();
  strain_energy_out = result->elements()[0].toTensor().to(options);
  kirchhoff_stress_out = result->elements()[1].toTensor().to(options);
  spatial_stiffness_out = result->elements()[2].toTensor().to(options);
}

} // namespace commet_solve

#endif // INCLUDE_NCM_DOMAIN_VECTORIZED_DOMAIN_HPP_
