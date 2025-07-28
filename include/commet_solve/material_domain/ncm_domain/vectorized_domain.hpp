#ifndef INCLUDE_NCM_DOMAIN_VECTORIZED_DOMAIN_HPP_
#define INCLUDE_NCM_DOMAIN_VECTORIZED_DOMAIN_HPP_

#include "../material_domain.hpp"

#include <stdexcept>
#include <string>

// Deal.II defines an Assert marco
// This ends up clashing with the torch library
// Hence we undef Assert here.
// This also means that the order of includes is important (deal.II must come first, then torch)
#undef Assert
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include "tensor_conversion_utils.hpp"

namespace commet_solve
{

using t_Tensor = torch::Tensor;
using t_Slice = torch::indexing::Slice;

enum class NCMEvaluationMethod
{
	USING_F,
	USING_C,
	OPT_F,
};

const static std::map<NCMEvaluationMethod, std::string> //
	ncm_evaluation_method_names({
		{NCMEvaluationMethod::USING_F, "W_NN_from_F"},	  //
		{NCMEvaluationMethod::USING_C, "W_NN_from_C"},	  //
		{NCMEvaluationMethod::OPT_F, "psi_tau_cc_from_F"} //
	});

template <int dim, typename Number = double>
class VectorizedMaterialDomain : public MaterialDomain<dim, Number>
{
  public:
	VectorizedMaterialDomain()
		: module_loaded(false)
		, evaluation_method(NCMEvaluationMethod::USING_F) {};
	VectorizedMaterialDomain(VectorizedMaterialDomain &&) = delete;
	VectorizedMaterialDomain(const VectorizedMaterialDomain &) = delete;
	VectorizedMaterialDomain &operator=(VectorizedMaterialDomain &&) = delete;
	VectorizedMaterialDomain &operator=(const VectorizedMaterialDomain &) = delete;
	~VectorizedMaterialDomain() = default;

	void set_evaluation_method(const NCMEvaluationMethod &method)
	{
		this->evaluation_method = method;
		if (module_loaded)
			this->check_module();
	};

	void load_model(const std::string &pth_to_model)
	{

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
		// 		std::cout << "Arg: " << thing.name() << "\t type: " << thing.type()->str() << std::endl;
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

	void add_entry(const dealii::types::global_cell_index &cell, const unsigned int &qp) override
	{
		qp_data[{cell, qp}] = MinimalMaterialPointData<dim, Number>();
	};

	void update_F(const dealii::types::global_cell_index &cell,
				  const unsigned int &qp,
				  const Tensor<2, dim, Number> &F) override
	{
		qp_data.at({cell, qp}).F = F;
	};

	void get_vals(const dealii::types::global_cell_index &cell,
				  const unsigned int &qp,
				  Tensor<2, dim, Number> &F,
				  Number &psi,
				  SymmetricTensor<2, dim, Number> &tau,
				  SymmetricTensor<4, dim, Number> &cc) override
	{
		const MinimalMaterialPointData<dim, Number> &point = qp_data.at({cell, qp});

		F = point.F;
		psi = point.psi;
		tau = point.tau;
		cc = point.cc;
	};

	void compute_constitutive_behaviour() override
	{

		throw std::logic_error("Not implemented yet...");
	};

	void evaluate_model(const torch::Tensor &F,
						const torch::Tensor &structural_tensors,
						torch::Tensor &strain_energy_out,
						torch::Tensor &kirchhoff_stress_out,
						torch::Tensor &spatial_stiffness_out);

  protected:
	std::unordered_map<point_index, MinimalMaterialPointData<dim, Number>, PointIndexHash> qp_data;
	TensorLayout get_return_tensor_layout()
	{
		switch (this->evaluation_method)
		{
		case NCMEvaluationMethod::USING_F:
		case NCMEvaluationMethod::USING_C:
			return TensorLayout::STANDARD;
		case NCMEvaluationMethod::OPT_F:
			return TensorLayout::VOIGT;
		}

        throw std::logic_error("Couldn't determine return tensor layout in VectorizedMaterialDomain..."
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
void VectorizedMaterialDomain<dim, Number>::check_module()
{

	bool found_method = false;
	std::string neccessary_method = ncm_evaluation_method_names.at(this->evaluation_method);

	for (const auto &method : module.get_methods())
	{
		found_method = method.name() == neccessary_method;
		if (found_method)
		{
			const auto &schema = method.function().getSchema();
			// Note for later if you also want to check types you can do so using arguments.at(i).type()->str();
			if (schema.arguments().size() != 3)
				throw std::runtime_error("Found the neccessary method '" + neccessary_method + "', but it takes " +
										 std::to_string(schema.arguments().size()) +
										 " arguments, when it should take exactly 3 arguments: " +
										 "self, F: torch.Tensor, structural_tensors: torch.Tensor.");
			if (schema.returns().size() != 1)
				throw std::runtime_error("Found the neccessary method '" + neccessary_method + "', but it returns " +
										 std::to_string(schema.returns().size()) +
										 " value(s), when it should return 1 value.");

			break;
		}
	}
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model(const torch::Tensor &F,
														   const torch::Tensor &structural_tensors,
														   torch::Tensor &strain_energy_out,
														   torch::Tensor &kirchhoff_stress_out,
														   torch::Tensor &spatial_stiffness_out)
{
	switch (evaluation_method)
	{
	case NCMEvaluationMethod::USING_F:
		this->evaluate_model_from_F(
			F, structural_tensors, strain_energy_out, kirchhoff_stress_out, spatial_stiffness_out);
		break;
	case NCMEvaluationMethod::USING_C:
		this->evaluate_model_from_C(
			F, structural_tensors, strain_energy_out, kirchhoff_stress_out, spatial_stiffness_out);
		break;
	case NCMEvaluationMethod::OPT_F:
		this->evaluate_model_from_F_opt(
			F, structural_tensors, strain_energy_out, kirchhoff_stress_out, spatial_stiffness_out);
		break;
	}
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model_from_F(const torch::Tensor &F,
																  const torch::Tensor &structural_tensors,
																  torch::Tensor &strain_energy_out,
																  torch::Tensor &kirchhoff_stress_out,
																  torch::Tensor &spatial_stiffness_out)
{

	F.requires_grad_(true);
	torch::Tensor Is = torch::zeros_like(F);
	Is.index_put_({t_Slice(), t_Slice(), t_Slice()}, torch::eye(dim));
	Is.requires_grad_(true);

	t_Tensor W_NN = module.run_method("W_NN_from_F", Is, structural_tensors).toTensor();

	t_Tensor grad_output = torch::ones_like(W_NN);

	t_Tensor H = -torch::autograd::grad({W_NN},
										{Is},
										/*grad_outputs=*/{grad_output},
										/*retain_graph=*/true,
										/*create_graph=*/true)[0][0];
	t_Tensor W0 = -W_NN[0].clone();

	strain_energy_out = module.run_method("W_NN_from_F", F, structural_tensors).toTensor();

	grad_output = torch::ones_like(strain_energy_out);

	kirchhoff_stress_out = torch::autograd::grad({strain_energy_out},
												 {F},
												 {grad_output},
												 /*retain_graph=*/true,
												 /*create_graph=*/true)[0];

	t_Tensor grad_grad_output = torch::ones({strain_energy_out.size(0)});

	t_Tensor dP_ij_dF = torch::zeros_like(kirchhoff_stress_out);

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
		{
			const t_Tensor &grad_slice = kirchhoff_stress_out.index({t_Slice(), i, j});

			dP_ij_dF = torch::autograd::grad({grad_slice},
											 {F},
											 {grad_grad_output},
											 /*retain_graph=*/true,
											 /*create_graph=*/false)[0];

			for (int k = 0; k < dim; k++)
				for (int l = 0; l < dim; l++)
					spatial_stiffness_out.index({t_Slice(), i, j, k, l}) = dP_ij_dF.index({t_Slice(), k, l});
		}

	strain_energy_out = strain_energy_out + W0;

	kirchhoff_stress_out = kirchhoff_stress_out + H;
	kirchhoff_stress_out = torch::einsum("...il,...jl->...ij", {kirchhoff_stress_out, F});

	spatial_stiffness_out = torch::einsum("miJkL,mjJ,mlL->mijkl", {spatial_stiffness_out, F, F});
	spatial_stiffness_out = spatial_stiffness_out - torch::einsum("mik,mjl->mijkl", {Is, kirchhoff_stress_out});
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model_from_C(const torch::Tensor &F,
																  const torch::Tensor &structural_tensors,
																  torch::Tensor &strain_energy_out,
																  torch::Tensor &kirchhoff_stress_out,
																  torch::Tensor &spatial_stiffness_out)
{

	torch::Tensor Cs = torch::zeros_like(F);

	Cs.index_put_({t_Slice(), t_Slice(), t_Slice()}, torch::eye(dim));
	Cs.requires_grad_(true);

	t_Tensor W_NN = module.run_method("W_NN_from_C", Cs, structural_tensors).toTensor();

	t_Tensor grad_output = torch::ones_like(W_NN);

	t_Tensor H = -torch::autograd::grad({W_NN},
										{Cs},
										/*grad_outputs=*/{grad_output},
										/*retain_graph=*/true,
										/*create_graph=*/true)[0][0] *
				 2;
	t_Tensor W0 = -W_NN[0].clone();

	Cs = torch::einsum("mki,mkj->mij", {F, F});
	Cs.requires_grad_(true);

	strain_energy_out = module.run_method("W_NN_from_C", Cs, structural_tensors).toTensor();

	grad_output = torch::ones_like(strain_energy_out);

	kirchhoff_stress_out = 2 * torch::autograd::grad({strain_energy_out},
													 {Cs},
													 {grad_output},
													 /*retain_graph=*/true,
													 /*create_graph=*/true)[0];

	t_Tensor grad_grad_output = torch::ones({strain_energy_out.size(0)});

	t_Tensor dS_ij_dC = torch::zeros_like(kirchhoff_stress_out);

	for (int i = 0; i < dim; i++)
		for (int j = i; j < dim; j++)
		{

			const t_Tensor &grad_slice =
				// kirchhoff_stress_out.index({torch::indexing::Slice(torch::indexing::None), i, j});
				kirchhoff_stress_out.index({t_Slice(), i, j});

			// std::cout << "5.1" << std::endl;
			dS_ij_dC = 2 * torch::autograd::grad({grad_slice},
												 {Cs},
												 {grad_grad_output}, //
												 /*retain_graph=*/true,
												 /*create_graph=*/false)[0];

			// std::cout << "5.2" << std::endl;
			for (int k = 0; k < dim; k++)
				for (int l = k; l < dim; l++)
				{

					// std::cout << "5.3" << std::endl;
					spatial_stiffness_out.index({t_Slice(), i, j, k, l}) = dS_ij_dC.index({t_Slice(), k, l});
					spatial_stiffness_out.index({t_Slice(), j, i, k, l}) = dS_ij_dC.index({t_Slice(), k, l});
					spatial_stiffness_out.index({t_Slice(), j, i, l, k}) = dS_ij_dC.index({t_Slice(), k, l});
					spatial_stiffness_out.index({t_Slice(), i, j, l, k}) = dS_ij_dC.index({t_Slice(), k, l});
					// std::cout << "5.4" << std::endl;
				}
		}

	spatial_stiffness_out = torch::einsum("mIJKL,miI,mjJ,mkK,mlL->mijkl", {spatial_stiffness_out, F, F, F, F});

	kirchhoff_stress_out = kirchhoff_stress_out + H;
	kirchhoff_stress_out = torch::einsum("...ik,...kl,...jl->...ij", {F, kirchhoff_stress_out, F});

	strain_energy_out = strain_energy_out + W0;
}

template <int dim, typename Number>
void VectorizedMaterialDomain<dim, Number>::evaluate_model_from_F_opt(const torch::Tensor &F,
																	  const torch::Tensor &structural_tensors,
																	  torch::Tensor &strain_energy_out,
																	  torch::Tensor &kirchhoff_stress_out,
																	  torch::Tensor &spatial_stiffness_out)
{
	torch::NoGradGuard no_grad;

	at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
	auto result = module.run_method("psi_tau_cc_from_F", F, structural_tensors).toTuple();
	strain_energy_out = result->elements()[0].toTensor().to(options);
	kirchhoff_stress_out = result->elements()[1].toTensor().to(options);
	spatial_stiffness_out = result->elements()[2].toTensor().to(options);
}

} // namespace commet_solve

#endif // INCLUDE_NCM_DOMAIN_VECTORIZED_DOMAIN_HPP_
