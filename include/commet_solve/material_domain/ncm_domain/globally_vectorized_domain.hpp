#ifndef INCLUDE_NCM_DOMAIN_GLOBALLY_VECTORIZED_DOMAIN_HPP_
#define INCLUDE_NCM_DOMAIN_GLOBALLY_VECTORIZED_DOMAIN_HPP_

#include "tensor_conversion_utils.hpp"
#include "vectorized_domain.hpp"

namespace commet_solve
{

template <int dim, typename Number = double>
class GloballyVectorizedDomain : public VectorizedMaterialDomain<dim, Number>
{
  public:
	GloballyVectorizedDomain() = default;
	GloballyVectorizedDomain(GloballyVectorizedDomain &&) = delete;
	GloballyVectorizedDomain(const GloballyVectorizedDomain &) = delete;
	GloballyVectorizedDomain &operator=(GloballyVectorizedDomain &&) = delete;
	GloballyVectorizedDomain &operator=(const GloballyVectorizedDomain &) = delete;
	~GloballyVectorizedDomain() = default;

	void compute_constitutive_behaviour() override
	{
        if(!F.is_contiguous())
            std::cout << "F is not contiguous!!!" << std::endl;
        F = F.contiguous();
		unsigned int count = 0;
		for (auto &[point_key, point_data] : this->qp_data)
		{
			deal_to_torch_tensor<dim, Number>(count, point_data.F, F, TensorLayout::STANDARD);

			count++;
		}

		this->evaluate_model(F, structural_vectors, energy, tau, cc); 
        if(!tau.is_contiguous())
            std::cout << "tau is not contiguous!!!" << std::endl;
        if(!cc.is_contiguous())
            std::cout << "cc is not contiguous!!!" << std::endl;
        tau = tau.contiguous();
        cc = cc.contiguous();

		TensorLayout layout = determine_tensor_layout(tau);
		count = 0;
		for (auto &[point_key, point_data] : this->qp_data)
		{
			torch_to_deal_tensor<dim, Number>(count, tau, point_data.tau, layout);
			torch_to_deal_tensor<dim, Number>(count, cc, point_data.cc, layout);
			count++;
		}
	};

	void close() override
	{

		const at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
        const int64_t size = static_cast<int64_t>(this->qp_data.size());
		F = torch::zeros({size, dim, dim}, options);
		structural_vectors = torch::zeros({size, 0, dim}, options);

		energy = torch::zeros({size}, options);

		switch (this->get_return_tensor_layout())
		{
		case TensorLayout::STANDARD: {
			tau = torch::zeros({size, dim, dim}, options);
			cc = torch::zeros({size, dim, dim, dim, dim}, options);
			break;
		}
		case TensorLayout::VOIGT: {
			tau = torch::zeros({size, 6}, options);
			cc = torch::zeros({size, 6, 6}, options);
			break;
		}
		}

		this->open_or_closed = MaterialDomainState::CLOSED;
	};

  protected:
	torch::Tensor F;
	torch::Tensor structural_vectors;

	torch::Tensor energy;
	torch::Tensor tau;
	torch::Tensor cc;

  private:
};

} // namespace commet_solve

#endif // INCLUDE_NCM_DOMAIN_GLOBALLY_VECTORIZED_DOMAIN_HPP_
