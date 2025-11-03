#ifndef INCLUDE_NCM_DOMAIN_BATCH_VECTORIZED_DOMAIN_HPP_
#define INCLUDE_NCM_DOMAIN_BATCH_VECTORIZED_DOMAIN_HPP_

#include "vectorized_domain.hpp"
#include <algorithm>

namespace commet_solve
{

template <int dim, typename Number = double>
class BatchVectorizedDomain : public VectorizedMaterialDomain<dim, Number>
{
  public:
	BatchVectorizedDomain(const unsigned int &batch_size,
                       const unsigned int & n_structural_vectors=0)
		:VectorizedMaterialDomain<dim, Number>(n_structural_vectors)
        , batch_size(batch_size) {};

	BatchVectorizedDomain(BatchVectorizedDomain &&) = delete;
	BatchVectorizedDomain(const BatchVectorizedDomain &) = delete;
	BatchVectorizedDomain &operator=(BatchVectorizedDomain &&) = delete;
	BatchVectorizedDomain &operator=(const BatchVectorizedDomain &) = delete;
	~BatchVectorizedDomain() = default;

	void compute_constitutive_behaviour() override
	{
		const TensorLayout layout = this->get_return_tensor_layout();
		auto it_read = this->qp_data.begin();
		auto it_write = this->qp_data.begin();

		while (it_read != this->qp_data.end())
		{

            if(!F.is_contiguous())
                std::cout << "F is not contiguous!!!" << std::endl;
            F = F.contiguous();

			for (unsigned int i = 0; i < this->batch_size; i++)
			{
				deal_to_torch_tensor<dim, Number>(i, it_read->second.F, F, TensorLayout::STANDARD);

            if(this->n_structural_vectors> 0)
                mat_point_structural_vector_to_torch_tensor<dim, Number>(
                    i,
                    it_read->second.orientation_vectors,
                    this->n_structural_vectors,
                    structural_vectors);

				it_read++;
				if (it_read == this->qp_data.end())
					break;
			}

			this->evaluate_model(F, structural_vectors, energy, tau, cc);

            if(!tau.is_contiguous())
                std::cout << "tau is not contiguous!!!" << std::endl;
            if(!cc.is_contiguous())
                std::cout << "cc is not contiguous!!!" << std::endl;
            tau = tau.contiguous();
            cc = cc.contiguous();
			for (unsigned int i = 0; i < this->batch_size; i++)
			{
				torch_to_deal_tensor<dim, Number>(i, tau, it_write->second.tau, layout);
				torch_to_deal_tensor<dim, Number>(i, cc, it_write->second.cc, layout);
				it_write++;
				if (it_write == this->qp_data.end())
					break;
			}
		}
	};

	void close() override
	{

		const at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);


        int64_t size = static_cast<int64_t>(this->qp_data.size());
        size = std::min(size, static_cast<int64_t>(batch_size));

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

  private:
	unsigned int batch_size;
	torch::Tensor F;
	torch::Tensor structural_vectors;

	torch::Tensor energy;
	torch::Tensor tau;
	torch::Tensor cc;
};

} // namespace commet_solve

#endif // INCLUDE_NCM_DOMAIN_BATCH_VECTORIZED_DOMAIN_HPP_
