#ifndef INCLUDE_MATERIAL_DOMAIN_MATERIAL_DOMAIN_HPP_
#define INCLUDE_MATERIAL_DOMAIN_MATERIAL_DOMAIN_HPP_

#include <cmath>
#include <deal.II/fe/fe_values.h>
#include <unordered_map>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <ranges>
#include <span>

#include "../types.hpp"
#include "../utilities.hpp"
#include "commet_solve/output/output_flags.hpp"

#include "../field/scalar_field.hpp"
#include "../field/tensor_field.hpp"
#include "../field/vector_field.hpp"
#include "../field/vfield_manager.hpp"

namespace commet_solve
{

using namespace dealii;


// template <int dim, typename Number = double>
// struct MaterialPointData
// {
// 	unsigned int el_id;
// 	unsigned int qp;
//     std::array<Tensor<1, dim, Number>, N_ORIENTATION_VECS> orientation_vectors;
// 	Point<dim, Number> position;
// };


// template <int dim, typename Number = double>
// struct HyperelasticMaterialPointData : MaterialPointData<dim, Number>
// {
// 	Number psi;
// 	Tensor<2, dim, Number> F;
// 	SymmetricTensor<2, dim, Number> tau;
// 	SymmetricTensor<4, dim, Number> cc;
// };

enum class MaterialDomainState
{
	OPEN,
	CLOSED
};

/**
 * @brief This holds all the data for the material points of a given domain.
 * It also manages any updating of those material points (ie constitutive behaviour)
 */
template <int dim, typename Number = double>
class MaterialDomain
{
  public:
	MaterialDomain()
		: open_or_closed(MaterialDomainState::OPEN) {};
	MaterialDomain(MaterialDomain &&) = delete;
	MaterialDomain(const MaterialDomain &) = delete;
	MaterialDomain &operator=(MaterialDomain &&) = delete;
	MaterialDomain &operator=(const MaterialDomain &) = delete;
	~MaterialDomain() = default;

	void set_dt(const Number &new_dt)
	{
		dt = new_dt;
	};
	Number get_dt()
	{
		return dt;
	};

	virtual void add_entry(const dealii::types::global_cell_index &cell, const unsigned int &qp) = 0;

	virtual void add_entry(const dealii::types::global_cell_index &cell_idx,
						   const DoFCellAccessor<dim, dim, false> &cell,
						   const FEValues<dim, dim> &fe_values,
						    ScalarFieldManager<dim, Number> &scalar_fields,
						    VectorFieldManager<dim, Number> &vector_fields,
						    TensorFieldManager<dim, Number> &tensor_fields) = 0;

	virtual void update_F(const dealii::types::global_cell_index &cell,
						  const unsigned int &qp,
						  const Tensor<2, dim, Number> &F) = 0;

	virtual void get_vals(const dealii::types::global_cell_index &cell,
						  const unsigned int &qp,
						  Tensor<2, dim, Number> &F,
						  Number &psi,
						  SymmetricTensor<2, dim, Number> &tau,
						  SymmetricTensor<4, dim, Number> &cc) = 0;

	virtual std::span<const Tensor<1, dim, Number>> 
    get_structural_vectors(const dealii::types::global_cell_index &cell, const unsigned int &qp) {
	        return {};
    };


	virtual Number get_scalar_value(const dealii::types::global_cell_index & /*cell*/,
									const unsigned int & /*qp*/,
									const scalar_output_flag & /*flag*/)
	{
		return NAN;
	};

	virtual Tensor<1, dim, Number> get_vector_value(const dealii::types::global_cell_index & /*cell*/,
													const unsigned int & /*qp*/,
													const vector_output_flag & /*flag*/)
	{
		return Tensor<1, dim, Number>({NAN, NAN, NAN});
	};

	virtual Tensor<2, dim, Number> get_tensor_value(const dealii::types::global_cell_index & /*cell*/,
													const unsigned int & /*qp*/,
													const tensor_output_flag & /*flag*/)
	{
		return Tensor<2, dim, Number>({{NAN, NAN, NAN}, {NAN, NAN, NAN}, {NAN, NAN, NAN}});
	};

	virtual void compute_constitutive_behaviour() = 0;

	virtual void close() {};

  protected:
	Number dt;
	MaterialDomainState open_or_closed;

  private:
};

} // namespace commet_solve

#endif // INCLUDE_MATERIAL_DOMAIN_MATERIAL_DOMAIN_HPP_
