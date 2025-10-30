#ifndef INCLUDE_FIELD_VFIELD_MANAGER_HPP_
#define INCLUDE_FIELD_VFIELD_MANAGER_HPP_

#include "vector_field.hpp"
#include "../logger.hpp"
#include "../config.hpp"

namespace commet_solve
{

template <int dim, typename Number = double>
class VectorFieldManager
{
  public:
	VectorFieldManager() = default;
	VectorFieldManager(VectorFieldManager &&) = delete;
	VectorFieldManager(const VectorFieldManager &) = delete;
	VectorFieldManager &operator=(VectorFieldManager &&) = delete;
	VectorFieldManager &operator=(const VectorFieldManager &) = delete;
	~VectorFieldManager() = default;

	void evaluate_field(const DoFCellAccessor<dim, dim, false> &cell,
						std::vector<Tensor<1, dim, Number>> &qp_values,
						const std::string &field_name)
	{
        // DEBUG_MSG("Evaluating in field manager")
		// fields[field_name]->evaluate_field(cell, qp_values, field_name);
		fields.at(field_name)->evaluate_field(cell, qp_values, field_name);
	};

	void add_field(const Triangulation<dim, dim> *tri,
				   const hp::MappingCollection<dim> &mapping,
				   const hp::QCollection<dim> &quadrature_formula,
				   const std::string &name,
				   const std::vector<Tensor<1, dim, Number>> &field,
				   const std::vector<types::coarse_cell_id> &coarse_cell_index_to_coarse_cell_id)
	{

		fields[name] = std::make_unique<ElementWiseVectorField<dim, Number>>(tri, mapping, quadrature_formula);
		fields[name]->add_field(name, field, coarse_cell_index_to_coarse_cell_id);
	};

	void add_analytical_field(const Triangulation<dim, dim> *tri,
							  const hp::MappingCollection<dim> &mapping,
							  const hp::QCollection<dim> &quadrature_formula,
							  const std::string &name,
							  const std::string &x_expression,
							  const std::string &y_expression,
							  const std::string &z_expression)
	{

		fields[name] = std::make_unique<AnalyticalVectorField<dim, Number>>(
			x_expression, y_expression, z_expression, tri, mapping, quadrature_formula);
		// fields[name]->add_field(name, field, coarse_cell_index_to_coarse_cell_id);
	};

	virtual void ouput_fields(DataOut<dim> &data_out)
	{
		for (const auto &[name, field] : fields)
		{
			field->ouput_field(name, data_out);
		}
	};

  private:
	map<string, unique_ptr<VectorField<dim, Number>>> fields;
};

} // namespace commet_solve

#endif // INCLUDE_FIELD_VFIELD_MANAGER_HPP_
