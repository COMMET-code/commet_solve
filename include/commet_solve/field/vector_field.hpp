#ifndef INCLUDE_FIELD_VECTOR_FIELD_HPP_
#define INCLUDE_FIELD_VECTOR_FIELD_HPP_

#include "../config.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/vector.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>

#include "exprtk.hpp"

namespace commet_solve
{
using namespace dealii;
using namespace std;

template <int dim, typename Number = double>
class VectorField
{
  public:
	VectorField() = default;
	VectorField(VectorField &&) = delete;
	VectorField(const VectorField &) = delete;
	VectorField &operator=(VectorField &&) = delete;
	VectorField &operator=(const VectorField &) = delete;
	~VectorField() = default;

	virtual void evaluate_field(const DoFCellAccessor<dim, dim, false> &cell,
								std::vector<Tensor<1, dim, Number>> &qp_values,
								const std::string &field_name) = 0;

	virtual void add_field(const std::string &name,
						   const std::vector<Tensor<1, dim, Number>> &field,
						   const std::vector<types::coarse_cell_id> &coarse_cell_index_to_coarse_cell_id) = 0;

	virtual void ouput_field(const std::string &name, DataOut<dim> &data_out) = 0;

  private:
};

template <int dim, typename Number = double>
class ElementWiseVectorField : public VectorField<dim, Number>
{
  public:
	ElementWiseVectorField(const Triangulation<dim, dim> *tri,
						   const hp::MappingCollection<dim> &mapping,
						   const hp::QCollection<dim> &quadrature_formula);
	ElementWiseVectorField(ElementWiseVectorField &&) = delete;
	ElementWiseVectorField(const ElementWiseVectorField &) = delete;
	ElementWiseVectorField &operator=(ElementWiseVectorField &&) = delete;
	ElementWiseVectorField &operator=(const ElementWiseVectorField &) = delete;
	~ElementWiseVectorField() = default;

	void evaluate_field(const DoFCellAccessor<dim, dim, false> &cell,
						std::vector<Tensor<1, dim, Number>> &qp_values,
						const std::string &field_name) override
	{
		const auto &field_cell = cell.as_dof_handler_iterator(df);
		fe_values.reinit(field_cell);
		const auto &present_fe_values = fe_values.get_present_fe_values();

		qp_values.resize(present_fe_values.n_quadrature_points);
		std::vector<Vector<Number>> tmp_vals(present_fe_values.n_quadrature_points, Vector<Number>(dim));

		present_fe_values.get_function_values(fields[field_name], tmp_vals);
		for (unsigned int qp = 0; qp < qp_values.size(); qp++)
			for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
				qp_values.at(qp)[i_comp] = tmp_vals.at(qp)(i_comp);
	};

	void add_field(const std::string &name,
				   const std::vector<Tensor<1, dim, Number>> &field,
				   const std::vector<types::coarse_cell_id> &coarse_cell_index_to_coarse_cell_id) override
	{
		fields[name] = LA::MPI::Vector(df.locally_owned_dofs(), MPI_COMM_WORLD);
		std::vector<types::global_dof_index> dof_indices;

		for (const auto &cell : df.active_cell_iterators())
			if (cell->is_locally_owned())
			{
				types::global_cell_index c_id = coarse_cell_index_to_coarse_cell_id.at(cell->active_cell_index());

				dof_indices.resize(fe[cell->active_fe_index()].n_dofs_per_cell());
				cell->get_dof_indices(dof_indices);
				const Tensor<1, dim, Number> &tensor = field.at(c_id);
				for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
				{
					fields[name][dof_indices.at(i_comp)] = tensor[i_comp];
				}
			}
	};

	void ouput_field(const std::string &name, DataOut<dim> &data_out) override
	{

		std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(
			dim, DataComponentInterpretation::component_is_part_of_vector);
		data_out.add_data_vector(df, fields[name], name, interpretation);
	};

	DoFHandler<dim, dim> df;
	hp::FECollection<dim, dim> fe;
	hp::FEValues<dim> fe_values;
	std::map<std::string, LA::MPI::Vector> fields;

  private:
};

template <int dim, typename Number>
ElementWiseVectorField<dim, Number>::ElementWiseVectorField(const Triangulation<dim, dim> *tri,
															const hp::MappingCollection<dim> &mapping,
															const hp::QCollection<dim> &quadrature_formula)
	: VectorField<dim, Number>()
	, df(*tri)
	, fe(FESystem<dim, dim>(FE_SimplexDGP<dim>(0), dim), FESystem<dim, dim>(FE_DGQ<dim>(0), dim))
	, fe_values(mapping, fe, quadrature_formula, update_values)
{
	for (const auto &cell : df.active_cell_iterators())
		if (cell->is_locally_owned())
		{
			if (cell->reference_cell() == ReferenceCells::Tetrahedron)
			{
				cell->set_active_fe_index(0);
			}
			else if (cell->reference_cell() == ReferenceCells::Hexahedron)
			{
				cell->set_active_fe_index(1);
			}
			else
				DEAL_II_NOT_IMPLEMENTED();
		}

	df.distribute_dofs(fe);
}

template <int dim, typename Number = double>
class AnalyticalVectorField : public VectorField<dim, Number>
{
  public:
	AnalyticalVectorField(const string &x_expression,
						  const string &y_expression,
						  const string &z_expression,
						  const Triangulation<dim, dim> *tri,
						  const hp::MappingCollection<dim> &mapping,
						  const hp::QCollection<dim> &quadrature_formula)
		: VectorField<dim, Number>()

		, df(*tri)
		, fe(FESystem<dim, dim>(FE_SimplexP<dim>(1), dim), FESystem<dim, dim>(FE_Q<dim>(1), dim))
		, fe_values(mapping, fe, quadrature_formula, update_values)
	{

		for (const auto &cell : df.active_cell_iterators())
			if (cell->is_locally_owned())
			{
				if (cell->reference_cell() == ReferenceCells::Tetrahedron)
				{
					cell->set_active_fe_index(0);
				}
				else if (cell->reference_cell() == ReferenceCells::Hexahedron)
				{
					cell->set_active_fe_index(1);
				}
				else
					DEAL_II_NOT_IMPLEMENTED();
			}

		df.distribute_dofs(fe);

		symbol_table.add_variable("x", x);
		symbol_table.add_variable("y", y);
		symbol_table.add_variable("z", z);
		symbol_table.add_constants();

		expression_x.register_symbol_table(symbol_table);
		expression_y.register_symbol_table(symbol_table);
		expression_z.register_symbol_table(symbol_table);

		exprtk::parser<Number> parser;
		parser.compile(x_expression, expression_x);
		parser.compile(y_expression, expression_y);
		parser.compile(z_expression, expression_z);

		nodal_values.reinit(df.locally_owned_dofs(), MPI_COMM_WORLD);
		std::vector<Point<dim>> support_points(df.n_dofs());
		DoFTools::map_dofs_to_support_points(mapping, df, support_points);
		Tensor<1, dim, Number> working;

		for (const auto &cell : df.active_cell_iterators())
		{
			if (cell->is_locally_owned()) // For MPI
			{
				const FiniteElement<dim> &present_fe = cell->get_fe();
				const unsigned int dofs_per_cell = present_fe.dofs_per_cell;
				std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

				cell->get_dof_indices(local_dof_indices);

				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
					unsigned int global_dof = local_dof_indices[i];
					unsigned int comp = present_fe.system_to_component_index(i).first;

					const Point<dim> &pt = support_points[global_dof];

					this->evaluate_field_at_point(pt, working);
					nodal_values(global_dof) = working[comp];
				}
			}
		}
	};
	AnalyticalVectorField(AnalyticalVectorField &&) = delete;
	AnalyticalVectorField(const AnalyticalVectorField &) = delete;
	AnalyticalVectorField &operator=(AnalyticalVectorField &&) = delete;
	AnalyticalVectorField &operator=(const AnalyticalVectorField &) = delete;
	~AnalyticalVectorField() = default;

	void evaluate_field(const DoFCellAccessor<dim, dim, false> &cell,
						std::vector<Tensor<1, dim, Number>> &qp_values,
						const std::string & /*field_name*/) override
	{

		const auto &field_cell = cell.as_dof_handler_iterator(df);
		fe_values.reinit(field_cell);
		const auto &present_fe_values = fe_values.get_present_fe_values();

		qp_values.resize(present_fe_values.n_quadrature_points);

		for (unsigned int qp = 0; qp < qp_values.size(); qp++)
			this->evaluate_field_at_point(present_fe_values.quadrature_point(qp), qp_values.at(qp));
	};

	void evaluate_field_at_point(const Point<dim> &p, Tensor<1, dim, Number> &t)
	{
		x = p[0];
		y = p[1];
		z = p[2];
		t[0] = expression_x.value();
		t[1] = expression_y.value();
		t[2] = expression_z.value();
	}

	void add_field(const std::string & /*name*/,
				   const std::vector<Tensor<1, dim, Number>> & /*field*/,
				   const std::vector<types::coarse_cell_id> & /*coarse_cell_index_to_coarse_cell_id*/) override {

	};

	void ouput_field(const std::string &name, DataOut<dim> &data_out) override
	{

		std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(
			dim, DataComponentInterpretation::component_is_part_of_vector);
		data_out.add_data_vector(df, nodal_values, name, interpretation);
	};

  private:
	DoFHandler<dim, dim> df;
	hp::FECollection<dim, dim> fe;
	hp::FEValues<dim> fe_values;
	LA::MPI::Vector nodal_values;

	exprtk::symbol_table<Number> symbol_table;
	exprtk::expression<Number> expression_x;
	exprtk::expression<Number> expression_y;
	exprtk::expression<Number> expression_z;

	Number x;
	Number y;
	Number z;
};

} // namespace commet_solve

#endif // INCLUDE_FIELD_VECTOR_FIELD_HPP_
