#ifndef INCLUDE_BOUNDARY_CONDITIONS_FULLY_DEFINED_DBC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_FULLY_DEFINED_DBC_HPP_

#include "commet_solve/time_functions/time_functions.hpp"
#include "dirichlet_bc.hpp"

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/vector_tools.h>

namespace commet_solve
{

using namespace dealii;
using namespace dealii::Functions;
using namespace dealii::VectorTools;
using namespace std;

template <int dim, typename Number = double>
class FullyDefinedDBC : public DirichletBC<dim, Number>
{
  public:

	FullyDefinedDBC(const unsigned int &bc_id,
					const std::vector<unsigned int> &components,
					const std::vector<Number> &end_values,
					const Number &end_time = 1
                 );

	FullyDefinedDBC(const unsigned int &bc_id,
					const std::vector<unsigned int> &components,
					const std::vector<Number> &end_values,
					unique_ptr<TimeFunction<Number, Number>> time_func);

	FullyDefinedDBC(FullyDefinedDBC &&) = delete;
	FullyDefinedDBC(const FullyDefinedDBC &) = delete;
	FullyDefinedDBC &operator=(FullyDefinedDBC &&) = delete;
	FullyDefinedDBC &operator=(const FullyDefinedDBC &) = delete;
	~FullyDefinedDBC();

	void apply(const DoFHandler<dim> &dof_handler,
			   AffineConstraints<Number> &constraints,
			   const Number &time,
			   const Number &d_time) override;

	const unsigned int n_components;
	const std::vector<Number> end_values;
	const Number end_time;

  private:
	unique_ptr<TimeFunction<Number, Number>> time_func;
};

template <int dim, typename Number>
FullyDefinedDBC<dim, Number>::FullyDefinedDBC(const unsigned int &bc_id,
											  const std::vector<unsigned int> &components,
											  const std::vector<Number> &end_values,
											  const Number &end_time)
	: DirichletBC<dim, Number>(bc_id, components)
	, n_components(components.size())
	, end_values(end_values)
	, end_time(end_time)
	, time_func(make_unique<LinearRamp<Number, Number>>(/*end_value=*/1,
														end_time,
														/*start_value=*/0,
														/*start_time=*/0))
{
}

template <int dim, typename Number>
FullyDefinedDBC<dim, Number>::FullyDefinedDBC(const unsigned int &bc_id,
					const std::vector<unsigned int> &components,
					const std::vector<Number> &end_values,
					unique_ptr<TimeFunction<Number, Number>> time_func)
	: DirichletBC<dim, Number>(bc_id, components)
	, n_components(components.size())
	, end_values(end_values)
	, end_time(time_func->get_end_time())
    , time_func(move(time_func))
{
}

template <int dim, typename Number>
FullyDefinedDBC<dim, Number>::~FullyDefinedDBC()
{
}

template <int dim, typename Number>
void FullyDefinedDBC<dim, Number>::apply(const DoFHandler<dim> &dof_handler,
										 AffineConstraints<Number> &constraints,
										 const Number &time,
										 const Number &d_time)
{

	// Create ComponentMask for prescribed indices, all values initialised to
	// false
	ComponentMask prescribed_indices(dim, false);

	// Create vector for prescribed values, all values initialised to zero (for
	// use with ConstantFunction)
	std::vector<double> prescribed_values(dim, 0.);

	// Iterate over the vector of pairs of (index, value)
	for (unsigned int i = 0; i < n_components; i++)
	{
		const unsigned int component = this->components.at(i);
		prescribed_indices.set(component, true);
		prescribed_values.at(component) = end_values.at(i) * this->time_func->get_increment(time, d_time);
	}

	// Using the populated vector of prescribed_values, create a ConstantFunction
	// with default value zero
	ConstantFunction<dim> prescribed_constant_function(prescribed_values);

	// Apply prescribed displacement values to prescribed component indices
	interpolate_boundary_values(
		dof_handler, this->boundary_id, prescribed_constant_function, constraints, prescribed_indices);
}

} // namespace commet_solve

#endif // INCLUDE_BOUNDARY_CONDITIONS_FULLY_DEFINED_DBC_HPP_
