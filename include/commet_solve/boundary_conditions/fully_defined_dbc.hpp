#ifndef INCLUDE_BOUNDARY_CONDITIONS_FULLY_DEFINED_DBC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_FULLY_DEFINED_DBC_HPP_

#include "dirichlet_bc.hpp"



#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>



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
					const Number &end_time = 1);
	FullyDefinedDBC(FullyDefinedDBC &&) = delete;
	FullyDefinedDBC(const FullyDefinedDBC &) = delete;
	FullyDefinedDBC &operator=(FullyDefinedDBC &&) = delete;
	FullyDefinedDBC &operator=(const FullyDefinedDBC &) = delete;
	~FullyDefinedDBC();

	void apply(const DoFHandler<dim> &dof_handler,
			   AffineConstraints<Number> &constraints,
			   const Number &time,
			   const Number &d_time) override;

	const unsigned int boundary_id;
	const std::vector<unsigned int> components;
	const unsigned int n_components;
	const std::vector<Number> end_values;
	const Number end_time;

  private:
};

template <int dim, typename Number>
FullyDefinedDBC<dim, Number>::FullyDefinedDBC(const unsigned int &bc_id,
											  const std::vector<unsigned int> &components,
											  const std::vector<Number> &end_values,
											  const Number &end_time)
	: DirichletBC<dim, Number>()
	, boundary_id(bc_id)
	, components(components)
	, n_components(components.size())
	, end_values(end_values)
	, end_time(end_time)
{

	// assertm(n_components == end_values.size(),
	//         "The number of components must be the same as the "
	//         "number of end values provided but\n"
	//         "components.size()=" +
	//             to_string(components.size()) +
	//             " and end_values.size()=" + to_string(end_values.size()));
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
		const unsigned int component = components.at(i);
		prescribed_indices.set(component, true);
		prescribed_values.at(component) = time * end_values.at(i) / end_time;
		prescribed_values.at(component) -= (time - d_time) * end_values.at(i) / end_time;
	}

	// Using the populated vector of prescribed_values, create a ConstantFunction
	// with default value zero
	ConstantFunction<dim> prescribed_constant_function(prescribed_values);

	// Apply prescribed displacement values to prescribed component indices
	interpolate_boundary_values(
		dof_handler, this->boundary_id, prescribed_constant_function, constraints, prescribed_indices);
	// interpolate_boundary_values(
	// 	MappingFE<dim>(FE_SimplexP<dim>(1)), dof_handler, this->boundary_id, prescribed_constant_function, constraints,
	// prescribed_indices);
}


} // namespace commet_solve

#endif // INCLUDE_BOUNDARY_CONDITIONS_FULLY_DEFINED_DBC_HPP_
