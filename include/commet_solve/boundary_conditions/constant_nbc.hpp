#ifndef INCLUDE_BOUNDARY_CONDITIONS_CONSTANT_NBC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_CONSTANT_NBC_HPP_
 
#include "neumann_bc.hpp"

namespace commet_solve{

template <int dim, typename Number = double>
class ConstNeumannBC : public NeumannBC<dim, Number>
{
  public:
	ConstNeumannBC(const unsigned int &boundary_id,
				   const Tensor<1, dim, Number> &end_value,
				   const Number &end_time,
				   const Tensor<1, dim, Number> &start_value = Tensor<1, dim, Number>(),
				   const Number &start_time = 0)
		: NeumannBC<dim, Number>(boundary_id)
		, end_value(end_value)
		, end_time(end_time)
		, start_value(start_value)
		, start_time(start_time) {};
	ConstNeumannBC(ConstNeumannBC &&) = delete;
	ConstNeumannBC(const ConstNeumannBC &) = delete;
	ConstNeumannBC &operator=(ConstNeumannBC &&) = delete;
	ConstNeumannBC &operator=(const ConstNeumannBC &) = delete;
	~ConstNeumannBC() = default;

	void apply(const LA::MPI::Vector &locally_relevant_u,
			   const AffineConstraints<Number> &constraints,
			   LA::MPI::SparseMatrix &system_matrix,
			   LA::MPI::Vector &system_rhs,
			   const Number &time,
			   const Number &dt) const override;

  private:
	const Tensor<1, dim, Number> end_value;
	const Number end_time;
	const Tensor<1, dim, Number> start_value;
	const Number start_time;
};

template <int dim, typename Number>
void ConstNeumannBC<dim, Number>::apply(const LA::MPI::Vector &/*locally_relevant_u*/,
										const AffineConstraints<Number> &constraints,
										LA::MPI::SparseMatrix &/*system_matrix*/,
										LA::MPI::Vector &system_rhs,
										const Number &time,
										const Number &/*dt*/) const
{

	Tensor<1, dim, Number> current_value =
		start_value + (time - start_time) * (end_value - start_value) / (end_time - start_time);
    if(this->surface_elements.empty())
        return;

    Vector<Number> cell_rhs(this->surface_elements.at(0).n_nodes()*dim);

	for (const SurfaceElement<dim, Number> &surf_el : this->surface_elements){

		// surf_el.add_to_residual(current_value, system_rhs);
		surf_el.add_to_residual(current_value, constraints, system_rhs, cell_rhs);
    }
		// surf_el.add_to_residual(current_value, r);
}



}

#endif  // INCLUDE_BOUNDARY_CONDITIONS_CONSTANT_NBC_HPP_
