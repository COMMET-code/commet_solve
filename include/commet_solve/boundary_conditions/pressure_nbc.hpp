#ifndef INCLUDE_BOUNDARY_CONDITIONS_PRESSURE_NBC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_PRESSURE_NBC_HPP_

#include "neumann_bc.hpp"
#include "../time_functions/time_functions.hpp"

namespace commet_solve
{

template <int dim, typename Number = double>
// class PressureNeumannBC {
class PressureNeumannBC : public NeumannBC<dim, Number>
{
  public:
	// PressureNeumannBC() = default;
	PressureNeumannBC(const unsigned int &boundary_id,
					  const Number &end_value,
					  const Number &end_time,
					  const Number &start_value = 0,
					  const Number &start_time = 0)

		: NeumannBC<dim, Number>(boundary_id)
		, mag_func(make_unique<LinearRamp<Number>>(end_value, end_time, start_value, start_time)) {};
	PressureNeumannBC(const unsigned int &boundary_id,
					  // unique_ptr<TimeFunction<Number>>& mag)
					  unique_ptr<TimeFunction<Number>> mag)

		: NeumannBC<dim, Number>(boundary_id)
		, mag_func(move(mag)) {};
	// , mag_func(new LinearRamp<Number>(end_value, end_time, start_value, start_time)){};
	// , end_value(end_value)
	// , end_time(end_time)
	// , start_value(start_value)
	// , start_time(start_time) {};
	PressureNeumannBC(PressureNeumannBC &&) = delete;
	PressureNeumannBC(const PressureNeumannBC &) = delete;
	PressureNeumannBC &operator=(PressureNeumannBC &&) = delete;
	PressureNeumannBC &operator=(const PressureNeumannBC &) = delete;
	~PressureNeumannBC() = default;

	void apply(const LA::MPI::Vector &locally_relevant_u,
			   const AffineConstraints<Number> &constraints,
			   LA::MPI::SparseMatrix &system_matrix,
			   LA::MPI::Vector &system_rhs,
			   const Number &time,
			   const Number &dt) const override;

  private:
	// const Tensor<1, dim, Number> end_value;
	unique_ptr<TimeFunction<Number>> mag_func;

	// const Number end_value;
	// const Number end_time;
	// const Number start_value;
	// const Number start_time;
};

template <int dim, typename Number>
void PressureNeumannBC<dim, Number>::apply(const LA::MPI::Vector & /*locally_relevant_u*/,
										   const AffineConstraints<Number> &constraints,
										   LA::MPI::SparseMatrix & /*system_matrix*/,
										   LA::MPI::Vector &system_rhs,
										   const Number &time,
										   const Number & /*dt*/) const
{

	// Tensor<1, dim, Number> current_value =
	// 	start_value + (time - start_time) * (end_value - start_value) / (end_time - start_time);
	Number current_value = mag_func->get_val(time);
	if (this->surface_elements.empty())
		return;

	Vector<Number> cell_rhs(this->surface_elements.at(0).n_nodes() * dim);

	// start_value + (time - start_time) * (end_value - start_value) / (end_time - start_time);

	// for(const SurfaceElement<dim, Number> & surf_el: this->surface_elements)
	// surf_el.add_to_residual(current_value, r);

	for (const SurfaceElement<dim, Number> &surf_el : this->surface_elements)
	{
		vector<Tensor<1, dim, Number>> qp_vals(surf_el.n_qps());
		const vector<Tensor<1, dim, Number>> &normals = surf_el.get_normals();
		for (unsigned int i = 0; i < surf_el.n_qps(); i++)
		{
			qp_vals[i] = normals[i] * current_value;
		}

		// surf_el.add_to_residual(qp_vals, r);
		surf_el.add_to_residual(qp_vals, constraints, system_rhs, cell_rhs);
	}
}

} // namespace commet_solve

#endif // INCLUDE_BOUNDARY_CONDITIONS_PRESSURE_NBC_HPP_
