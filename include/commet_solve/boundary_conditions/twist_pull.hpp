#ifndef INCLUDE_BOUNDARY_CONDITIONS_TWIST_PULL_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_TWIST_PULL_HPP_

#include "dirichlet_bc.hpp"
#include <cmath>
#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>


namespace commet_solve{



using namespace dealii;
using namespace dealii::Functions;
using namespace dealii::VectorTools;
using namespace std;

template <typename Number = double>
inline Number square(const Number &v)
{
	return v * v;
}

template <int dim, typename Number = double>
class RotateFunction : public TensorFunction<1, dim, Number>
{
  public:
	RotateFunction(const Point<dim> &centre,
				   const Tensor<1, dim, Number> &axis,
				   const Number &end_angle,
				   const Number &end_u,
				   const Number &end_time,
				   const Number &start_angle = 0,
				   const Number &start_u = 0,
				   const Number &start_time = 0)
		: TensorFunction<1, dim, Number>(start_time)
		, centre(centre)
		, axis(axis)
		, end_angle(end_angle)
		, end_u(end_u)
		, end_time(end_time)
		, start_angle(start_angle)
		, start_u(start_u)
		, start_time(start_time)
	{

		Number h_xz = sqrt(square(axis[0]) + square(axis[2]));
		// Number h_yz = sqrt(square(axis[1])+square(axis[2]));
		Number h = axis.norm();

		const Number cos_alpha = axis[2] / h_xz;
		const Number sin_alpha = axis[0] / h_xz;

		const Number cos_beta = h_xz / h;
		const Number sin_beta = axis[1] / h;

		Tensor<2, dim, Number> R1;
		R1 = 0;
		R1[0][0] = cos_alpha;
		R1[2][2] = cos_alpha;
		R1[0][2] = sin_alpha;
		R1[2][0] = -sin_alpha;
		R1[1][1] = 1;

		Tensor<2, dim, Number> R2;
		R2 = 0;
		R2[1][1] = cos_beta;
		R2[2][2] = cos_beta;
		R2[1][2] = sin_beta;
		R2[2][1] = -sin_beta;
		R2[0][0] = 1;

		R = R2 * R1;
		R_T = transpose(R);
	};

	RotateFunction(RotateFunction &&) = delete;
	RotateFunction(const RotateFunction &) = delete;
	RotateFunction &operator=(RotateFunction &&) = delete;
	RotateFunction &operator=(const RotateFunction &) = delete;
	~RotateFunction() = default;

	// Tensor<1, dim, Number> value(const Point<dim> &p) const override
	// {
	// 	// const Point<dim> trans_p = p - centre;
	// 	const Number current_u = start_u + this->get_time() * (end_u - start_u) / (end_time - start_time);
	// 	const Tensor<1, dim, Number> trans_p = p - centre;
	// 	const Tensor<2, dim, Number> current_R = get_current_R();

	// 	Tensor<1, dim, Number> end_p = current_R * trans_p;
	// 	end_p += centre;
	// 	end_p += current_u * axis / axis.norm();
	// 	return end_p - p;
	// };

	Tensor<1, dim, Number> value(const Point<dim> &p) const override
	{
		Tensor<1, dim, Number> current_u = get_u(p, this->get_time());
		Tensor<1, dim, Number> prev_u = get_u(p, this->get_time() - this->d_time);
		return current_u - prev_u;
	};

	Tensor<1, dim, Number> get_u(const Point<dim> &p, const Number &time) const
	{
		// const Point<dim> trans_p = p - centre;
		const Number current_u = start_u + time * (end_u - start_u) / (end_time - start_time);
		const Tensor<1, dim, Number> trans_p = p - centre;
		const Tensor<2, dim, Number> current_R = get_current_R(time);

		Tensor<1, dim, Number> end_p = current_R * trans_p;
		end_p += centre;
		end_p += current_u * axis / axis.norm();
		return end_p - p;
	};

	void set_d_time(const Number &new_d_time)
	{
		d_time = new_d_time;
	};

	const Point<dim> centre;
	const Tensor<1, dim, Number> axis;
	const Number end_angle;
	const Number end_u;
	const Number end_time;
	const Number start_angle;
	const Number start_u;
	const Number start_time;

  private:
	Number d_time;
	Tensor<2, dim, Number> R;
	Tensor<2, dim, Number> R_T;

	inline Tensor<2, dim, Number> get_current_R(const Number &time) const
	{
		const Number current_angle = start_angle + time * (end_angle - start_angle) / (end_time - start_time);

		const Number cos_angle = cos(current_angle);
		const Number sin_angle = sin(current_angle);

		Tensor<2, dim, Number> ref_R;
		ref_R = 0;
		ref_R[2][2] = 1;

		ref_R[0][0] = cos_angle;
		ref_R[1][1] = cos_angle;
		ref_R[0][1] = sin_angle;
		ref_R[1][0] = -sin_angle;
		return R_T * ref_R * R;
	};
};

template <int dim, typename Number = double>
class RotateFunctionComponent : public Function<dim, Number>
{
  public:
	RotateFunctionComponent(const Point<dim> &centre,
							const Tensor<1, dim, Number> &axis,
							const Number &end_angle,
							const Number &end_u,
							const Number &end_time,
							const Number &start_angle = 0,
							const Number &start_u = 0,
							const Number &start_time = 0)
		: Function<dim, Number>(dim, start_time)
		, rotate_function(centre, axis, end_angle, end_u, end_time, start_angle, start_u, start_time) {};

	RotateFunctionComponent(RotateFunctionComponent &&) = delete;
	RotateFunctionComponent(const RotateFunctionComponent &) = delete;
	RotateFunctionComponent &operator=(RotateFunctionComponent &&) = delete;
	RotateFunctionComponent &operator=(const RotateFunctionComponent &) = delete;
	~RotateFunctionComponent() = default;

	void set_time(const dealii::numbers::NumberTraits<double>::real_type new_time) override
	{
		Function<dim>::set_time(new_time);
		rotate_function.set_time(new_time);
	};

	void set_d_time(const Number &new_d_time)
	{
		rotate_function.set_d_time(new_d_time);
	};

	void vector_value(const Point<dim> &p, Vector<Number> &value) const override
	{
		Tensor<1, dim, Number> val = rotate_function.value(p);
		for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
			value[i_comp] = val[i_comp];
	};

  private:
	RotateFunction<dim, Number> rotate_function;
};

template <int dim, typename Number = double>
class RotateBoundaryCondition : public DirichletBC<dim>
{
  public:
	RotateBoundaryCondition(const unsigned int &bc_id,
							const Point<dim> &centre,
							const Tensor<1, dim, Number> &axis,
							const Number &end_angle,
							const Number &end_u,
							const Number &end_time,
							const Number &start_angle = 0,
							const Number &start_u = 0,
							const Number &start_time = 0);
	RotateBoundaryCondition(RotateBoundaryCondition &&) = delete;
	RotateBoundaryCondition(const RotateBoundaryCondition &) = delete;
	RotateBoundaryCondition &operator=(RotateBoundaryCondition &&) = delete;
	RotateBoundaryCondition &operator=(const RotateBoundaryCondition &) = delete;
	~RotateBoundaryCondition() = default;

	void apply(const DoFHandler<dim> &dof_handler,
								  AffineConstraints<Number> &constraints,
								  const Number &time,
								  const Number &d_time) override;

	const unsigned int boundary_id;
	const Point<dim> centre;
	const Tensor<1, dim, Number> axis;
	const Number end_angle;
	const Number end_time;
	const Number start_angle;
	const Number start_time;
	const Number start_u;
	const Number end_u;

  private:
	// RotateFunction<dim, Number> rotate_function;
	RotateFunctionComponent<dim, Number> rotate_function;
};

template <int dim, typename Number>
RotateBoundaryCondition<dim, Number>::RotateBoundaryCondition(const unsigned int &bc_id,
															  const Point<dim> &centre,
															  const Tensor<1, dim, Number> &axis,
															  const Number &end_angle,
															  const Number &end_u,
															  const Number &end_time,
															  const Number &start_angle,
															  const Number &start_u,
															  const Number &start_time)
	: DirichletBC<dim, Number>(bc_id, {0, 1, 2})
	, boundary_id(bc_id)
	, centre(centre)
	, axis(axis)
	, end_angle(end_angle)
	, end_time(end_time)
	, start_angle(start_angle)
	, start_time(start_time)
	, start_u(start_u)
	, end_u(end_u)
	, rotate_function(centre, axis, end_angle, end_u, end_time, start_angle, start_u, start_time)
{
}

template <int dim, typename Number>
void RotateBoundaryCondition<dim, Number>::apply(const DoFHandler<dim> &dof_handler,
																	AffineConstraints<Number> &constraints,
																	const Number &time,
																	const Number &d_time)
{
	ComponentMask prescribed_indices(dim, true);
	// prescribed_indices.set(2, false);
	rotate_function.set_time(time);
	rotate_function.set_d_time(d_time);
	// interpolate_boundary_values(dof_handler, this->boundary_id, rotate_function, constraints, prescribed_indices);
	interpolate_boundary_values(dof_handler, this->boundary_id, rotate_function, constraints);
}



}

#endif  // INCLUDE_BOUNDARY_CONDITIONS_TWIST_PULL_HPP_
