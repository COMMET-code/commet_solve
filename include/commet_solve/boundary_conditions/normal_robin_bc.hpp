#ifndef INCLUDE_BOUNDARY_CONDITIONS_NORMAL_ROBIN_BC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_NORMAL_ROBIN_BC_HPP_


#include "neumann_bc.hpp"
#include <deal.II/base/tensor.h>

namespace commet_solve
{

template <int dim, typename Number = double>
class NormalRobinBC : public NeumannBC<dim, Number>
{
  public:
	// NormalRobinBC() = default;
	NormalRobinBC(const unsigned int &boundary_id, const Number &alpha)
		: NeumannBC<dim, Number>(boundary_id)
    , alpha(alpha)
	{ }

	NormalRobinBC(NormalRobinBC &&) = delete;
	NormalRobinBC(const NormalRobinBC &) = delete;
	NormalRobinBC &operator=(NormalRobinBC &&) = delete;
	NormalRobinBC &operator=(const NormalRobinBC &) = delete;
	~NormalRobinBC() = default;

	void apply(const LA::MPI::Vector &locally_relevant_u,
			   const AffineConstraints<Number> &constraints,
			   LA::MPI::SparseMatrix &system_matrix,
			   LA::MPI::Vector &system_rhs,
			   const Number &time,
			   const Number &dt) const override;

  private:
	Number alpha;
};

template <int dim, typename Number>
void NormalRobinBC<dim, Number>::apply(const LA::MPI::Vector &locally_relevant_u,
								 const AffineConstraints<Number> &constraints,
								 LA::MPI::SparseMatrix &system_matrix,
								 LA::MPI::Vector &system_rhs,
								 const Number & /*time*/,
								 const Number & /*dt*/) const
{

	if (this->surface_elements.empty())
		return;

	vector<Tensor<1, dim, Number>> r_qp_vals(this->surface_elements.at(0).n_qps());
	vector<SymmetricTensor<2, dim, Number>> k_qp_vals(this->surface_elements.at(0).n_qps());

	FullMatrix<Number> cell_matrix(this->surface_elements.at(0).n_nodes() * dim,
								   this->surface_elements.at(0).n_nodes() * dim);
	Vector<Number> cell_rhs(this->surface_elements.at(0).n_nodes() * dim);

	for (const SurfaceElement<dim, Number> &surf_el : this->surface_elements)
	{

		// LOGGER.info("Get qp u's");
		const vector<Tensor<1, dim, Number>> u_at_qps = surf_el.get_u_at_qps(locally_relevant_u); 
        const vector<Tensor<1, dim, Number>> & normals = surf_el.get_normals();
		// LOGGER.info("Get qp r's");
		for (unsigned int qp = 0; qp < surf_el.n_qps(); qp++)
		{
			k_qp_vals.at(qp) = this->alpha*symmetrize(outer_product(normals.at(qp), normals.at(qp)));
			r_qp_vals.at(qp) = k_qp_vals.at(qp) * u_at_qps.at(qp);
		}

		// LOGGER.info("Add qp r's");
		surf_el.add_to_residual_and_stiffness(
			constraints, r_qp_vals, k_qp_vals, cell_matrix, cell_rhs, system_rhs, system_matrix);
	}
}

} // namespace commet_solve

#endif  // INCLUDE_BOUNDARY_CONDITIONS_NORMAL_ROBIN_BC_HPP_
