#ifndef INCLUDE_BOUNDARY_CONDITIONS_ROBIN_BC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_ROBIN_BC_HPP_

#include "neumann_bc.hpp"

namespace commet_solve
{

template <int dim, typename Number = double>
class RobinBC : public NeumannBC<dim, Number>
{
  public:
	// RobinBC() = default;
	RobinBC(const unsigned int &boundary_id, const Number &k)
		: NeumannBC<dim, Number>(boundary_id)
	{
		for (unsigned int i = 0; i < dim; i++)
		{
			K[i][i] = k;
		}
	}
	RobinBC(const unsigned int &boundary_id, const SymmetricTensor<2, dim> &k)
		: NeumannBC<dim, Number>(boundary_id)
		, K(k) {};

	RobinBC(RobinBC &&) = delete;
	RobinBC(const RobinBC &) = delete;
	RobinBC &operator=(RobinBC &&) = delete;
	RobinBC &operator=(const RobinBC &) = delete;
	~RobinBC() = default;

	void apply(const LA::MPI::Vector &locally_relevant_u,
			   const AffineConstraints<Number> &constraints,
			   LA::MPI::SparseMatrix &system_matrix,
			   LA::MPI::Vector &system_rhs,
			   const Number &time,
			   const Number &dt) const override;

  private:
	SymmetricTensor<2, dim> K;
};

template <int dim, typename Number>
void RobinBC<dim, Number>::apply(const LA::MPI::Vector &locally_relevant_u,
								 const AffineConstraints<Number> &constraints,
								 LA::MPI::SparseMatrix &system_matrix,
								 LA::MPI::Vector &system_rhs,
								 const Number & /*time*/,
								 const Number & /*dt*/) const
{

	if (this->surface_elements.empty())
		return;

	vector<Tensor<1, dim, Number>> r_qp_vals(this->surface_elements.at(0).n_qps());
	vector<SymmetricTensor<2, dim, Number>> k_qp_vals(this->surface_elements.at(0).n_qps(), this->K);

	FullMatrix<Number> cell_matrix(this->surface_elements.at(0).n_nodes() * dim,
								   this->surface_elements.at(0).n_nodes() * dim);
	Vector<Number> cell_rhs(this->surface_elements.at(0).n_nodes() * dim);

	for (const SurfaceElement<dim, Number> &surf_el : this->surface_elements)
	{

		// LOGGER.info("Get qp u's");
		const vector<Tensor<1, dim, Number>> u_at_qps = surf_el.get_u_at_qps(locally_relevant_u);
		// LOGGER.info("Get qp r's");
		for (unsigned int qp = 0; qp < surf_el.n_qps(); qp++)
		{
			r_qp_vals.at(qp) = K * u_at_qps.at(qp);
			// cout << "K: " << K << endl;
			// cout << "r: " << r_qp_vals.at(qp)  << endl;
		}

		// LOGGER.info("Add qp r's");
		surf_el.add_to_residual_and_stiffness(
			constraints, r_qp_vals, k_qp_vals, cell_matrix, cell_rhs, system_rhs, system_matrix);
	}
}

} // namespace commet_solve

#endif // INCLUDE_BOUNDARY_CONDITIONS_ROBIN_BC_HPP_
