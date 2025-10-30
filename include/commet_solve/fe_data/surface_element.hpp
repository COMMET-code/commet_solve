#ifndef INCLUDE_FE_DATA_SURFACE_ELEMENT_HPP_
#define INCLUDE_FE_DATA_SURFACE_ELEMENT_HPP_

#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <vector>

namespace commet_solve
{
using namespace std;
using namespace dealii;
using namespace dealii::types;

template <int dim, typename Number = double>
class SurfaceElement
{
  public:
	SurfaceElement(const vector<Number> jwx,
				   const vector<Tensor<1, dim, Number>> normals,
				   const vector<vector<Number>> N,
				   const vector<vector<Tensor<1, dim, Number>>> B,
				   const vector<global_dof_index> dofs)
		: jwx(jwx)
		, normals(normals)
		, N(N)
		, B(B)
		, dofs(dofs) {
			// cout << "creating surface el..." << endl;

		};

	SurfaceElement(SurfaceElement &&other)
		: jwx(other.jwx)
		, normals(other.normals)
		, N(other.N)
		, B(other.B)
		, dofs(other.dofs) {};
	SurfaceElement(const SurfaceElement &other)
		: jwx(other.jwx)
		, normals(other.normals)
		, N(other.N)
		, B(other.B)
		, dofs(other.dofs) {};
	~SurfaceElement() = default;

	void add_to_residual(const vector<Tensor<1, dim, Number>> &qp_vals,
						 const AffineConstraints<Number> &constraints,
						 LA::MPI::Vector &system_rhs,
						 Vector<double> &cell_rhs) const
	{

		// cout << "-------------------------------" << endl;
		// cout << "jxw: " << jwx << endl;
		// cout << "dofs: " << dofs << endl;
		Tensor<1, dim, Number> r_qp;
		cell_rhs = 0.;
		for (unsigned int qp = 0; qp < this->n_qps(); qp++)
		{

			// cout << "qp =" << qp << endl;
			r_qp = this->jwx.at(qp) * qp_vals.at(qp);

			for (unsigned int i_node = 0; i_node < this->n_nodes(); i_node++)
			{
				const Number &N_qp_i_node = N.at(qp).at(i_node);
				// cout << "i_node =" << i_node << endl;

				for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
					// r[this->dofs.at(i_comp + i_node * dim)] += r_qp[i_comp] * N_qp_i_node;
					cell_rhs(i_comp + i_node * dim) += r_qp[i_comp] * N_qp_i_node;
			}
		}

		constraints.distribute_local_to_global(cell_rhs, this->dofs, system_rhs);
	};
	void add_to_residual_and_stiffness(const AffineConstraints<Number> &constraints,
									   const vector<Tensor<1, dim, Number>> &r_qp_vals,
									   const vector<SymmetricTensor<2, dim, Number>> &k_qp_vals,
									   FullMatrix<double> &cell_matrix,
									   Vector<double> &cell_rhs,
									   LA::MPI::Vector &system_rhs,
									   LA::MPI::SparseMatrix &system_matrix) const
	{

		// FullMatrix<double> cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
		// Vector<double> cell_rhs(n_dofs_per_cell);

		// cout << "-------------------------------" << endl;
		// cout << "jxw: " << jwx << endl;
		// cout << "dofs: " << dofs << endl;
		// Tensor<1, dim, Number> r_qp;
		// Tensor<1, dim, Number> r_qp;

		// constraints.distribute_local_to_global(cell_matrix,
		// 									   cell_rhs,
		// 									   cell.get_dofs(),
		// 									   system_matrix,
		// 									   system_rhs,
		// 									   /*use_inhomogeneities_for_rhs*/ false);
		//

		cell_matrix = 0.;
		cell_rhs = 0.;
		for (unsigned int qp = 0; qp < this->n_qps(); qp++)
		{

			// cout << "qp =" << qp << endl;
			const Tensor<1, dim, Number> r_qp = this->jwx.at(qp) * r_qp_vals.at(qp);
			const SymmetricTensor<2, dim, Number> k_qp = this->jwx.at(qp) * k_qp_vals.at(qp);

			for (unsigned int i_node = 0; i_node < this->n_nodes(); i_node++)
			{
				const Number &N_qp_i_node = N.at(qp).at(i_node);
				for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
				{
					// r[this->dofs.at(i_comp + i_node * dim)] += r_qp[i_comp] * N_qp_i_node;
					// cell_rhs[i_comp + i_node * dim] += r_qp[i_comp] * N_qp_i_node;
					// cell_rhs(i_comp + i_node * dim) += r_qp[i_comp] * N_qp_i_node;
					cell_rhs(i_comp + i_node * dim) += -r_qp[i_comp] * N_qp_i_node;
				}

				for (unsigned int j_node = 0; j_node < this->n_nodes(); j_node++)
				{

					const Number &N_qp_j_node = N.at(qp).at(j_node);
					const SymmetricTensor<2, dim, Number> K_ij = k_qp * N_qp_j_node * N_qp_i_node;

					for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
					{
						for (unsigned int j_comp = 0; j_comp < dim; j_comp++)
						{
							// system_matrix.add(this->dofs.at(i_comp + i_node * dim),
							// 				  this->dofs.at(j_comp + j_node * dim),
							// 				  K_ij[i_comp][j_comp]);
							// cell_matrix(i_comp + i_node * dim, j_comp + j_node * dim) = K_ij[i_comp][j_comp];
							cell_matrix(i_comp + i_node * dim, j_comp + j_node * dim) += K_ij[i_comp][j_comp];
						}
					}
				}
			}
		}

		constraints.distribute_local_to_global(cell_matrix,
											   cell_rhs,
											   this->dofs,
											   system_matrix,
											   system_rhs,
											   /*use_inhomogeneities_for_rhs*/ false);
	};
	void add_to_residual(Tensor<1, dim, Number> qp_vals,
						 const AffineConstraints<Number> &constraints,
						 LA::MPI::Vector &r,
						 Vector<double> &cell_rhs) const
	{
		// this->add_to_residual(vector<Tensor<1, dim, Number>>(this->n_qps(), qp_vals), r);
		this->add_to_residual(vector<Tensor<1, dim, Number>>(this->n_qps(), qp_vals), constraints, r, cell_rhs);
	};

	inline unsigned int n_qps() const
	{
		return this->jwx.size();
	};
	inline unsigned int n_nodes() const
	{
		return this->dofs.size() / dim;
	};
	const vector<Tensor<1, dim, Number>> &get_normals() const
	{
		return normals;
	};
	// vector<Tensor<1, dim, Number>> get_u(const LA::MPI::Vector &locally_relevant_u) const
	vector<Tensor<1, dim, Number>> get_u_at_qps(const LA::MPI::Vector &locally_relevant_u) const
	{
		vector<Tensor<1, dim, Number>> u_nodes(this->n_nodes());
		for (unsigned int i_node = 0; i_node < this->n_nodes(); i_node++)
			for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
				u_nodes.at(i_node)[i_comp] = locally_relevant_u[this->dofs.at(i_comp + i_node * dim)];

		vector<Tensor<1, dim, Number>> u_at_qps(this->n_qps());
		for (unsigned int qp = 0; qp < this->n_qps(); qp++)
			for (unsigned int i_node = 0; i_node < this->n_nodes(); i_node++)
				u_at_qps.at(qp) += this->N.at(qp).at(i_node) * u_nodes.at(i_node);

		return u_at_qps;
	};

  private:
	const vector<Number> jwx;
	const vector<Tensor<1, dim, Number>> normals;
	const vector<vector<Number>> N;
	const vector<vector<Tensor<1, dim, Number>>> B;
	const vector<global_dof_index> dofs;
};

} // namespace commet_solve

#endif // INCLUDE_FE_DATA_SURFACE_ELEMENT_HPP_
