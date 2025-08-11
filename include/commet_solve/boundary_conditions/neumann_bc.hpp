#ifndef INCLUDE_BOUNDARY_CONDITIONS_NEUMANN_BC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_NEUMANN_BC_HPP_

#include "../config.hpp"
#include "../fe_data/surface_element.hpp"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

namespace commet_solve
{

template <int dim, typename Number = double>
class NeumannBC
{
  public:
	NeumannBC(const unsigned int &boundary_id)
		: boundary_id(boundary_id) {};
	NeumannBC(NeumannBC &&) = delete;
	NeumannBC(const NeumannBC &) = delete;
	NeumannBC &operator=(NeumannBC &&) = delete;
	NeumannBC &operator=(const NeumannBC &) = delete;
	~NeumannBC() = default;

	void initialize(const DoFHandler<dim, dim> &dof_handler,
					const hp::MappingCollection<dim> mapping,
					const hp::FECollection<dim> fe,
					const hp::QCollection<dim> quadrature_formula);

	virtual void apply(const LA::MPI::Vector &locally_relevant_u,
					   const AffineConstraints<Number> &constraints,
					   LA::MPI::SparseMatrix &system_matrix,
					   LA::MPI::Vector &system_rhs,
					   const Number &time,
					   const Number &dt) const = 0;

  protected:
	const unsigned int boundary_id;
	vector<SurfaceElement<dim, Number>> surface_elements;
};

template <int dim, typename Number>
void NeumannBC<dim, Number>::initialize(const DoFHandler<dim, dim> &dof_handler,
										const hp::MappingCollection<dim> mapping,
										const hp::FECollection<dim> fe,
										const hp::QCollection<dim> /*quadrature_formula*/)
{

	hp::FEFaceValues<dim> hp_fe_face_values(
		mapping,
        fe,
        // quadrature_formula,
	 hp::QCollection<dim-1>(QGaussSimplex<dim-1>(fe.max_degree() + 1), QGauss<dim-1>(fe.max_degree() + 1)),
        update_values | update_gradients | update_JxW_values | update_normal_vectors);

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned() && cell->at_boundary())
		{
			for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++)
			{
				const auto &face = cell->face(i_face);

				if (face->at_boundary())
				{

					const unsigned int &face_boundary_id = face->boundary_id();


					if (face_boundary_id == boundary_id)
					{ 
						hp_fe_face_values.reinit(cell, face);
						const auto &fe_face_values = hp_fe_face_values.get_present_fe_values();
						const Quadrature<dim - 1> &face_quadrature_formula = fe_face_values.get_quadrature();
						const unsigned int n_qps = face_quadrature_formula.size();
                        const unsigned int dofs_per_face = cell->get_fe().n_dofs_per_face(i_face); 
                        const unsigned int nodes_per_face = dofs_per_face /3;

                        vector<global_dof_index> local_dof_indices(dofs_per_face);
                        vector<Tensor<1, dim, Number>> normals(n_qps);
                        vector<Number> jxw(n_qps);
                        vector<vector<Number>> N(n_qps, vector<Number>(nodes_per_face));
                        vector<vector<Tensor<1, dim, Number>>> B(n_qps, vector<Tensor<1, dim, Number>>(nodes_per_face));


						face->get_dof_indices(local_dof_indices, 
                            cell->active_fe_index());

						for (unsigned int qp = 0; qp < n_qps; qp++)
						{
							normals.at(qp) = fe_face_values.normal_vector(qp);

							for (unsigned int i_node = 0; i_node < nodes_per_face; i_node++)
							{
								B.at(qp).at(i_node) =
									fe_face_values.shape_grad(
                                        cell->get_fe().face_to_cell_index(i_node * dim, i_face), 
                                        qp
                                    );

								N.at(qp).at(i_node) =
									fe_face_values.shape_value(
                                        cell->get_fe().face_to_cell_index(i_node * dim, i_face),
                                        qp);
							}
						}

						this->surface_elements.push_back(SurfaceElement<dim, Number>(
							fe_face_values.get_JxW_values(), normals, N, B, local_dof_indices));
					}
				}
			}
		}
	}
}

} // namespace commet_solve

#endif // INCLUDE_BOUNDARY_CONDITIONS_NEUMANN_BC_HPP_
