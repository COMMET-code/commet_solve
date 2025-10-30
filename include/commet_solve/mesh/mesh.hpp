#ifndef INCLUDE_MESH_MESH_HPP_
#define INCLUDE_MESH_MESH_HPP_

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

namespace commet_solve
{
using namespace dealii;
using namespace GridGenerator;
using namespace GridTools;

template <unsigned int dim>
void output_triangulation_vtk(const string &name, const Triangulation<dim> &triangulation)
{
	std::ofstream out(name + ".vtk");
	GridOut grid_out;
	grid_out.write_vtk<dim>(triangulation, out);
}

template <unsigned int dim, typename tri_type>
void set_rectangular_boundary_ids(tri_type &triangulation)
{
	BoundingBox<dim> bounding_box = GridTools::compute_bounding_box(triangulation);

	for (const auto &cell : triangulation.active_cell_iterators())
	{
		if (cell->at_boundary())
			for (const auto &face : cell->face_iterators())
			{
				if (face->at_boundary())
					for (unsigned int i = 0; i < dim; i++)
					{
						if (almost_equals(face->center()[i], bounding_box.lower_bound(i)))
							face->set_boundary_id(i + 1);
						else if (almost_equals(face->center()[i], bounding_box.upper_bound(i)))
							face->set_boundary_id(i + 1 + dim);
					}
			}
	}
}


template <unsigned int dim>
void labeled_hyper_rectangle_with_hole(Triangulation<dim>&  final_triangulation, 
                                       const double &length = 1, 
                                       const double &radius = 1, 
                                       const double &thickness = 1, 
                                       const unsigned int &n_refinements_pre_extrude = 0, 
                                       const unsigned int &n_refinements_post_extrude = 0)
{
	assert(length > 0);
	assert(radius < length);
	Triangulation<2> triangulation;
	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, radius, length, length);
	for (const auto &cell : triangulation.active_cell_iterators())
	{
		if (cell->at_boundary())
			for (const auto &face : cell->face_iterators())
			{
				if (face->at_boundary())
					face->set_boundary_id(0);
			}
	}

	if (n_refinements_pre_extrude > 0)
		triangulation.refine_global(n_refinements_pre_extrude);

	GridGenerator::extrude_triangulation(triangulation,
										 {0, thickness},
										 final_triangulation,
										 /*copy_manifold_ids*/ true);

	final_triangulation.reset_manifold(0);
	final_triangulation.set_manifold(0, CylindricalManifold<dim>(2));

	set_rectangular_boundary_ids<dim>(final_triangulation);

	if (n_refinements_post_extrude > 0)
		final_triangulation.refine_global(n_refinements_post_extrude);
}


template <unsigned int dim>
void labeled_quarter_hyper_rectangle_with_hole(Triangulation<dim>& final_final_triangulation, 
                                                             const double &length = 1,
															 const double &radius = 1,
															 const double &thickness = 1,
															 const unsigned int &n_refinements_pre_extrude = 0,
															 const unsigned int &n_refinements_post_extrude = 0)
{
	assert(length > 0);
	assert(radius < length);
	Triangulation<2> triangulation;
	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, radius, length, length);
	for (const auto &cell : triangulation.active_cell_iterators())
	{
		if (cell->at_boundary())
			for (const auto &face : cell->face_iterators())
			{
				if (face->at_boundary())
					face->set_boundary_id(0);
			}
	}

	if (n_refinements_pre_extrude > 0)
		triangulation.refine_global(n_refinements_pre_extrude);

	Triangulation<dim> final_triangulation;
	GridGenerator::extrude_triangulation(triangulation,
										 {0, thickness},
										 final_triangulation,
										 /*copy_manifold_ids*/ true);

	final_triangulation.reset_manifold(0);
	final_triangulation.set_manifold(0, CylindricalManifold<dim>(2));

	set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;

	for (const auto &cell : final_triangulation.active_cell_iterators())
		if (cell->center()[0] < 0 || cell->center()[1] < 0)
			cells_to_remove.insert(cell);

	GridGenerator::create_triangulation_with_removed_cells(
		final_triangulation, cells_to_remove, final_final_triangulation);

	set_rectangular_boundary_ids<dim>(final_final_triangulation);

	if (n_refinements_post_extrude > 0)
		final_final_triangulation.refine_global(n_refinements_post_extrude);
}

} // namespace commet_solve

#endif // INCLUDE_MESH_MESH_HPP_
