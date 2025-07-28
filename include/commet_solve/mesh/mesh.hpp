#ifndef INCLUDE_MESH_MESH_HPP_
#define INCLUDE_MESH_MESH_HPP_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools_geometry.h>

namespace commet_solve{
using namespace dealii;



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



}

#endif  // INCLUDE_MESH_MESH_HPP_
