#ifndef INCLUDE_PARSE_INPUT_PARSE_MESH_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_MESH_HPP_

#include "../mesh/json_mesh.hpp"
#include "../mesh/mesh.hpp"
#include "parse_utils.hpp"
#include <csignal>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <nlohmann/json.hpp>
#include <string>

namespace commet_solve::parse
{

using json = nlohmann::json;
using namespace std;
using namespace dealii;

enum class mesh_from
{
	builtin,
	read
};
static const map<string, mesh_from> //
	MESH_FROM({{"builtin", mesh_from::builtin}, {"read", mesh_from::read}});

enum class builtin_meshes
{
	cube,
	cylinder
};
static const map<string, builtin_meshes> //
	BUILTIN_MESHES({{"cube", builtin_meshes::cube}, {"cylinder", builtin_meshes::cylinder}});

template <int dim, typename tri_type>
void parse_builtin_mesh(const json &mesh_inp, tri_type &tri, std::map<std::string, unsigned int> &b_id_map)
{
	// parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	unsigned int refines = value_or_default("global_refines", mesh_inp, 0);
	switch (json_key_to_map_value("type", mesh_inp, BUILTIN_MESHES))
	{
	case builtin_meshes::cube: {
		GridGenerator::hyper_cube(tri);
		set_rectangular_boundary_ids<dim, tri_type>(tri);
		b_id_map["left"] = 1;
		b_id_map["bottom"] = 2;
		b_id_map["back"] = 3;
		b_id_map["right"] = 4;
		b_id_map["top"] = 5;
		b_id_map["front"] = 6;
		break;
	}
	case builtin_meshes::cylinder: {
		throw std::logic_error("Cylinder builtin not implemented yet...");
		break;
	}
	}

	if (refines > 0)
		tri.refine_global(refines);
	// return tri;
}

template <int dim, typename tri_type>
void parse_mesh(const json &mesh_inp, tri_type &tri, std::map<std::string, unsigned int> &b_id_map)
{
	// string from = mesh_inp["from"].get<string>();
	// switch (MESH_FROM.at(from))
	switch (json_key_to_map_value("from", mesh_inp, MESH_FROM))
	{
	case mesh_from::builtin: {
		parse_builtin_mesh<dim, tri_type>(mesh_inp, tri, b_id_map);
		return;
	}
	case mesh_from::read: {
		// Triangulation<dim, dim> triangulation;
		unsigned int refines = value_or_default("global_refines", mesh_inp, 0);
		triangulation_from_json(tri, compulsory_value<std::string>("path", mesh_inp), b_id_map);
		// tri.copy_triangulation(triangulation);
		if (refines > 0)
			tri.refine_global(refines);
		break;
	}
	}
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_MESH_HPP_
