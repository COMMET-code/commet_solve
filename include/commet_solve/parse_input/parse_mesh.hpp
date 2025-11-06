#ifndef INCLUDE_PARSE_INPUT_PARSE_MESH_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_MESH_HPP_

#include "../mesh/json_mesh.hpp"
#include "../mesh/mesh.hpp"
#include "parse_utils.hpp"
#include <csignal>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace commet_solve::parse {

using json = nlohmann::json;
using namespace std;
using namespace dealii;

enum class mesh_from { builtin, read };

static const map<string, mesh_from> MESH_FROM({{"built-in", mesh_from::builtin},
                                               {"read", mesh_from::read}});

enum class builtin_meshes {
  cube,
  cylinder,
  plate_with_hole,
  quarter_plate_with_hole,
};

static const map<string, builtin_meshes> BUILTIN_MESHES(
    {{"cube", builtin_meshes::cube},
     {"plate_with_hole", builtin_meshes::plate_with_hole},
     {"quarter_plate_with_hole", builtin_meshes::quarter_plate_with_hole},
     {"cylinder", builtin_meshes::cylinder}});

template <int dim, typename tri_type>
void parse_builtin_mesh(const json &mesh_inp, tri_type &tri,
                        std::map<std::string, unsigned int> &b_id_map) {
  // parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
  unsigned int refines = value_or_default("global_refines", mesh_inp, 0);
  switch (json_key_to_map_value("type", mesh_inp, BUILTIN_MESHES)) {
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
  case builtin_meshes::plate_with_hole: {

    labeled_hyper_rectangle_with_hole<dim>(
        tri, value_or_default<double>("length", mesh_inp, 1),
        value_or_default<double>("radius", mesh_inp, 0.25),
        value_or_default<double>("thickness", mesh_inp, 1),
        value_or_default<unsigned int>("planar_refinements", mesh_inp, 0),
        value_or_default<unsigned int>("global_refinements", mesh_inp, 0));

    set_rectangular_boundary_ids<dim, tri_type>(tri);
    break;
  }
  case builtin_meshes::quarter_plate_with_hole: {

    labeled_quarter_hyper_rectangle_with_hole<dim>(
        tri, value_or_default<double>("length", mesh_inp, 1),
        value_or_default<double>("radius", mesh_inp, 0.25),
        value_or_default<double>("thickness", mesh_inp, 1),
        value_or_default<unsigned int>("planar_refinements", mesh_inp, 0),
        value_or_default<unsigned int>("global_refinements", mesh_inp, 0));

    set_rectangular_boundary_ids<dim, tri_type>(tri);
    break;
  }
  case builtin_meshes::cylinder: {
    throw std::logic_error("Cylinder builtin not implemented yet...");
    break;
  }
  }

  if (refines > 0)
    tri.refine_global(refines);
}

template <int dim, typename tri_type>
void parse_mesh(const json &mesh_inp, tri_type &tri,
                std::map<std::string, unsigned int> &b_id_map) {
  switch (json_key_to_map_value("from", mesh_inp, MESH_FROM)) {
  case mesh_from::builtin: {
    parse_builtin_mesh<dim, tri_type>(mesh_inp, tri, b_id_map);
    return;
  }
  case mesh_from::read: {
    commet_solve::LOGGER.info("Reading mesh");

    unsigned int refines = value_or_default("global_refines", mesh_inp, 0);
    string pth_to_mesh = compulsory_value<std::string>("path", mesh_inp);
    stringstream sspth_to_mesh(pth_to_mesh);
    string file_name;
    string extension;
    commet_solve::LOGGER.info("starting get line...: " + pth_to_mesh);
    while (std::getline(sspth_to_mesh, file_name, '/'))
      ;
    stringstream ss_file_name(file_name);
    commet_solve::LOGGER.info("file_name: " + file_name);
    while (std::getline(ss_file_name, extension, '.'))
      ;
    commet_solve::LOGGER.info("extension: " + extension);

    if (extension == "msh") {
      GridIn<dim, dim> grid_in(tri);
      // std::ifstream in_stream(pth_to_mesh);
      grid_in.read_msh(pth_to_mesh);

    } else if (extension.substr(0, 4) == "json")
      triangulation_from_json(tri, pth_to_mesh, b_id_map);
    else
      throw std::runtime_error(
          "Could not determine type of mesh to be read for file extension: " +
          extension);

    if (refines > 0)
      tri.refine_global(refines);
    break;
  }
  }
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_MESH_HPP_
