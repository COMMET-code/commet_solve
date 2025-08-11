#ifndef INCLUDE_PARSE_INPUT_PARSE_INPUT_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_INPUT_HPP_

#include "../time.hpp"
#include "../solver.hpp"

#include "commet_solve/parse_input/parse_utils.hpp"
#include "parse_mesh.hpp"
#include "parse_materials.hpp"
#include "parse_stage.hpp"
#include "parse_utils.hpp"
#include "parse_output_flags.hpp"
#include "parse_fields.hpp"

#include "../mesh/json_mesh.hpp"


#include <deal.II/grid/tria.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace commet_solve::parse{

using json = nlohmann::json;

void parse_input_file(const json & inp_file_contents){
    const int dim = 3;
    typedef double Number;

	Time<double> time(10, .05);
    std::map<std::string, unsigned int> b_id_map;

	// parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	//     parse_mesh<dim>(compulsory_value<json>("mesh", inp_file_contents), tri, b_id_map);

    std::vector<types::coarse_cell_id> coarse_cell_index_to_coarse_cell_id;
	parallel::fullydistributed::Triangulation<dim, dim> tria_pft(MPI_COMM_WORLD);
    {
        Triangulation<dim> tri;
        parse_mesh<dim>(compulsory_value<json>("mesh", inp_file_contents), tri, b_id_map);

        GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), tri);

        const TriangulationDescription::Description<dim, dim> description =
            TriangulationDescription::Utilities::create_description_from_triangulation(tri, MPI_COMM_WORLD); 
            coarse_cell_index_to_coarse_cell_id = description.coarse_cell_index_to_coarse_cell_id;

        tria_pft.create_triangulation(description);

    }


    const unsigned int order = value_or_default<unsigned int>("order", compulsory_value<json>("mesh", inp_file_contents), 1);
	FiniteStrainSolver<dim, double> solver(&tria_pft, time, coarse_cell_index_to_coarse_cell_id, order);

    // Parse materials
    parse_materials<dim>(compulsory_value<json>("materials", inp_file_contents), solver);

    // Parse stages
    parse_stages<dim>(compulsory_value<json>("stages", inp_file_contents),
                      solver,
                      b_id_map);

    // parse_outputs_flags<dim>(compulsory_value<json>("outputs", inp_file_contents), solver);
    parse_outputs_flags<dim>(value_or_default<json>("outputs", inp_file_contents, {}), solver);

    parse_fields<dim, Number>(value_or_default<json>("fields", inp_file_contents, {}),
                 solver);

    if(inp_file_contents.contains("miscellaneous")){
        const json & misc = inp_file_contents["miscellaneous"].get<json>();
        solver.set_output_vtus(value_or_default<bool>("output_vtus", misc, true));
        solver.set_nr_threshold(value_or_default<double>("nr_threshold", misc, solver.get_nr_threshold()));
        solver.set_max_nr_iterations(value_or_default<unsigned int>("max_nr_iterations", misc, solver.get_max_nr_iterations()));
    }


    solver.initialize();
    solver.solve();


}


}

#endif  // INCLUDE_PARSE_INPUT_PARSE_INPUT_HPP_
