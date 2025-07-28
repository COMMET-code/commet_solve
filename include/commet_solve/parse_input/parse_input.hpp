#ifndef INCLUDE_PARSE_INPUT_PARSE_INPUT_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_INPUT_HPP_

#include "../time.hpp"
#include "../solver.hpp"

#include "commet_solve/parse_input/parse_utils.hpp"
#include "parse_mesh.hpp"
#include "parse_materials.hpp"
#include "parse_stage.hpp"


#include <deal.II/grid/tria.h>
#include <nlohmann/json.hpp>

namespace commet_solve::parse{

using json = nlohmann::json;

void parse_input_file(const json & inp_file_contents){
    const int dim = 3;

	Time<double> time(10, .05);

	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
    parse_mesh<dim>(compulsory_value<json>("mesh", inp_file_contents), tri);
    

    const unsigned int order = value_or_default<unsigned int>("order", compulsory_value<json>("mesh", inp_file_contents), 1);
	FiniteStrainSolver<dim, double> solver(&tri, time, order);

    // Parse materials
    parse_materials<dim>(compulsory_value<json>("materials", inp_file_contents), solver);

    // Parse stages
    parse_stages<dim>(compulsory_value<json>("stages", inp_file_contents), solver);

    solver.initialize();

    solver.solve();


}


}

#endif  // INCLUDE_PARSE_INPUT_PARSE_INPUT_HPP_
