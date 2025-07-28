#include <commet_solve/material_domain/isotropic_hyperelastic_domain.hpp>
#include <commet_solve/solver.hpp>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <fstream>
#include <iostream>
// #include <deal.II/base/tensor.h>
//
#include <deal.II/grid/grid_tools.h>
//

#include <deal.II/grid/grid_generator.h>
#include <memory>
#include <string>
#include <vector>

#include "commet_solve/boundary_conditions/fully_defined_dbc.hpp"
#include "commet_solve/boundary_conditions/twist_pull.hpp"
#include "commet_solve/material_domain/ncm_domain/batch_vectorized_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/globally_vectorized_domain.hpp"
#include "commet_solve/mesh/mesh.hpp"
#include "commet_solve/time.hpp"

#include "commet_solve/parse_input/parse_input.hpp"

using namespace dealii;
void tenth_test(const std::string &input_path)
{


    nlohmann::json inp = nlohmann::json::parse(std::ifstream(input_path.c_str()),
                         /* callback */ nullptr,
                         /* allow exceptions */ true,
                         /* ignore_comments */ true);

	// nlohmann::json inp = nlohmann::json::parse(std::ifstream(input_path.c_str()),
	//                                             nullptr,
                                            // );
	// std::ifstream(input_path.c_str()) >> inp;
	//     inp.
	// std::ifstream(input_path.c_str()) >> inp;
	// std::cout << inp << std::endl;
	commet_solve::parse::parse_input_file(inp);
}

int main(int argc, char *argv[])
{
	// dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	// torch::set_num_threads(1);

	// std::string inp_path(argv[1]);
	// tenth_test(inp_path);



  try
    {

	dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	torch::set_num_threads(1);

	std::string inp_path(argv[1]);
	tenth_test(inp_path);

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

	return 0;
}
