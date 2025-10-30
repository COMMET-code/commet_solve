#ifndef INCLUDE_PARSE_INPUT_PARSE_MATERIALS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_MATERIALS_HPP_

#include "../mesh/mesh.hpp"
#include "commet_solve/material_domain/isotropic_hyperelastic_domain.hpp"
#include "commet_solve/material_domain/material_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/batch_vectorized_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/globally_vectorized_domain.hpp"
#include "commet_solve/solver.hpp"
#include "parse_ncm_materials.hpp"
#include "parse_traditional_materials.hpp"
#include "parse_utils.hpp"
#include <csignal>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace commet_solve::parse
{

using json = nlohmann::json;
using namespace std;
using namespace dealii;

enum class material_types
{
	ncm,
	traditional,
};


static const map<string, material_types> //
	MATERIAL_TYPES({
    {"ncm", material_types::ncm},
    {"traditional", material_types::traditional}
	});


template <int dim, typename Number>
void parse_material(const json &material_spec,
                    FiniteStrainSolver<dim, Number> &solver){


	switch (json_key_to_map_value("type", material_spec, MATERIAL_TYPES)){
        case material_types::ncm:{
            parse_ncm_material<dim, Number>(material_spec, solver);
            return;
        }
        case material_types::traditional:{
            parse_traditional_material<dim, Number>(material_spec, solver);
            return;
        }

    }

}

template <int dim, typename Number>
void parse_materials(const json &material_inp, FiniteStrainSolver<dim, Number> &solver)
{
	for (const auto &material_spec : material_inp)
		parse_material(material_spec, solver);
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_MATERIALS_HPP_
