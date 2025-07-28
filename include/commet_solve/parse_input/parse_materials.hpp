#ifndef INCLUDE_PARSE_INPUT_PARSE_MATERIALS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_MATERIALS_HPP_

#include "../mesh/mesh.hpp"
#include "commet_solve/material_domain/isotropic_hyperelastic_domain.hpp"
#include "commet_solve/material_domain/material_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/batch_vectorized_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/globally_vectorized_domain.hpp"
#include "commet_solve/solver.hpp"
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

enum class materials
{
	globally_vectorized_ncm,
	batch_vectorized_ncm,
	neohookean
};

static const map<string, materials> //
	MATERIALS({{"globally_vectorized_ncm", materials::globally_vectorized_ncm},
			   {"batch_vectorized_ncm", materials::batch_vectorized_ncm},
			   {"neohookean", materials::neohookean}});

static const map<string, NCMEvaluationMethod> //
	NCM_EVALUATION({{"optimized", NCMEvaluationMethod::OPT_F},
					{"using_F", NCMEvaluationMethod::USING_F},
					{"using_C", NCMEvaluationMethod::USING_C}});

template <int dim, typename Number>
void parse_material(const json &material_spec, FiniteStrainSolver<dim, Number>& solver)
{

	switch (json_key_to_map_value("type", material_spec, MATERIALS))
	{
	case materials::batch_vectorized_ncm: {
		const unsigned int batch_size = compulsory_value<unsigned int>("batch_size", material_spec);
		auto domain = std::make_unique<commet_solve::BatchVectorizedDomain<dim, Number>>(batch_size);
		domain->set_evaluation_method(json_key_to_map_value("evaluation_method", material_spec, NCM_EVALUATION));

		domain->load_model(compulsory_value<string>("path_to_torchscript", material_spec));
		solver.add_material_domain(compulsory_value<unsigned int>("id", material_spec), std::move(domain));

		return;
	}
	case materials::globally_vectorized_ncm: {

		auto domain = std::make_unique<GloballyVectorizedDomain<dim, Number>>();
		domain->set_evaluation_method(json_key_to_map_value("evaluation_method", material_spec, NCM_EVALUATION));

		domain->load_model(compulsory_value<string>("path_to_torchscript", material_spec));
		solver.add_material_domain(compulsory_value<unsigned int>("id", material_spec), std::move(domain));

		return;
	}
	case materials::neohookean: {

		const Number mu = compulsory_value<Number>("mu", material_spec);
		const Number lambda = compulsory_value<Number>("lambda", material_spec);

		auto domain = std::make_unique<NeoHookeanDomain<dim, Number>>(lambda, mu);
		solver.add_material_domain(compulsory_value<unsigned int>("id", material_spec), std::move(domain));

		return;
	}
	}
}

template <int dim, typename Number>
void parse_materials(const json &material_inp, FiniteStrainSolver<dim, Number>& solver)
{
	for (const auto &material_spec : material_inp)
		parse_material(material_spec, solver);
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_MATERIALS_HPP_
