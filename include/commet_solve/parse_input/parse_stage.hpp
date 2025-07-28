#ifndef INCLUDE_PARSE_INPUT_PARSE_STAGE_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_STAGE_HPP_

#include "commet_solve/stage.hpp"
#include "parse_utils.hpp"

#include "commet_solve/boundary_conditions/fully_defined_dbc.hpp"
#include "commet_solve/boundary_conditions/twist_pull.hpp"
#include <memory>

namespace commet_solve::parse
{

using json = nlohmann::json;
using namespace std;
using namespace dealii;

enum class dbcs
{
	standard,
	pull_twist
};
static const map<string, dbcs> //
	DBCS({{"standard", dbcs::standard}, {"pull_twist", dbcs::pull_twist}});

template <int dim, typename Number>
void parse_dbc(const json &dbc_spec, Stage<dim, Number> *stage)
{
	const unsigned int b_id = compulsory_value<unsigned int>("boundary_id", dbc_spec);

	switch (json_key_to_map_value("type", dbc_spec, DBCS))
	{
	case dbcs::standard: {
		stage->add_dbc(make_unique<FullyDefinedDBC<dim, Number>>(
			b_id,
			compulsory_value<std::vector<unsigned int>>("components", dbc_spec),
			compulsory_value<std::vector<Number>>("values", dbc_spec),
			stage->end_time));
		return;
	}
	case dbcs::pull_twist: {
		stage->add_dbc(make_unique<RotateBoundaryCondition<dim, Number>>(
			b_id,
			compulsory_vector<dim, Number, Point<dim, Number>>("centre", dbc_spec),
			compulsory_vector<dim, Number, Tensor<1, dim, Number>>("axis", dbc_spec),
			compulsory_value<Number>("theta", dbc_spec),
			compulsory_value<Number>("u", dbc_spec),
			stage->end_time));
		return;
	}
	}
}

// template <int dim, typename Number>
// void parse_dbcs(const json & stage_dbc){}

template <int dim, typename Number>
void parse_stage(const json &stage_spec, FiniteStrainSolver<dim, Number> &solver)
{
	const Number end_time = value_or_default<Number>("end_time", stage_spec, 1);
	const Number dt = value_or_default<Number>("time_increment_size", stage_spec, 1);
	auto stage = make_unique<Stage<dim, Number>>(end_time, dt);

	for (json &dbc_spec : compulsory_value<json>("dirichlet_boundary_conditions", stage_spec))
		parse_dbc(dbc_spec, stage.get());

	solver.add_stage(move(stage));
}

template <int dim, typename Number>
void parse_stages(const json &stage_inp, FiniteStrainSolver<dim, Number> &solver)
{

	for (const auto &stage_spec : stage_inp)
		parse_stage(stage_spec, solver);
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_STAGE_HPP_
