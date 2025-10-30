#ifndef INCLUDE_PARSE_INPUT_PARSE_STAGE_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_STAGE_HPP_

#include "commet_solve/stage.hpp"
#include "parse_utils.hpp"

#include "commet_solve/boundary_conditions/fully_defined_dbc.hpp"
#include "commet_solve/boundary_conditions/twist_pull.hpp"

#include "commet_solve/boundary_conditions/constant_nbc.hpp"
#include "commet_solve/boundary_conditions/pressure_nbc.hpp"
#include "commet_solve/boundary_conditions/robin_bc.hpp"
#include "commet_solve/boundary_conditions/normal_robin_bc.hpp"
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

unsigned int 
get_bid(const json &spec, 
        const std::map<std::string, unsigned int> &b_id_map ){
    const string key = "boundary_id";

	if (spec.contains(key)){
        if(spec[key].is_string()) 
            return b_id_map.at(spec[key].get<string>());
        else if(spec[key].is_number_integer()) 
            return spec[key].get<unsigned int>();
        else
		throw std::logic_error("Value for boundary_id in the following file section is invalid: "+to_string(spec));
    }
	else
		throw std::logic_error("Required key '" + key + "' is missing from file section: '" + to_string(spec) + "'.");
}

template <int dim, typename Number>
void parse_dbc(const json &dbc_spec,
               Stage<dim, Number> *stage,
               std::map<std::string, unsigned int> &b_id_map)
{
	// const unsigned int b_id = compulsory_value<unsigned int>("boundary_id", dbc_spec);

	// const unsigned int b_id = b_id_map.at(compulsory_value<std::string>("boundary_id",
	//                                                                      dbc_spec)); 
    const unsigned int b_id = get_bid(dbc_spec, b_id_map);

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

enum class nbcs
{
	constant,
	pressure,
	robin,
	normal_robin
};
static const map<string, nbcs> //
	NBCS({{"constant", nbcs::constant},//
    {"pressure", nbcs::pressure}, //
    {"robin", nbcs::robin},//
    {"normal_robin", nbcs::normal_robin}
});
template <int dim, typename Number>
void parse_nbc(const json &nbc_spec, Stage<dim, Number> *stage, std::map<std::string, unsigned int> &b_id_map)
{

	// const unsigned int b_id = compulsory_value<unsigned int>("boundary_id", nbc_spec);
    const unsigned int b_id = get_bid(nbc_spec, b_id_map);
	// const unsigned int b_id = b_id_map.at(compulsory_value<std::string>("boundary_id", nbc_spec));

	switch (json_key_to_map_value("type", nbc_spec, NBCS))
	{
	case nbcs::constant: {
		stage->add_nbc(make_unique<ConstNeumannBC<dim, Number>>(
			b_id, compulsory_vector<dim, Number, Tensor<1, dim, Number>>("traction", nbc_spec), stage->end_time));
		return;
	}
	case nbcs::pressure: {
		stage->add_nbc(make_unique<PressureNeumannBC<dim, Number>>(
			b_id, compulsory_value<Number>("pressure", nbc_spec), stage->end_time));
		return;
	}
	case nbcs::robin: {
		stage->add_nbc(make_unique<RobinBC<dim, Number>>(b_id, compulsory_value<Number>("stiffness", nbc_spec)));
		return;
	}
	case nbcs::normal_robin: {
		stage->add_nbc(make_unique<NormalRobinBC<dim, Number>>(b_id, compulsory_value<Number>("stiffness", nbc_spec)));
		return;
	}
	}
}

// template <int dim, typename Number>
// void parse_dbcs(const json & stage_dbc){}

template <int dim, typename Number>
void parse_stage(const json &stage_spec,
				 FiniteStrainSolver<dim, Number> &solver,
				 std::map<std::string, unsigned int> &b_id_map)
{
	const Number end_time = value_or_default<Number>("end_time", stage_spec, 1);
	const Number dt = value_or_default<Number>("time_increment_size", stage_spec, 1);
	auto stage = make_unique<Stage<dim, Number>>(end_time, dt);

	for (json &dbc_spec : compulsory_value<json>("dirichlet_boundary_conditions", stage_spec))
		parse_dbc(dbc_spec, stage.get(), b_id_map);

	for (json &nbc_spec : compulsory_value<json>("neumann_boundary_conditions", stage_spec))
		parse_nbc(nbc_spec, stage.get(), b_id_map);

	solver.add_stage(move(stage));
}

template <int dim, typename Number>
void parse_stages(const json &stage_inp,
				  FiniteStrainSolver<dim, Number> &solver,
				  std::map<std::string, unsigned int> &b_id_map)
{

	for (const auto &stage_spec : stage_inp)
		parse_stage(stage_spec, solver, b_id_map);
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_STAGE_HPP_
