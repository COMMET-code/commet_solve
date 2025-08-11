#ifndef INCLUDE_PARSE_INPUT_PARSE_FIELDS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_FIELDS_HPP_

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

enum class field_inp_types
{
	element_wise,
	analytical
};

static const map<string, field_inp_types> FIELD_INP_TYPES({
    {"element_wise", field_inp_types::element_wise},
    {"analytical", field_inp_types::analytical}
});


template <int dim, typename Number>
void parse_analytical_vector_field(const json &v_field_spec,
                                   FiniteStrainSolver<dim, Number> &solver)
{
	const string name = compulsory_value<string>("name", v_field_spec);
	const string x_expression = compulsory_value<string>("x_func", v_field_spec);
	const string y_expression = compulsory_value<string>("y_func", v_field_spec);
	const string z_expression = compulsory_value<string>("z_func", v_field_spec);


	solver.add_analytical_vector_field(name, x_expression, y_expression, z_expression);
}

template <int dim, typename Number>
void parse_elementwise_vector_field(const json &v_field_spec, FiniteStrainSolver<dim, Number> &solver)
{

	const string name = compulsory_value<string>("name", v_field_spec);
	const string path = compulsory_value<string>("path", v_field_spec);

	vector<Tensor<1, dim>> data;
	{
		json j_data;
		ifstream(path.c_str()) >> j_data;
		data.resize(j_data.size());
		unsigned int count = 0;
		for (vector<double> vals : j_data.get<vector<vector<double>>>())
		{
			data.at(count) = Tensor<1, dim>({vals[0], vals[1], vals[2]});
			count++;
		}
	}

	solver.add_vector_field(name, data);
}

template <int dim, typename Number>
void parse_vector_field(const json &v_field_spec, FiniteStrainSolver<dim, Number> &solver)
{
	switch (json_key_to_map_value("type", v_field_spec, FIELD_INP_TYPES))
	{

	case field_inp_types::element_wise: {
		parse_elementwise_vector_field<dim, Number>(v_field_spec, solver);
		return;
	}
	case field_inp_types::analytical: {
		parse_analytical_vector_field<dim, Number>(v_field_spec, solver);
		return;
        }
	}
}

template <int dim, typename Number>
void parse_fields(const json &field_inp, FiniteStrainSolver<dim, Number> &solver)
{
	for (const auto &vec_spec : value_or_default<json>("vector", field_inp, {}))
		parse_vector_field<dim, Number>(vec_spec, solver);
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_FIELDS_HPP_
