#ifndef INCLUDE_PARSE_INPUT_PARSE_TRADITIONAL_MATERIALS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_TRADITIONAL_MATERIALS_HPP_

#include "../material_domain/materials/hyperelastic_material.hpp"
#include "../material_domain/materials/iso_vol_hyperelastic_material.hpp"
#include "../material_domain/materials/iso_materials.hpp"
#include "../material_domain/materials/vol_materials.hpp"
#include "commet_solve/material_domain/isotropic_hyperelastic_domain.hpp"
#include "commet_solve/material_domain/materials/neohookean.hpp"
#include "commet_solve/parse_input/parse_utils.hpp"
#include <memory>

namespace commet_solve::parse
{

using json = nlohmann::json;
using namespace std;
using namespace dealii;

enum class standard_hyperelastic_materials
{
	neohookean
};


static const map<string, standard_hyperelastic_materials> //
	STANDARD_HYPERELASTIC_MATERIALS({{"neohookean", standard_hyperelastic_materials::neohookean}});



enum class elastic_material_types
{
	iso_vol,
	neohookean
};


static const map<string, elastic_material_types> //
	ELASTIC_MATERIAL_TYPES({
    {"iso_vol", elastic_material_types::iso_vol},
    {"neohookean", elastic_material_types::neohookean}
});


enum class iso_materials
{
	isihara,
	hgo,
};

static const map<string, iso_materials> //
	ISO_MATERIALS({
		{"isihara", iso_materials::isihara},
		{"hgo", iso_materials::hgo},
	});

enum class vol_materials
{
	square,
	square_and_log,
};

static const map<string, vol_materials> //
	VOL_MATERIALS({
		{"square", vol_materials::square},
		{"square_and_log", vol_materials::square_and_log},
	});

template <int dim, typename Number>
std::unique_ptr<HyperelasticMaterial<dim, Number>> parse_standard_material(const json &material_spec)
{

	switch (json_key_to_map_value("type", material_spec, ELASTIC_MATERIAL_TYPES))
	{

	case elastic_material_types::iso_vol: {
            throw std::logic_error("How did you get to this point in the control flow?...");
        }
	case elastic_material_types::neohookean: {
		return make_unique<Neohookean<dim, Number>>(compulsory_value<double>("lambda", material_spec),
													compulsory_value<double>("mu", material_spec));
	}
	}
    throw std::logic_error("Could not select valid hyperelastic material model...");
}

template <int dim, typename Number>
std::unique_ptr<IsoHyperelasticMaterial<dim, Number>> parse_iso_material(const json &material_spec)
{

	switch (json_key_to_map_value("type", material_spec, ISO_MATERIALS))
	{

	case iso_materials::isihara: {
		return make_unique<Isihara<dim, Number>>(compulsory_value<double>("c1", material_spec),
												 compulsory_value<double>("c2", material_spec),
												 compulsory_value<double>("c3", material_spec));
	}
	case iso_materials::hgo: {
		return make_unique<HGO<dim, Number>>(compulsory_value<double>("a", material_spec), 
                                       compulsory_value<double>("b", material_spec), 
                                       compulsory_array_zero_padded<N_ORIENTATION_VECS, double>("as", material_spec),
                                       compulsory_array_zero_padded<N_ORIENTATION_VECS, double>("bs", material_spec)
                                       );
	}
	}
    throw std::logic_error("Could not select valid `iso` component of material behaviour...");
}

template <int dim, typename Number>
std::unique_ptr<VolHyperelasticMaterial<Number>> parse_vol_material(const json &material_spec)
{

	switch (json_key_to_map_value("type", material_spec, VOL_MATERIALS))
	{
	case vol_materials::square: {
		return make_unique<Square<Number>>(compulsory_value<double>("k", material_spec));
	}
	case vol_materials::square_and_log: {
		return make_unique<SquareAndLog<Number>>(compulsory_value<double>("k", material_spec));
	}
	}

    throw std::logic_error("Could not select valid `vol` component of material behaviour...");
}

template <int dim, typename Number>
void parse_traditional_material(const json &material_spec, FiniteStrainSolver<dim, Number> &solver)
{
	vector<string> orientation_vector_names = value_or_default<vector<string>>("orientation_vectors", material_spec, {});
    auto domain = std::make_unique<HyperelasticDomain<dim, Number>>();
    domain->orientation_field_names = orientation_vector_names; 
    auto elasticity_spec = compulsory_value<json>("elasticity", material_spec);

	// switch (json_key_to_map_value("type", elasticity_spec, ELASTICITY_TYPES))
	switch (json_key_to_map_value("type", elasticity_spec, ELASTIC_MATERIAL_TYPES))
	{
	case elastic_material_types::iso_vol: {

        unique_ptr<IsoHyperelasticMaterial<dim, Number>> 
            iso_mat = parse_iso_material<dim, Number>(compulsory_value<json>("iso", elasticity_spec));
        unique_ptr<VolHyperelasticMaterial<Number>> 
            vol_mat = parse_vol_material<dim, Number>(compulsory_value<json>("vol", elasticity_spec));

        auto material = make_unique<IsoVolHyperelasticMaterial<dim, Number>>(move(iso_mat), move(vol_mat));
        domain->set_material_model(move(material));
		solver.add_material_domain(compulsory_value<unsigned int>("id", material_spec), std::move(domain));
		return ;
        }
        default:{
        domain->set_material_model(parse_standard_material<dim, Number>(elasticity_spec));
		solver.add_material_domain(compulsory_value<unsigned int>("id", material_spec), std::move(domain));
		return ;
	}
	}
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_TRADITIONAL_MATERIALS_HPP_
