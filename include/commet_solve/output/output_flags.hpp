#ifndef INCLUDE_OUTPUT_OUTPUT_FLAGS_HPP_
#define INCLUDE_OUTPUT_OUTPUT_FLAGS_HPP_

#include <map>
#include <string>

namespace commet_solve
{

using namespace std;

template <typename u, typename v>
map<u, v> invert_map(const map<v, u> &in_map)
{
	map<u, v> out_map;
	for (const auto &[key, value] : in_map)
		out_map[value] = key;
	return out_map;
}

enum class scalar_output_flag
{
	jacobian,
	I1,
	I2,
	I3,
	I4_0,
	I5_0,
	I4_1,
	I5_1,
	I4_2,
	I5_2,
	I6_01,
	I6_12,
	I6_20,
	I4,
	I5,
	I6,
	I7,
	I8,
	I1bar,
	I2bar,
	I4bar,
	I5bar,
	I6bar,
	I7bar,
	I8bar,
	strain_energy,
	iso_strain_energy,
	vol_strain_energy
};

static const map<scalar_output_flag, string> SCALAR_OUTPUT_NAMES{
	{scalar_output_flag::jacobian, "jacobian"},
	{scalar_output_flag::I1, "I1"},
	{scalar_output_flag::I2, "I2"},
	{scalar_output_flag::I3, "I3"},
	{scalar_output_flag::I4, "I4"},
	{scalar_output_flag::I5, "I5"},
	{scalar_output_flag::I6, "I6"},
	{scalar_output_flag::I7, "I7"},
	{scalar_output_flag::I8, "I8"},
	{scalar_output_flag::I4_0, "I4_0"},
	{scalar_output_flag::I5_0, "I5_0"},
	{scalar_output_flag::I4_1, "I4_1"},
	{scalar_output_flag::I5_1, "I5_1"},
	{scalar_output_flag::I4_2, "I4_2"},
	{scalar_output_flag::I5_2, "I5_2"},
	{scalar_output_flag::I6_01, "I6_01"},
	{scalar_output_flag::I6_12, "I6_12"},
	{scalar_output_flag::I6_20, "I6_20"},
	{scalar_output_flag::I1bar, "I1bar"},
	{scalar_output_flag::I2bar, "I2bar"},
	{scalar_output_flag::I4bar, "I4bar"},
	{scalar_output_flag::I5bar, "I5bar"},
	{scalar_output_flag::I6bar, "I6bar"},
	{scalar_output_flag::I7bar, "I7bar"},
	{scalar_output_flag::I8bar, "I8bar"},
	{scalar_output_flag::strain_energy, "strain_energy"},
	{scalar_output_flag::iso_strain_energy, "iso_strain_energy"},
	{scalar_output_flag::vol_strain_energy, "vol_strain_energy"}};

static map<string, scalar_output_flag> INVERTED_SCALAR_OUTPUT_NAMES = invert_map(SCALAR_OUTPUT_NAMES);

enum class vector_output_flag
{
	principal_stress_0,
	principal_stress_1,
	principal_stress_2,
	principal_stretch_0,
	principal_stretch_1,
	principal_stretch_2,
	current_fibre_0,
	current_fibre_1,
	current_fibre_2
};

static const map<vector_output_flag, string> VECTOR_OUTPUT_NAMES{
	{vector_output_flag::principal_stress_0, "principal_stress_0"},
	{vector_output_flag::principal_stress_1, "principal_stress_1"},
	{vector_output_flag::principal_stress_2, "principal_stress_2"},
	{vector_output_flag::principal_stretch_0, "principal_stretch_0"},
	{vector_output_flag::principal_stretch_1, "principal_stretch_1"},
	{vector_output_flag::principal_stretch_2, "principal_stretch_2"},
	{vector_output_flag::current_fibre_0, "current_fibre_0"},
	{vector_output_flag::current_fibre_1, "current_fibre_1"},
	{vector_output_flag::current_fibre_2, "current_fibre_2"}};

static map<string, vector_output_flag> INVERTED_VECTOR_OUTPUT_NAMES = invert_map(VECTOR_OUTPUT_NAMES);

enum class tensor_output_flag
{					  //
	kirchhoff_stress, //
	B,				  //
	C,				  //
	F,
	E
};

static const map<tensor_output_flag, string> TENSOR_OUTPUT_NAMES{
	{tensor_output_flag::kirchhoff_stress, "kirchhoff_stress"}, //
	{tensor_output_flag::B, "B"},								//
	{tensor_output_flag::C, "C"},								//
	{tensor_output_flag::F, "F"},
	{tensor_output_flag::E, "E"}};

static map<string, tensor_output_flag> INVERTED_TENSOR_OUTPUT_NAMES = invert_map(TENSOR_OUTPUT_NAMES);

} // namespace commet_solve

#endif // INCLUDE_OUTPUT_OUTPUT_FLAGS_HPP_
