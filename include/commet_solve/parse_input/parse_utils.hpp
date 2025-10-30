#ifndef INCLUDE_PARSE_INPUT_PARSE_UTILS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_UTILS_HPP_

#include <array>
#include <deal.II/base/tensor.h>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

namespace commet_solve::parse
{

using namespace std;
using namespace dealii;
using json = nlohmann::json;

template <typename T>
string string_of_keys(const map<string, T> &map_data)
{
	string output = "{";
	for (auto &[k, v] : map_data)
	{
		output += k + ", ";
	}
	output = output.substr(0, output.size() - 2);
	output += "}";
	return output;
}

template <typename T>
bool contains_key(const string &key, const map<string, T> &map_data)
{

	if (map_data.find(key) != map_data.end())
		return true;

	return false;
}

template <typename T>
T json_key_to_map_value(const string &key, const json &data, const map<string, T> &map_data)
{

	if (not data.contains(key))
		throw std::logic_error("Missing required key '" + key + "' in input file section: " + to_string(data));

	const string value = data[key].get<string>();

	if (contains_key(value, map_data))
		return map_data.at(value);

	else
		throw std::logic_error("Error in input file section '" + to_string(data) + "'. \n" + "Value for '" + key +
							   "' is '" + value + "', but permissible values are: \n" + string_of_keys(map_data) + ".");
}

template <typename T>
T value_or_default(const string &key, const json &data, const T &default_value)
{
	if (data.contains(key))
		return data[key].get<T>();
	else
		return default_value;
}

template <typename T>
T compulsory_value(const string &key, const json &data)
{
	if (data.contains(key))
		return data[key].get<T>();
	else
		throw std::logic_error("Required key '" + key + "' is missing from file section: '" + to_string(data) + "'.");
}

template <int dim, typename Number, typename v_type>
v_type  compulsory_vector(const string &key, const json &data)
{
	vector<Number> vals = compulsory_value<vector<Number>>(key, data);

	if (vals.size() == dim)
	{
		v_type out;
		for (unsigned int i = 0; i < dim; i++)
			out[i] = vals.at(i);

		return out;
	}
	else
		throw std::logic_error("In file section '" + to_string(data) + "'. Number of entries for '" + key +
							   "' is : " + to_string(vals.size()) + " but it must be " + to_string(dim) + ".");
}

template <unsigned int array_size, typename Number>
array<Number, array_size> compulsory_array_zero_padded(const string &key, const json &data)
{
	vector<Number> vals = compulsory_value<vector<Number>>(key, data);
    array<Number, array_size> out = {{0.0}};

	if (vals.size() <= array_size)
	{
		for (unsigned int i = 0; i < vals.size(); i++)
			out[i] = vals.at(i);

		return out;
	}
	else
		throw std::logic_error("In file section '" + to_string(data) + "'. Number of entries for '" + key +
							   "' is : " + to_string(vals.size()) + " but it must be at most " + to_string(array_size) + ".");
}

} // namespace commet_solve::parse

#endif // INCLUDE_PARSE_INPUT_PARSE_UTILS_HPP_
