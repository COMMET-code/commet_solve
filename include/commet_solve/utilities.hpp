#ifndef INCLUDE_COMMET_SOLVE_UTILITIES_HPP_
#define INCLUDE_COMMET_SOLVE_UTILITIES_HPP_

#include <deal.II/base/types.h>
#include <utility>
#include <nlohmann/json.hpp>

#include <fmt/core.h>
#include "types.hpp"

namespace commet_solve
{

using namespace dealii;
using json=nlohmann::json;

struct PointIndexHash
{
	std::size_t operator()(const point_index &p) const
	{
		// Simple but effective hashing for pairs of integers
		return std::hash<types::global_cell_index>{}(p.first) ^ (std::hash<unsigned int>{}(p.second) << 1);
	}
};

template <typename Number = double>
bool almost_equals(const Number &v1, const Number &v2, const double &eps = 1e-7)
{
	return fabs(v1 - v2) < eps;
}

// Define how to serialize a Tensor<1, dim, Number> (assuming it behaves like std::array<Number, dim>)
template <int dim, typename Number>
json tensor_to_json(const Tensor<1, dim, Number> &tensor)
{
    auto j = json::array();
    for (unsigned int i = 0; i < dim; ++i)
        j.push_back(tensor[i]);
    return j;
}

template <typename Number = double>
inline std::string format_float(const Number & val){
    return fmt::format("{:.2e}", val);
}

} // namespace commet_solve

#endif // INCLUDE_COMMET_SOLVE_UTILITIES_HPP_
