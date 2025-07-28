#ifndef INCLUDE_COMMET_SOLVE_UTILITIES_HPP_
#define INCLUDE_COMMET_SOLVE_UTILITIES_HPP_

#include <deal.II/base/types.h>
#include <utility>

#include "types.hpp"

namespace commet_solve
{

using namespace dealii;

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

} // namespace commet_solve

#endif // INCLUDE_COMMET_SOLVE_UTILITIES_HPP_
