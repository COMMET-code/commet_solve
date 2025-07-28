#ifndef INCLUDE_COMMET_SOLVE_STAGE_HPP_
#define INCLUDE_COMMET_SOLVE_STAGE_HPP_

#include "boundary_conditions/dirichlet_bc.hpp"
#include <vector>

namespace commet_solve
{

template <int dim, typename Number = double>
class Stage
{
  public:
	Stage(const Number &end_time = 1, const Number &dt = 1)
		: end_time(end_time)
		, dt(dt) {};
	Stage(Stage &&) = delete;
	Stage(const Stage &) = delete;
	Stage &operator=(Stage &&) = delete;
	Stage &operator=(const Stage &) = delete;
	~Stage() = default;


	void add_dbc(std::unique_ptr<DirichletBC<dim>> dbc)
	{
		dbcs.push_back(move(dbc));
	}

	std::vector<std::unique_ptr<DirichletBC<dim, Number>>> &get_dbcs()
	{
		return dbcs;
	};

	const Number end_time;
	const Number dt;

  private:
	std::vector<std::unique_ptr<DirichletBC<dim, Number>>> dbcs;
};

} // namespace commet_solve

#endif // INCLUDE_COMMET_SOLVE_STAGE_HPP_
