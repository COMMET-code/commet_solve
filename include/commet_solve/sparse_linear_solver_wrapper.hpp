
#ifndef INCLUDE_COMMET_SOLVE_SPARSE_LINEAR_SOLVER_WRAPPER_HPP_
#define INCLUDE_COMMET_SOLVE_SPARSE_LINEAR_SOLVER_WRAPPER_HPP_

#include "config.hpp"
// #include "logger.hpp"
#include "nlohmann/json_fwd.hpp"
#include <deal.II/lac/petsc_precondition.h>
#include <memory>

namespace commet_solve
{
using namespace std;
using namespace dealii;

using json = nlohmann::json;

enum class solver_type
{
	CG,
	GMRES,
	DIRECT
};

static const map<string, solver_type> SOLVER_TYPES({{"cg", solver_type::CG},
													{"gmres", solver_type::GMRES},
													{"direct", solver_type::DIRECT}});

enum class preconditioner_type
{
	IDENTITY,
	JACOBI,
	BLOCK_JACOBI,
	AMG
};

static const map<string, preconditioner_type> PRECONDITIONER_TYPES({{"identity", preconditioner_type::IDENTITY},
																	{"jacobi", preconditioner_type::JACOBI},
																	{"block_jacobi", preconditioner_type::BLOCK_JACOBI},
																	{"amg", preconditioner_type::AMG}});

template <int dim, typename Number = double>
class SparseSolverWrapper
{
  public:
	SparseSolverWrapper() = default;
	// SparseSolverWrapper(const json & args);
	SparseSolverWrapper(SparseSolverWrapper &&) = delete;
	SparseSolverWrapper(const SparseSolverWrapper &) = delete;
	SparseSolverWrapper &operator=(SparseSolverWrapper &&) = delete;
	SparseSolverWrapper &operator=(const SparseSolverWrapper &) = delete;
	~SparseSolverWrapper() = default;

	void set_settings(const json &settings);

	SolverControl solve(LA::MPI::SparseMatrix &A, LA::MPI::Vector &x, const LA::MPI::Vector &b);

  private:
	solver_type solver_setting = solver_type::CG;
	preconditioner_type preconditioner_setting = preconditioner_type::JACOBI;
	Number tol_mult = 1e-6;
	bool log_history = false;
	bool symmetric = false;

	unique_ptr<PETScWrappers::SolverBase> get_solver(SolverControl &solver_control)
	{
		switch (this->solver_setting)
		{
		case solver_type::CG:
			return make_unique<PETScWrappers::SolverCG>(solver_control);
		case solver_type::GMRES:
			return make_unique<PETScWrappers::SolverGMRES>(solver_control);
		case solver_type::DIRECT:
			return make_unique<PETScWrappers::SparseDirectMUMPS>(solver_control);
		default:
			throw std::runtime_error("The solver wrapper solver could not be retrieved.");
		}
	}

	unique_ptr<PETScWrappers::PreconditionBase> get_preconditioner(const LA::MPI::SparseMatrix &A)
	{
		switch (this->preconditioner_setting)
		{
		case preconditioner_type::IDENTITY:
			return make_unique<PETScWrappers::PreconditionNone>(A);
		case preconditioner_type::JACOBI:
			return make_unique<PETScWrappers::PreconditionJacobi>(A);
		case preconditioner_type::BLOCK_JACOBI:
			return make_unique<PETScWrappers::PreconditionBlockJacobi>(A);
		case preconditioner_type::AMG: {
			PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
			data.symmetric_operator = this->symmetric;
			data.relaxation_type_coarse =
				PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::SORJacobi;

			return make_unique<PETScWrappers::PreconditionBoomerAMG>(A, data);
		}
		default: {
			throw std::runtime_error("The solver wrapper preconditioner could not be retrieved.");
		}
		}
	}

	void set_solver_from_string(const string &solver)
	{
		try
		{
			this->solver_setting = SOLVER_TYPES.at(solver);
		}
		catch (const std::out_of_range &)
		{
			string available = "[";
			for (const auto &item : SOLVER_TYPES)
				available += item.first + ", ";
			available = available.substr(0, available.size() - 2);
			available += "]";

			// TODO this should probably throw something and exit
			// LOGGER.info("Trying to set solver type with to: '" + solver + "' but available solvers are: " +
			// available);
		}
	}

	void set_preconditioner_from_string(const string &preconditioner)
	{
		try
		{
			this->preconditioner_setting = PRECONDITIONER_TYPES.at(preconditioner);
		}
		catch (const std::out_of_range &)
		{
			string available = "[";
			for (const auto &item : PRECONDITIONER_TYPES)
				available += item.first + ", ";
			available = available.substr(0, available.size() - 2);
			available += "]";

			// TODO this should probably throw something and exit
			// LOGGER.info("Trying to set preconditioner type with to: '" + preconditioner + "' but available
			// preconditioners are: " + available);
		}
	}
};

template <int dim, typename Number>
void SparseSolverWrapper<dim, Number>::set_settings(const json &settings)
{

	if (settings.contains("solver_type"))
	{
		this->set_solver_from_string(settings["solver_type"].get<string>());
	}

	if (settings.contains("preconditioner_type"))
	{
		this->set_preconditioner_from_string(settings["preconditioner_type"].get<string>());
	}

	if (settings.contains("symmetric"))
	{
		this->symmetric = settings["symmetric"].get<bool>();
	}
}

template <int dim, typename Number>
SolverControl SparseSolverWrapper<dim, Number>::solve(LA::MPI::SparseMatrix &A,
													  LA::MPI::Vector &x,
													  const LA::MPI::Vector &b)
{
	SolverControl solver_control(A.m(), tol_mult * b.l2_norm(), log_history);
	if (this->solver_setting == solver_type::DIRECT)
	{
		PETScWrappers::SparseDirectMUMPS solver(solver_control);
		if (this->symmetric)
			solver.set_symmetric_mode(true);
		solver.solve(A, x, b);
		return solver_control;
	}
	else
	{
		unique_ptr<PETScWrappers::SolverBase> solver = this->get_solver(solver_control);

		unique_ptr<PETScWrappers::PreconditionBase> preconditioner = this->get_preconditioner(A);

		solver->solve(A, x, b, *preconditioner.get());
		return solver_control;
	}
}

// template <int dim, typename Number>
// SparseSolverWrapper<dim, Number>::SparseSolverWrapper(const json & args)
// {
// }

} // namespace fs_mechanics

#endif // INCLUDE_COMMET_SOLVE_SPARSE_LINEAR_SOLVER_WRAPPER_HPP_
