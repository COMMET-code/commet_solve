#ifndef INCLUDE_COMMET_SOLVE_CONFIG_HPP_
#define INCLUDE_COMMET_SOLVE_CONFIG_HPP_

#include <deal.II/base/mpi.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace commet_solve{

using namespace dealii;
// #define SHOW_DEBUG_MESSAGES

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) &&                                             \
	!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA
//

static const unsigned int N_ORIENTATION_VECS=3;


}

#endif  // INCLUDE_COMMET_SOLVE_CONFIG_HPP_
