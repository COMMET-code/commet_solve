#ifndef INCLUDE_BOUNDARY_CONDITIONS_DIRICHLET_BC_HPP_
#define INCLUDE_BOUNDARY_CONDITIONS_DIRICHLET_BC_HPP_

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>

namespace commet_solve{

using namespace dealii;

 template<int dim, typename Number = double>
class DirichletBC {
public:
    DirichletBC() = default;
    DirichletBC(DirichletBC &&) = delete;
    DirichletBC(const DirichletBC &) = delete;
    DirichletBC &operator=(DirichletBC &&) = delete;
    DirichletBC &operator=(const DirichletBC &) = delete;
    ~DirichletBC() = default;

    virtual
    void apply(const DoFHandler<dim> &dof_handler,
               AffineConstraints<Number> &constraints,
               const Number &time,
               const Number &d_time)=0;

private:
    
};
            

}

#endif  // INCLUDE_BOUNDARY_CONDITIONS_DIRICHLET_BC_HPP_
