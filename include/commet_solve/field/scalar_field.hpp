#ifndef INCLUDE_FIELD_SCALAR_FIELD_HPP_
#define INCLUDE_FIELD_SCALAR_FIELD_HPP_

#include "../config.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/vector.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <stdexcept>


namespace commet_solve{

using namespace dealii;
using namespace std;

template<int dim, typename Number = double>
class ScalarField {
public:
    ScalarField() = default;
    ScalarField(ScalarField &&) = delete;
    ScalarField(const ScalarField &) = delete;
    ScalarField &operator=(ScalarField &&) = delete;
    ScalarField &operator=(const ScalarField &) = delete;
    ~ScalarField() = default;

	virtual void evaluate_field(const DoFCellAccessor<dim, dim, false> &cell,
								std::vector<Number> &qp_values,
								const std::string &field_name){};


private:
    
};

template<int dim, typename Number = double>
class ScalarFieldManager {
public:
    ScalarFieldManager() = default;
    ScalarFieldManager(ScalarFieldManager &&) = delete;
    ScalarFieldManager(const ScalarFieldManager &) = delete;
    ScalarFieldManager &operator=(ScalarFieldManager &&) = delete;
    ScalarFieldManager &operator=(const ScalarFieldManager &) = delete;
    ~ScalarFieldManager() = default;


	virtual void evaluate_field(const DoFCellAccessor<dim, dim, false> &/*cell*/,
								std::vector<Number> &/*qp_values*/,
								const std::string &/*field_name*/)
    {
        throw std::logic_error("ScalarFieldManager is not completed yet...");
    };



private:

    map<string, unique_ptr<ScalarField<dim, Number>>> fields;
    
};
            
            

}

#endif  // INCLUDE_FIELD_SCALAR_FIELD_HPP_

