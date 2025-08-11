#ifndef INCLUDE_FIELD_TENSOR_FIELD_HPP_
#define INCLUDE_FIELD_TENSOR_FIELD_HPP_

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

namespace commet_solve
{
using namespace dealii;
using namespace std;

template <int dim, typename Number = double>
class TensorField
{
  public:TensorField
() = default;TensorField
(TensorField
 &&) = delete;TensorField
(const TensorField
 &) = delete;TensorField
 &operator=(TensorField
 &&) = delete;TensorField
 &operator=(const TensorField
 &) = delete;
	~TensorField
() = default;

	virtual void evaluate_field(const DoFCellAccessor<dim, dim, false> &cell,
								std::vector<Tensor<2, dim, Number>> &qp_values,
								const std::string &field_name);

  private:
};



template<int dim, typename Number = double>
class TensorFieldManager {
public:
    TensorFieldManager() = default;
    TensorFieldManager(TensorFieldManager &&) = delete;
    TensorFieldManager(const TensorFieldManager &) = delete;
    TensorFieldManager &operator=(TensorFieldManager &&) = delete;
    TensorFieldManager &operator=(const TensorFieldManager &) = delete;
    ~TensorFieldManager() = default;


	virtual void evaluate_field(const DoFCellAccessor<dim, dim, false> &/*cell*/,
								std::vector<Number> &/*qp_values*/,
								const std::string & /*field_name*/)
    {
        throw std::logic_error("TensorFieldManager is not completed yet...");
    };



private:

    map<string, unique_ptr<TensorField<dim, Number>>> fields;
    
};
            
            

} // namespace commet_solve

#endif // INCLUDE_FIELD_TENSOR_FIELD_HPP_
