#ifndef INCLUDE_MATERIAL_DOMAIN_MATERIAL_POINTS_HPP_
#define INCLUDE_MATERIAL_DOMAIN_MATERIAL_POINTS_HPP_


#include "commet_solve/config.hpp"
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>


namespace commet_solve{
using namespace std;
using namespace dealii;

template <int dim, typename Number = double>
struct MaterialPointData
{
	unsigned int el_id;
	unsigned int qp;
    std::array<Tensor<1, dim, Number>, N_ORIENTATION_VECS> orientation_vectors;
	Point<dim, Number> position;
};


template <int dim, typename Number = double>
struct HyperelasticMaterialPointData : MaterialPointData<dim, Number>
{
	Number psi;
	Tensor<2, dim, Number> F;
	SymmetricTensor<2, dim, Number> tau;
	SymmetricTensor<4, dim, Number> cc;
};



}

#endif  // INCLUDE_MATERIAL_DOMAIN_MATERIAL_POINTS_HPP_
