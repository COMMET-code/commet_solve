#ifndef INCLUDE_MATERIALS_NEOHOOKEAN_HPP_
#define INCLUDE_MATERIALS_NEOHOOKEAN_HPP_

#include "hyperelastic_material.hpp"
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

namespace commet_solve{

 template<int dim, typename Number = double>
class Neohookean final: public HyperelasticMaterial<dim, Number> {
public:
    Neohookean(const Number &lambda, const Number &mu) 
		: lambda(lambda)
		, mu(mu)
		, lambda_2mu(lambda + 2 * mu) {};
    Neohookean(Neohookean &&) = delete;
    Neohookean(const Neohookean &) = delete;
    Neohookean &operator=(Neohookean &&) = delete;
    Neohookean &operator=(const Neohookean &) = delete;
    ~Neohookean() = default;


    void evaluate_model(
        const Tensor<2, dim, Number>& F,
        const array<Tensor<1,dim,Number>, N_ORIENTATION_VECS> & /*orientation_vectors*/, 
        Number& strain_energy,
        SymmetricTensor<2, dim, Number>& kirchhoff_stress,
        SymmetricTensor<4, dim, Number>& spatial_stiffness
    ) const final{

		using namespace Physics::Elasticity;

		const SymmetricTensor<2, dim, Number> B = Kinematics::b(F);
		const Number I1 = dealii::trace(B); // Add namespace to avoid potential clash with torch namespace
		const Number I3 = determinant(B);

		strain_energy = (2 * mu * (I1 - 3 - log(I3)) + lambda * (I3 - 1 - log(I3))) / 4.;
		kirchhoff_stress = mu * B + (lambda * I3 - lambda_2mu) * StandardTensors<3>::I / 2;
		spatial_stiffness = lambda * I3 * StandardTensors<dim>::IxI - (lambda * I3 - lambda_2mu) * StandardTensors<dim>::S;

    };

private:
	const Number lambda, mu, lambda_2mu;
    
};
            



}

#endif  // INCLUDE_MATERIALS_NEOHOOKEAN_HPP_
