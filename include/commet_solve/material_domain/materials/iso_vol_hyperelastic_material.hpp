#ifndef INCLUDE_MATERIALS_ISO_VOL_HYPERELASTIC_MATERIAL_HPP_
#define INCLUDE_MATERIALS_ISO_VOL_HYPERELASTIC_MATERIAL_HPP_

#include "hyperelastic_material.hpp"
#include <memory>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

namespace commet_solve
{

template <int dim, typename Number = double>
class IsoHyperelasticMaterial
{
  public:
	IsoHyperelasticMaterial() = default;
	IsoHyperelasticMaterial(IsoHyperelasticMaterial &&) = delete;
	IsoHyperelasticMaterial(const IsoHyperelasticMaterial &) = delete;
	IsoHyperelasticMaterial &operator=(IsoHyperelasticMaterial &&) = delete;
	IsoHyperelasticMaterial &operator=(const IsoHyperelasticMaterial &) = delete;
	~IsoHyperelasticMaterial() = default;

	virtual void evaluate_iso(const Tensor<2, dim, Number> &F_iso,
							  const array<Tensor<1, dim, Number>, N_ORIENTATION_VECS> &orientation_vectors,
							  Number &strain_energy,
							  SymmetricTensor<2, dim, Number> &kirchhoff_stress,
							  SymmetricTensor<4, dim, Number> &spatial_stiffness) const;

  private:
};

template <typename Number = double>
class VolHyperelasticMaterial
{
  public:
	VolHyperelasticMaterial() = default;
	VolHyperelasticMaterial(VolHyperelasticMaterial &&) = delete;
	VolHyperelasticMaterial(const VolHyperelasticMaterial &) = delete;
	VolHyperelasticMaterial &operator=(VolHyperelasticMaterial &&) = delete;
	VolHyperelasticMaterial &operator=(const VolHyperelasticMaterial &) = delete;
	~VolHyperelasticMaterial() = default;

	virtual void evaluate_vol(const Number &J, Number &strain_energy, Number &denergy_dJ, Number &d2energy_dJ2) const;

  private:
};

template <int dim, typename Number = double>
class IsoVolHyperelasticMaterial final : public HyperelasticMaterial<dim, Number>
{
  public:
	IsoVolHyperelasticMaterial(unique_ptr<IsoHyperelasticMaterial<dim, Number>> iso,
							   unique_ptr<VolHyperelasticMaterial<Number>> vol)
		: iso(move(iso))
		, vol(move(vol)) {};
	IsoVolHyperelasticMaterial(IsoVolHyperelasticMaterial &&) = delete;
	IsoVolHyperelasticMaterial(const IsoVolHyperelasticMaterial &) = delete;
	IsoVolHyperelasticMaterial &operator=(IsoVolHyperelasticMaterial &&) = delete;
	IsoVolHyperelasticMaterial &operator=(const IsoVolHyperelasticMaterial &) = delete;
	~IsoVolHyperelasticMaterial() = default;

	void evaluate_model(const Tensor<2, dim, Number> &F,
						const array<Tensor<1, dim, Number>, N_ORIENTATION_VECS> &orientation_vectors,
						Number &strain_energy,
						SymmetricTensor<2, dim, Number> &kirchhoff_stress,
						SymmetricTensor<4, dim, Number> &spatial_stiffness) const final
	{
		using namespace Physics::Elasticity;


		const Number J = determinant(F);
		const Tensor<2, dim, Number> F_iso = pow(J, -1. / 3.) * F;

		strain_energy = 0;
		kirchhoff_stress = 0;
		spatial_stiffness = 0;

        this->iso->evaluate_iso(F_iso,
                                orientation_vectors,
                                strain_energy,
                                kirchhoff_stress,
                                spatial_stiffness);
        {
            Number denergy_dJ, d2energy_dJ2;
            this->vol->evaluate_vol(J, strain_energy, denergy_dJ, d2energy_dJ2); 

            kirchhoff_stress += J * denergy_dJ * StandardTensors<dim>::I; 
            spatial_stiffness += J * (denergy_dJ + J * d2energy_dJ2) * StandardTensors<dim>::IxI - 2 * J * denergy_dJ * StandardTensors<dim>::S;
        }

	};

  private:
	unique_ptr<IsoHyperelasticMaterial<dim, Number>> iso;
	unique_ptr<VolHyperelasticMaterial<Number>> vol;
};

} // namespace commet_solve

#endif // INCLUDE_MATERIALS_ISO_VOL_HYPERELASTIC_MATERIAL_HPP_
