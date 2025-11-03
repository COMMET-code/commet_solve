#ifndef INCLUDE_MATERIALS_ISO_MATERIALS_HPP_
#define INCLUDE_MATERIALS_ISO_MATERIALS_HPP_

#include "../../tensor_utils.hpp"
#include "iso_vol_hyperelastic_material.hpp"

namespace commet_solve
{

template <int dim, typename Number = double>
class Isihara : public IsoHyperelasticMaterial<dim, Number>
{
  public:
	Isihara(const Number &c1, const Number &c2, const Number &c3)
		: c1(c1)
		, c2(c2)
		, c3(c3) {};

	Isihara(Isihara &&) = delete;
	Isihara(const Isihara &) = delete;
	Isihara &operator=(Isihara &&) = delete;
	Isihara &operator=(const Isihara &) = delete;
	~Isihara() = default;

	void evaluate_iso(const Tensor<2, dim, Number> &F_iso,
					  const array<Tensor<1, dim, Number>, N_ORIENTATION_VECS> & /*orientation_vectors*/,
					  Number &strain_energy,
					  SymmetricTensor<2, dim, Number> &kirchhoff_stress,
					  SymmetricTensor<4, dim, Number> &spatial_stiffness) const final
	{
		using namespace Physics::Elasticity;

		const SymmetricTensor<2, dim, Number> B_iso = Physics::Elasticity::Kinematics::b(F_iso);
		const SymmetricTensor<2, dim, Number> B_iso2 = symmetrize(B_iso * Tensor<2, dim, Number>(B_iso));
		const Number I1 = trace(B_iso);
		const Number I2 = 0.5 * (I1 * I1 - trace(B_iso2));
		const Number I1_3 = I1 - dim;
		const SymmetricTensor<2, dim, Number> A2 = B_iso * I1 - B_iso2;

		const Number dPhi_dI1 = c1 + 2 * c3 * I1_3;
		const Number d2Phi_dI1 = 2 * c3;

		strain_energy += c1 * I1_3 + c2 * (I2 - dim) + c3 * I1_3 * I1_3;

		kirchhoff_stress += 2 * dPhi_dI1 * (B_iso - (I1 / 3.) * StandardTensors<dim>::I);
		kirchhoff_stress += 2 * c2 * (A2 - (2. * I2 / 3.) * StandardTensors<dim>::I);

		SymmetricTensor<4, dim> IoA_AoI = IoA_and_AoI<dim, Number>(B_iso);
		const SymmetricTensor<4, dim> &IxI = StandardTensors<dim>::IxI;
		const SymmetricTensor<4, dim> &IoI = StandardTensors<dim>::S;

		spatial_stiffness += 4 * dPhi_dI1 * (I1 * (IoI + IxI / 3) - IoA_AoI) / 3;
		spatial_stiffness += 4 * d2Phi_dI1 * (outer_product(B_iso, B_iso) - IoA_AoI * I1 / 3. + I1 * I1 * IxI / 9.);

		IoA_AoI = IoA_and_AoI<dim, Number>(A2);
		SymmetricTensor<4, dim> AA2 = outer_product(B_iso, B_iso) - odot(B_iso);
		spatial_stiffness += 4 * c2 * (-2. * (IoA_AoI - I2 * IoI) / 3. + 4. * I2 * IxI / 9.);
		spatial_stiffness += 4 * c2 * AA2;
	};

	const Number c1;
	const Number c2;
	const Number c3;
};

template <int dim, typename Number = double>
class HGO : public IsoHyperelasticMaterial<dim, Number>
{
  public:
	// HGO() = default;
	HGO(const Number & a,
        const Number & b,
        const array<Number, N_ORIENTATION_VECS>& as,
        const array<Number, N_ORIENTATION_VECS>& bs,
        const array<Number, N_ORIENTATION_VECS>& a_cross,
        const array<Number, N_ORIENTATION_VECS>& b_cross
     )
    : a(a)  
    , b(b)  
    , as(as)  
    , bs(bs)  
    , a_cross(a_cross)  
    , b_cross(b_cross)  
    {};
	HGO(HGO &&) = delete;
	HGO(const HGO &) = delete;
	HGO &operator=(HGO &&) = delete;
	HGO &operator=(const HGO &) = delete;
	~HGO() = default;

	void evaluate_iso(const Tensor<2, dim, Number> &F_iso,
					  const array<Tensor<1, dim, Number>, N_ORIENTATION_VECS> &orientation_vectors,
					  Number &strain_energy,
					  SymmetricTensor<2, dim, Number> &kirchhoff_stress,
					  SymmetricTensor<4, dim, Number> &spatial_stiffness) const final
	{
		using namespace Physics::Elasticity;
		const SymmetricTensor<4, dim> &IxI = StandardTensors<dim>::IxI;
		const SymmetricTensor<4, dim> &IoI = StandardTensors<dim>::S;

		const SymmetricTensor<2, dim, Number> B_iso = Physics::Elasticity::Kinematics::b(F_iso);
		const Number I1 = trace(B_iso);

		strain_energy += a * (exp(b * (I1 - dim)) - 1) / (2 * b);
		{
			const Number dPsi_dI1 = a * exp(b * (I1 - dim)) / 2;
			const Number d2Psi_dI1 = a * b * exp(b * (I1 - dim)) / 2;

			kirchhoff_stress += 2 * dPsi_dI1 * (B_iso - I1 * StandardTensors<dim>::I / 3);
			spatial_stiffness += 4 * (d2Psi_dI1 * outer_product(B_iso, B_iso) -
									  (d2Psi_dI1 * I1 + dPsi_dI1) * IoA_and_AoI<dim, Number>(B_iso) / 3 +
									  I1 * (d2Psi_dI1 * I1 + dPsi_dI1) * IxI / 9 + I1 * dPsi_dI1 * IoI / 3);
		}

		for (unsigned int i = 0; i < N_ORIENTATION_VECS; i++)
		{
			const Number &ai = as[i];
			const Number &bi = bs[i];

			const Tensor<1, dim> a_iso = F_iso * orientation_vectors[i];
			const Number I4 = a_iso.norm_square();
			const Number I4_1 = I4 - 1;
			if (I4_1 < 0)
				break;

			strain_energy += ai * (exp(bi * pow(I4_1, 2)) - 1) / (2 * bi);
			const Number dPsi_dI = ai * exp(bi * pow(I4_1, 2)) * I4_1;
			const Number d2Psi_dI = ai * exp(bi * pow(I4_1, 2)) * (1 + 2 * bi * pow(I4_1, 2));

			const SymmetricTensor<2, dim, Number> A = symmetrize(outer_product(a_iso, a_iso));
			kirchhoff_stress += 2 * dPsi_dI * (A - I4 * StandardTensors<dim>::I / 3);

			spatial_stiffness +=
				4 * (d2Psi_dI * outer_product(A, A) - (d2Psi_dI * I4 + dPsi_dI) * IoA_and_AoI<dim, Number>(A) / 3. +
					 I4 * (d2Psi_dI * I4 + dPsi_dI) * StandardTensors<dim>::IxI / 9. +
					 I4 * dPsi_dI * StandardTensors<dim>::S / 3.);
		}

        // Assert(N_ORIENTATION_VECS==3, ExcInternalError());
		for (unsigned int i = 0; i < N_ORIENTATION_VECS; i++){
            const unsigned int j = (i+1)%N_ORIENTATION_VECS;
			const Number &ai = a_cross[i];
			const Number &bi = b_cross[i];

			const Tensor<1, dim> a_i_iso = F_iso * orientation_vectors[i];
			const Tensor<1, dim> a_j_iso = F_iso * orientation_vectors[j];

			const Number I4 = a_i_iso*a_j_iso;
			strain_energy += ai * (exp(bi * pow(I4, 2)) - 1) / (2 * bi);
			const Number dPsi_dI = ai * exp(bi * pow(I4, 2)) * I4;
			const Number d2Psi_dI = ai * exp(bi * pow(I4, 2)) * (1 + 2 * bi * pow(I4, 2));
			const SymmetricTensor<2, dim, Number> A = symmetrize(outer_product(a_i_iso, a_j_iso));
			kirchhoff_stress += 2 * dPsi_dI * (A - I4 * StandardTensors<dim>::I / 3);

			spatial_stiffness +=
				4 * (d2Psi_dI * outer_product(A, A) - (d2Psi_dI * I4 + dPsi_dI) * IoA_and_AoI<dim, Number>(A) / 3. +
					 I4 * (d2Psi_dI * I4 + dPsi_dI) * StandardTensors<dim>::IxI / 9. +
					 I4 * dPsi_dI * StandardTensors<dim>::S / 3.);
        }

	}

  private:
	const Number a;
	const Number b;

	const array<Number, N_ORIENTATION_VECS> as;
	const array<Number, N_ORIENTATION_VECS> bs;

	// TODO come back to the cross terms
	const array<Number, N_ORIENTATION_VECS> a_cross;
	const array<Number, N_ORIENTATION_VECS> b_cross;

};

} // namespace commet_solve

#endif // INCLUDE_MATERIALS_ISO_MATERIALS_HPP_
