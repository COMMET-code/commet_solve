#ifndef INCLUDE_MATERIALS_HYPERELASTIC_MATERIAL_HPP_
#define INCLUDE_MATERIALS_HYPERELASTIC_MATERIAL_HPP_

#include "../../config.hpp"
#include <array>
#include <deal.II/base/tensor.h>

namespace commet_solve{

using namespace dealii;
using namespace std;

template<int dim, typename Number = double>
class HyperelasticMaterial {
public:
    HyperelasticMaterial() = default;
    HyperelasticMaterial(HyperelasticMaterial &&) = delete;
    HyperelasticMaterial(const HyperelasticMaterial &) = delete;
    HyperelasticMaterial &operator=(HyperelasticMaterial &&) = delete;
    HyperelasticMaterial &operator=(const HyperelasticMaterial &) = delete;
    ~HyperelasticMaterial() = default;

    virtual
    void evaluate_model(
        const Tensor<2, dim, Number>& F,
        const array<Tensor<1,dim,Number>, N_ORIENTATION_VECS> & orientation_vectors, 
        Number& strain_energy,
        SymmetricTensor<2, dim, Number>& kirchhoff_stress,
        SymmetricTensor<4, dim, Number>& spatial_stiffness
    ) const;

private:
    
};
            



}

#endif  // INCLUDE_MATERIALS_HYPERELASTIC_MATERIAL_HPP_
