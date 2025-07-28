#ifndef INCLUDE_MATERIAL_DOMAIN_ISOTROPIC_HYPERELASTIC_DOMAIN_HPP_
#define INCLUDE_MATERIAL_DOMAIN_ISOTROPIC_HYPERELASTIC_DOMAIN_HPP_

#include "material_domain.hpp"

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

namespace commet_solve {

template <int dim, typename Number = double>
class IsotropicHyperelasticDomain : public MaterialDomain<dim, Number> {
public:
  IsotropicHyperelasticDomain() = default;
  IsotropicHyperelasticDomain(IsotropicHyperelasticDomain &&) = delete;
  IsotropicHyperelasticDomain(const IsotropicHyperelasticDomain &) = delete;
  IsotropicHyperelasticDomain &
  operator=(IsotropicHyperelasticDomain &&) = delete;
  IsotropicHyperelasticDomain &
  operator=(const IsotropicHyperelasticDomain &) = delete;
  ~IsotropicHyperelasticDomain() = default;

  void add_entry(const dealii::types::global_cell_index &cell,
                 const unsigned int &qp) override {
    qp_data[{cell, qp}] = MinimalMaterialPointData<dim, Number>();
  };

  void update_F(const dealii::types::global_cell_index &cell,
                const unsigned int &qp,
                const Tensor<2, dim, Number> &F) override {
    qp_data.at({cell, qp}).F = F;
  };

  void get_vals(const dealii::types::global_cell_index &cell,
                const unsigned int &qp, Tensor<2, dim, Number> &F, Number &psi,
                SymmetricTensor<2, dim, Number> &tau,
                SymmetricTensor<4, dim, Number> &cc) override {
    const MinimalMaterialPointData<dim, Number> &point = qp_data.at({cell, qp});

    F = point.F;
    psi = point.psi;
    tau = point.tau;
    cc = point.cc;
  };

  void compute_constitutive_behaviour() override {
    for (auto &[point_key, point_data] : qp_data)
      this->constitutive_equation(point_data);
  };

protected:
  std::unordered_map<point_index, MinimalMaterialPointData<dim, Number>,
                     PointIndexHash>
      qp_data;

  virtual void
  constitutive_equation(MinimalMaterialPointData<dim, Number> &data) = 0;

private:
};

template <int dim, typename Number = double>
class NeoHookeanDomain : public IsotropicHyperelasticDomain<dim, Number> {
public:
  NeoHookeanDomain(const Number &lambda, const Number &mu)
      : lambda(lambda), mu(mu), lambda_2mu(lambda + 2 * mu) {};
  NeoHookeanDomain(NeoHookeanDomain &&) = delete;
  NeoHookeanDomain(const NeoHookeanDomain &) = delete;
  NeoHookeanDomain &operator=(NeoHookeanDomain &&) = delete;
  NeoHookeanDomain &operator=(const NeoHookeanDomain &) = delete;
  ~NeoHookeanDomain() = default;

protected:
  void
  constitutive_equation(MinimalMaterialPointData<dim, Number> &data) override {
    using namespace Physics::Elasticity;

    const SymmetricTensor<2, dim, Number> B = Kinematics::b(data.F);
    const Number I1 = dealii::trace(B);
    const Number I3 = determinant(B);

    data.psi = (2 * mu * (I1 - 3 - log(I3)) + lambda * (I3 - 1 - log(I3))) / 4.;
    data.tau = mu * B + (lambda * I3 - lambda_2mu) * StandardTensors<3>::I / 2;
    data.cc = lambda * I3 * StandardTensors<3>::IxI -
              (lambda * I3 - lambda_2mu) * StandardTensors<3>::S;
  };

private:
  const Number lambda, mu, lambda_2mu;
};

} // namespace commet_solve

#endif // INCLUDE_MATERIAL_DOMAIN_ISOTROPIC_HYPERELASTIC_DOMAIN_HPP_
