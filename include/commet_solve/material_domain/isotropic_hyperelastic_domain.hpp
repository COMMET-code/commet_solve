#ifndef INCLUDE_MATERIAL_DOMAIN_ISOTROPIC_HYPERELASTIC_DOMAIN_HPP_
#define INCLUDE_MATERIAL_DOMAIN_ISOTROPIC_HYPERELASTIC_DOMAIN_HPP_

#include "../config.hpp"
#include "../logger.hpp"
#include "commet_solve/output/output_flags.hpp"
#include "material_domain.hpp"
#include "material_points.hpp"
#include "materials/hyperelastic_material.hpp"

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <memory>

namespace commet_solve {

template <int dim, typename Number = double>
class HyperelasticDomain : public MaterialDomain<dim, Number> {
public:
  HyperelasticDomain() = default;
  HyperelasticDomain(HyperelasticDomain &&) = delete;
  HyperelasticDomain(const HyperelasticDomain &) = delete;
  HyperelasticDomain &operator=(HyperelasticDomain &&) = delete;
  HyperelasticDomain &operator=(const HyperelasticDomain &) = delete;
  ~HyperelasticDomain() = default;

  void set_material_model(
      std::unique_ptr<HyperelasticMaterial<dim, Number>> material_model) {
    this->material_model = move(material_model);
  };

  void add_entry(const dealii::types::global_cell_index &cell,
                 const unsigned int &qp) override {
    qp_data[{cell, qp}] = HyperelasticMaterialPointData<dim, Number>();
  };

  void add_entry(const dealii::types::global_cell_index &cell_idx,
                 const DoFCellAccessor<dim, dim, false> &cell,
                 const FEValues<dim, dim> &fe_values,
                 ScalarFieldManager<dim, Number> & /*scalar_fields*/,
                 VectorFieldManager<dim, Number> &vector_fields,
                 TensorFieldManager<dim, Number> & /*tensor_fields*/) override {

    // const DoFCellAccessor<dim, dim, false> &cell,
    // 						std::vector<Tensor<1, dim,
    // Number>> &qp_values, 						const std::string &field_name

    vector<vector<Tensor<1, dim, Number>>> orientation_vectors(
        orientation_field_names.size(),
        vector<Tensor<1, dim, Number>>(fe_values.n_quadrature_points));
    // commet_solve::LOGGER.info("Getting field");
    for (unsigned int i = 0; i < orientation_field_names.size(); i++)
      vector_fields.evaluate_field(cell, orientation_vectors.at(i),
                                   orientation_field_names.at(i));

    for (unsigned int qp = 0; qp < fe_values.n_quadrature_points; qp++) {
      qp_data[{cell_idx, qp}] = HyperelasticMaterialPointData<dim, Number>();

      // commet_solve::LOGGER.info("Setting field");
      for (unsigned int i = 0; i < orientation_field_names.size(); i++)
        qp_data[{cell_idx, qp}].orientation_vectors[i] =
            orientation_vectors.at(i).at(qp);
    }
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
    const HyperelasticMaterialPointData<dim, Number> &point =
        qp_data.at({cell, qp});

    F = point.F;
    psi = point.psi;
    tau = point.tau;
    cc = point.cc;
  };

  // virtual std::ranges::view<Tensor<1, dim, Number>>
  std::span<const Tensor<1, dim, Number>>
  get_structural_vectors(const dealii::types::global_cell_index &cell,
                         const unsigned int &qp) override {
    // return std::views::take(qp_data.at({cell, qp}).orientation_vectors,
    // this->orientation_field_names.size());

    auto &vec = qp_data.at({cell, qp}).orientation_vectors;

    std::size_t n = std::min(vec.size(), this->orientation_field_names.size());

    return std::span<const Tensor<1, dim, Number>>(vec.data(), n);
  }

  virtual Number get_scalar_value(const dealii::types::global_cell_index &cell,
                                  const unsigned int &qp,
                                  const scalar_output_flag &flag) override {
    switch (flag) {
    case scalar_output_flag::jacobian:
      return determinant(qp_data.at({cell, qp}).F);
    case scalar_output_flag::strain_energy:
      return qp_data.at({cell, qp}).psi;
    case scalar_output_flag::I1:
      return trace(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F));
    case scalar_output_flag::I2: {
      const Tensor<2, dim, Number> b =
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F);
      return 0.5 * (pow(trace(b), 2) - trace(b * b));
    }
    case scalar_output_flag::I3:
      return pow(determinant(qp_data.at({cell, qp}).F), 2);
    default:
      return MaterialDomain<dim, Number>::get_scalar_value(cell, qp, flag);
    }
  };

  virtual Tensor<1, dim, Number>
  get_vector_value(const dealii::types::global_cell_index &cell,
                   const unsigned int &qp,
                   const vector_output_flag &flag) override {
    switch (flag) {
    case vector_output_flag::principal_stress_0: {
      auto vecs_and_vals = eigenvectors(
          qp_data.at({cell, qp}).tau, SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[0].first * vecs_and_vals[0].second /
             vecs_and_vals[0].second.norm();
    }
    case vector_output_flag::principal_stress_1: {
      auto vecs_and_vals = eigenvectors(
          qp_data.at({cell, qp}).tau, SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[1].first * vecs_and_vals[1].second /
             vecs_and_vals[1].second.norm();
    }
    case vector_output_flag::principal_stress_2: {
      auto vecs_and_vals = eigenvectors(
          qp_data.at({cell, qp}).tau, SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[2].first * vecs_and_vals[2].second /
             vecs_and_vals[2].second.norm();
    }
    case vector_output_flag::principal_stretch_0: {
      auto vecs_and_vals = eigenvectors(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F),
          SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[0].first * vecs_and_vals[0].second /
             vecs_and_vals[0].second.norm();
    }
    case vector_output_flag::principal_stretch_1: {
      auto vecs_and_vals = eigenvectors(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F),
          SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[1].first * vecs_and_vals[1].second /
             vecs_and_vals[1].second.norm();
    }
    case vector_output_flag::principal_stretch_2: {
      auto vecs_and_vals = eigenvectors(
          Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F),
          SymmetricTensorEigenvectorMethod::jacobi);
      return vecs_and_vals[2].first * vecs_and_vals[2].second /
             vecs_and_vals[2].second.norm();
    }
    case vector_output_flag::current_fibre_0: {
      return qp_data.at({cell, qp}).F *
             qp_data.at({cell, qp}).orientation_vectors[0];
    }
    default:
      return MaterialDomain<dim, Number>::get_vector_value(cell, qp, flag);
    }
  };

  virtual Tensor<2, dim, Number>
  get_tensor_value(const dealii::types::global_cell_index &cell,
                   const unsigned int &qp,
                   const tensor_output_flag &flag) override {
    switch (flag) {
    case tensor_output_flag::kirchhoff_stress: {
      return qp_data.at({cell, qp}).tau;
    }
    case tensor_output_flag::F: {
      return qp_data.at({cell, qp}).F;
    }
    case tensor_output_flag::B: {
      return Physics::Elasticity::Kinematics::b(qp_data.at({cell, qp}).F);
    }
    case tensor_output_flag::C: {
      return Physics::Elasticity::Kinematics::C(qp_data.at({cell, qp}).F);
    }
    case tensor_output_flag::E: {
      return Physics::Elasticity::Kinematics::E(qp_data.at({cell, qp}).F);
    }
    default:
      return MaterialDomain<dim, Number>::get_tensor_value(cell, qp, flag);
    }
  };

  void compute_constitutive_behaviour() override {
    for (auto &[point_key, point_data] : qp_data)
      this->constitutive_equation(point_data);
  };

  vector<string> orientation_field_names;

protected:
  std::unordered_map<point_index, HyperelasticMaterialPointData<dim, Number>,
                     PointIndexHash>
      qp_data;
  std::unique_ptr<HyperelasticMaterial<dim, Number>> material_model;
  virtual void
  constitutive_equation(HyperelasticMaterialPointData<dim, Number> &data) {
    material_model->evaluate_model(data.F, data.orientation_vectors, data.psi,
                                   data.tau, data.cc);
  };

private:
};

template <int dim, typename Number = double>
class NeoHookeanDomain : public HyperelasticDomain<dim, Number> {
public:
  NeoHookeanDomain(const Number &lambda, const Number &mu)
      : lambda(lambda), mu(mu), lambda_2mu(lambda + 2 * mu) {};
  NeoHookeanDomain(NeoHookeanDomain &&) = delete;
  NeoHookeanDomain(const NeoHookeanDomain &) = delete;
  NeoHookeanDomain &operator=(NeoHookeanDomain &&) = delete;
  NeoHookeanDomain &operator=(const NeoHookeanDomain &) = delete;
  ~NeoHookeanDomain() = default;

protected:
  void constitutive_equation(
      HyperelasticMaterialPointData<dim, Number> &data) override {
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
