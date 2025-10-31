#ifndef INCLUDE_COMMET_SOLVE_SOLVER_HPP_
#define INCLUDE_COMMET_SOLVE_SOLVER_HPP_

#include "boundary_conditions/dirichlet_bc.hpp"
#include "commet_solve/logger.hpp"
#include "config.hpp"
#include "fe_data/fe_data.hpp"
#include "material_domain/material_domain.hpp"
#include "output/output_flags.hpp"
#include "time.hpp"

#include <algorithm>
#include <deal.II/base/symmetric_tensor.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include <deal.II/numerics/data_out.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_component_interpretation.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <sstream>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/base/data_out_base.h>
#include <vector>

#include "stage.hpp"

#include "field/scalar_field.hpp"
#include "field/tensor_field.hpp"
#include "field/vector_field.hpp"
#include "field/vfield_manager.hpp"

#include "sparse_linear_solver_wrapper.hpp"

namespace commet_solve {

using namespace dealii;
using json = nlohmann::json;

template <int dim, typename Number = double> class FiniteStrainSolver {
public:
  FiniteStrainSolver(Triangulation<dim> *tri, Time<Number> &time,
                     const std::vector<types::coarse_cell_id>
                         &coarse_cell_index_to_coarse_cell_id,
                     const unsigned int &order = 1);
  FiniteStrainSolver(FiniteStrainSolver &&) = delete;
  FiniteStrainSolver(const FiniteStrainSolver &) = delete;
  FiniteStrainSolver &operator=(FiniteStrainSolver &&) = delete;
  FiniteStrainSolver &operator=(const FiniteStrainSolver &) = delete;
  ~FiniteStrainSolver() {
    solver_timer.print_summary();

    pcout << timer_stream.str() << "\n";

    // LOGGER.info("Compute time data:\n" + timer_stream.str());
  };

  void add_material_domain(const unsigned int &material_id,
                           std::unique_ptr<MaterialDomain<dim, Number>> mat) {
    material_domains[material_id] = move(mat);
  }

  void add_stage(std::unique_ptr<Stage<dim, Number>> stage) {
    this->stages.emplace_back(move(stage));
  }

  void add_scalar_output(const scalar_output_flag &flag) {
    scalar_outputs[flag] =
        LA::MPI::Vector(df_sca.locally_owned_dofs(), mpi_communicator);
  }
  void add_vector_output(const vector_output_flag &flag) {
    vector_outputs[flag] =
        LA::MPI::Vector(df_vec.locally_owned_dofs(), mpi_communicator);
  }
  void add_tensor_output(const tensor_output_flag &flag) {
    tensor_outputs[flag] =
        LA::MPI::Vector(df_ten.locally_owned_dofs(), mpi_communicator);
  }

  void initialize();
  void setup_system_with_constraints(Stage<dim, Number> *stage);
  void assemble_linear_system(Stage<dim, Number> *stage);
  void solve_linear_system_old();
  void solve_linear_system();
  void output();
  void nr(Stage<dim, Number> *stage);
  void increment_constraints(Stage<dim, Number> *stage);
  void homogeneous_constraints();
  void write_problem_size();
  void write_compute_times();
  void project_outputs();
  void solve();
  void set_output_vtus(const bool &value) { output_vtus_setting = value; };
  void set_write_mesh_data(const bool &value) {
    this->write_fe_data_setting = value;
  };
  bool get_write_mesh_data() const { return this->write_fe_data_setting; };
  void set_write_fe_data(const bool &value) {
    this->write_fe_data_setting = value;
  };
  void set_nr_threshold(const Number &new_threshold) {
    this->nr_threshold = new_threshold;
  };
  Number get_nr_threshold() { return this->nr_threshold; };
  void set_max_nr_iterations(const unsigned int &new_max_iterations) {
    this->max_nr_iterations = new_max_iterations;
  };
  unsigned int get_max_nr_iterations() { return this->max_nr_iterations; };

  void add_dbc(std::unique_ptr<DirichletBC<dim>> dbc) {
    dbcs.push_back(move(dbc));
  }

  void add_vector_field(const std::string &name,
                        vector<Tensor<1, dim, Number>> data) {
    vector_fields.add_field(&df.get_triangulation(), mapping,
                            quadrature_formula, name, data,
                            coarse_cell_index_to_coarse_cell_id);
  };
  void add_analytical_vector_field(const std::string &name,
                                   const std::string &x_expression,
                                   const std::string &y_expression,
                                   const std::string &z_expression) {
    vector_fields.add_analytical_field(&df.get_triangulation(), mapping,
                                       quadrature_formula, name, x_expression,
                                       y_expression, z_expression);
  };

  void set_sparse_solver_settings(const json &settings) {
    this->sparse_solver.set_settings(settings);
  };

private:
  std::vector<types::coarse_cell_id> coarse_cell_index_to_coarse_cell_id;
  DoFHandler<dim> df;
  DoFHandler<dim> df_sca;
  DoFHandler<dim> df_vec;
  DoFHandler<dim> df_ten;
  Time<Number> time;
  MPI_Comm mpi_communicator;
  const unsigned int pid;
  ConditionalOStream pcout;

  const hp::MappingCollection<dim> mapping;
  const hp::FECollection<dim> fe;
  const hp::QCollection<dim> quadrature_formula;

  const hp::FECollection<dim> fe_sca;
  const hp::FECollection<dim> fe_vec;
  const hp::FECollection<dim> fe_ten;

  // vector<FENumbering<dim>> fe_vec_numbering;

  std::map<unsigned int, std::unique_ptr<MaterialDomain<dim, Number>>>
      material_domains;
  std::vector<std::unique_ptr<DirichletBC<dim, Number>>> dbcs;
  std::vector<std::unique_ptr<Stage<dim, Number>>> stages;

  FEData<dim, Number> fe_data;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<Number> constraints;
  AffineConstraints<Number> dummy_constraints;

  SparseSolverWrapper<dim, Number> sparse_solver;
  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector locally_relevant_u;
  LA::MPI::Vector locally_owned_u;
  LA::MPI::Vector locally_owned_du;
  LA::MPI::Vector system_rhs;

  std::map<scalar_output_flag, LA::MPI::Vector> scalar_outputs;
  std::map<vector_output_flag, LA::MPI::Vector> vector_outputs;
  std::map<tensor_output_flag, LA::MPI::Vector> tensor_outputs;

  ScalarFieldManager<dim, Number> scalar_fields;
  VectorFieldManager<dim, Number> vector_fields;
  TensorFieldManager<dim, Number> tensor_fields;

  std::ostringstream timer_stream;
  TimerOutput solver_timer;
  bool output_vtus_setting = true;
  Number nr_threshold = 1e-6;
  unsigned int max_nr_iterations = 7;
  bool write_fe_data_setting = false;
  const string rel_mesh_data_pth = "./mesh_data";
  const string rel_field_data_pth = "./field_data";

  std::vector<std::pair<Number, std::string>> times_and_names;

  void write_fe_data();
  void write_fe_surface_data();
  void write_stage_constraints(const unsigned int &stage_number);
  void write_nodal_values();
  void write_nodal_forces();
};

template <int dim, typename Number>
FiniteStrainSolver<dim, Number>::FiniteStrainSolver(
    Triangulation<dim> *tri, Time<Number> &time,
    const std::vector<types::coarse_cell_id>
        &coarse_cell_index_to_coarse_cell_id,
    const unsigned int &order)
    : coarse_cell_index_to_coarse_cell_id(coarse_cell_index_to_coarse_cell_id),
      df(*tri), df_sca(*tri), df_vec(*tri), df_ten(*tri), time(time),
      mpi_communicator(MPI_COMM_WORLD),
      pid(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, (pid == 0)),
      mapping(MappingFE<dim>(FE_SimplexP<dim>(order)),
              MappingFE<dim>(FE_Q<dim>(order))),
      fe(FESystem<dim, dim>(FE_SimplexP<dim>(order), dim),
         FESystem<dim, dim>(FE_Q<dim>(order), dim)),
      quadrature_formula(QGaussSimplex<dim>(order + 1), QGauss<dim>(order + 1)),
      fe_sca(FE_SimplexDGP<dim>(order), FE_DGQ<dim>(order)),
      fe_vec(FESystem<dim, dim>(FE_SimplexDGP<dim>(order), dim),
             FESystem<dim, dim>(FE_DGQ<dim>(order), dim)),
      fe_ten(FESystem<dim, dim>(FE_SimplexDGP<dim>(order), dim * dim),
             FESystem<dim, dim>(FE_DGQ<dim>(order), dim * dim)),
      solver_timer(timer_stream, TimerOutput::summary,
                   TimerOutput::wall_times) {

  TimerOutput::Scope section(solver_timer, "FiniteStrainSolver_constructor");
  for (const auto &cell : df.active_cell_iterators())
    if (cell->is_locally_owned()) {
      if (cell->reference_cell() == ReferenceCells::Tetrahedron) {
        cell->set_active_fe_index(0);
        cell->as_dof_handler_iterator(df_sca)->set_active_fe_index(0);
        cell->as_dof_handler_iterator(df_vec)->set_active_fe_index(0);
        cell->as_dof_handler_iterator(df_ten)->set_active_fe_index(0);
      } else if (cell->reference_cell() == ReferenceCells::Hexahedron) {
        cell->set_active_fe_index(1);
        cell->as_dof_handler_iterator(df_sca)->set_active_fe_index(1);
        cell->as_dof_handler_iterator(df_vec)->set_active_fe_index(1);
        cell->as_dof_handler_iterator(df_ten)->set_active_fe_index(1);
      } else
        DEAL_II_NOT_IMPLEMENTED();
    }

  df.distribute_dofs(fe);
  df_sca.distribute_dofs(fe_sca);
  df_vec.distribute_dofs(fe_vec);
  df_ten.distribute_dofs(fe_ten);
  // pcout << "Number of elements: " <<
  // df.get_triangulation().n_global_active_cells() << std::endl; pcout <<
  // "Number of degrees of freedom: " << df.n_dofs() << std::endl;
  LOGGER.info("Number of elements: " +
              to_string(df.get_triangulation().n_global_active_cells()));
  LOGGER.info("Number of degrees of freedom: " + to_string(df.n_dofs()));

  locally_owned_dofs = df.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(df);

  locally_relevant_u.reinit(locally_owned_dofs, locally_relevant_dofs,
                            mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  locally_owned_du.reinit(locally_owned_dofs, mpi_communicator);
  locally_owned_u.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::solve() {
  this->write_problem_size();
  output();

  if (this->write_fe_data_setting)
    this->write_nodal_values();
  unsigned int count = 0;
  {

    TimerOutput::Scope section(solver_timer, "solve_problem_total");
    for (auto &stage : this->stages) {

      count++;

      LOGGER.info("================================================");
      LOGGER.info("============ Loading stage " + std::to_string(count) +
                  " ===================");
      LOGGER.info("================================================");

      this->time.set_end(stage->end_time);
      this->time.set_dt(stage->dt);

      this->setup_system_with_constraints(stage.get());
      if (this->write_fe_data_setting)
        this->write_stage_constraints(count);
      while (not time.finished()) {

        this->time.increment();

        LOGGER.info("Time step: " + to_string(time.get_timestep()) +
                    "\tTime: " + to_string(time.current()) +
                    "\t Delta t: " + to_string(time.get_delta_t()));

        this->nr(stage.get());
        this->output();
        if (this->write_fe_data_setting)
          this->write_nodal_values();
      }
    }
  }

  this->write_compute_times();
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::setup_system_with_constraints(
    Stage<dim, Number> *stage) {

  TimerOutput::Scope section(solver_timer, "setup_system_with_constraints");
  this->constraints.clear();
  this->constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  for (const auto &dbc : stage->get_dbcs())
    dbc->apply(df, constraints, time.current(), time.get_delta_t());

  // ComponentMask prescribed_indices(dim, true);

  // std::vector<double> prescribed_values_left(dim, 0.);
  // Functions::ConstantFunction<dim>
  // prescribed_function_left(prescribed_values_left);

  // VectorTools::interpolate_boundary_values(df, 1, prescribed_function_left,
  // constraints, prescribed_indices);

  // std::vector<double> prescribed_values_right(dim, 0.);
  // // prescribed_values_right.at(0) = 1;
  // // prescribed_values_right.at(0) = 0.5;
  // prescribed_values_right.at(0) = time.get_delta_t();
  // Functions::ConstantFunction<dim>
  // prescribed_function_right(prescribed_values_right);

  // VectorTools::interpolate_boundary_values(df, 4, prescribed_function_right,
  // constraints, prescribed_indices);

  this->constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(df, dsp, constraints,
                                  /*keep_constrained_dofs = */ false);
  // /*keep_constrained_dofs = */ true);
  // sparsity_pattern.copy_from(dsp);
  //
  SparsityTools::distribute_sparsity_pattern(
      dsp, df.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                       mpi_communicator);
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::increment_constraints(
    Stage<dim, Number> *stage) {

  TimerOutput::Scope section(solver_timer, "increment_constraints");
  this->constraints.clear();
  this->constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  for (const auto &dbc : stage->get_dbcs())
    dbc->apply(df, constraints, time.current(), time.get_delta_t());

  // ComponentMask prescribed_indices(dim, true);

  // std::vector<double> prescribed_values_left(dim, 0.);
  // Functions::ConstantFunction<dim>
  // prescribed_function_left(prescribed_values_left);
  // VectorTools::interpolate_boundary_values(df, 1, prescribed_function_left,
  // constraints, prescribed_indices);

  // std::vector<double> prescribed_values_right(dim, 0.);
  // prescribed_values_right.at(0) = time.get_delta_t();
  // Functions::ConstantFunction<dim>
  // prescribed_function_right(prescribed_values_right);
  // VectorTools::interpolate_boundary_values(df, 4, prescribed_function_right,
  // constraints, prescribed_indices);

  this->constraints.close();

  // DynamicSparsityPattern dsp(locally_relevant_dofs);
  // DoFTools::make_sparsity_pattern(df,
  // 								dsp,
  // 								constraints,
  // 								/*keep_constrained_dofs
  // = */ false);
  // // /*keep_constrained_dofs = */ true);
  // // sparsity_pattern.copy_from(dsp);
  // //
  // SparsityTools::distribute_sparsity_pattern(dsp, df.locally_owned_dofs(),
  // mpi_communicator, locally_relevant_dofs);
  // system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
  // mpi_communicator);
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::initialize() {

  TimerOutput::Scope section(solver_timer, "initialize");
  hp::FEValues<dim> hp_fe_values(mapping, fe, quadrature_formula,
                                 update_quadrature_points | update_values |
                                     update_gradients | update_JxW_values);
  hp::FEValues<dim> project_hp_fe_values(mapping, fe_sca, quadrature_formula,
                                         update_quadrature_points |
                                             update_values | update_gradients |
                                             update_JxW_values);

  // hp::FEValues<dim> sca_fe_values(mapping, fe_sca, quadrature_formula,
  // update_values | update_gradients | update_JxW_values); hp::FEValues<dim>
  // vec_fe_values(mapping, fe_sca, quadrature_formula, update_values |
  // update_gradients | update_JxW_values); hp::FEValues<dim>
  // ten_fe_values(mapping, fe_ten, quadrature_formula, update_values |
  // update_gradients | update_JxW_values);

  std::vector<types::global_dof_index> local_dof_indices;

  std::vector<types::global_dof_index> sca_dof_indices;
  std::vector<types::global_dof_index> vec_dof_indices;
  std::vector<types::global_dof_index> ten_dof_indices;

  for (const auto &cell : df.active_cell_iterators())
    if (cell->is_locally_owned()) {
      hp_fe_values.reinit(cell);

      const auto &sca_cell = cell->as_dof_handler_iterator(df_sca);
      const auto &vec_cell = cell->as_dof_handler_iterator(df_vec);
      const auto &ten_cell = cell->as_dof_handler_iterator(df_ten);

      project_hp_fe_values.reinit(sca_cell);

      types::global_cell_index cell_id = cell->global_active_cell_index();
      const auto &fe_values = hp_fe_values.get_present_fe_values();
      const auto &project_fe_values =
          project_hp_fe_values.get_present_fe_values();

      const unsigned int qps_per_cell =
          fe_values.quadrature_point_indices().size();
      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

      const unsigned int nodes_per_cell = dofs_per_cell / dim;

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      sca_dof_indices.resize(sca_cell->get_fe().n_dofs_per_cell());
      sca_cell->get_dof_indices(sca_dof_indices);

      vec_dof_indices.resize(vec_cell->get_fe().n_dofs_per_cell());
      vec_cell->get_dof_indices(vec_dof_indices);

      ten_dof_indices.resize(ten_cell->get_fe().n_dofs_per_cell());
      ten_cell->get_dof_indices(ten_dof_indices);

      std::vector<std::vector<Number>> N(qps_per_cell,
                                         std::vector<Number>(nodes_per_cell));
      std::vector<std::vector<Number>> project_N(
          qps_per_cell, std::vector<Number>(nodes_per_cell));

      std::vector<std::vector<Tensor<1, dim, Number>>> B(
          qps_per_cell, std::vector<Tensor<1, dim, Number>>(nodes_per_cell));

      this->material_domains.at(cell->material_id())
          ->add_entry(cell_id, *cell, fe_values, scalar_fields, vector_fields,
                      tensor_fields);

      for (const unsigned int qp : fe_values.quadrature_point_indices()) {
        for (unsigned int i = 0; i < nodes_per_cell; i++) {
          project_N.at(qp).at(i) = project_fe_values.shape_value(i, qp);

          const unsigned int i_dof = i * dim;
          N.at(qp).at(i) = fe_values.shape_value(i_dof, qp);
          B.at(qp).at(i) = fe_values.shape_grad(i_dof, qp);
        }
      }

      fe_data.add_cell(CellData<dim, Number>(
          cell_id, cell->active_fe_index(), cell->material_id(), nodes_per_cell,
          qps_per_cell, local_dof_indices, sca_dof_indices, vec_dof_indices,
          ten_dof_indices, fe_values.get_JxW_values(), project_N, N, B));
    }

  for (auto &[domain_id, material_domain] : this->material_domains) {
    material_domain->close();
  }

  for (auto &stage : this->stages)
    for (auto &nbc : stage->get_nbcs())
      nbc->initialize(df, mapping, fe, quadrature_formula);
  if (this->write_fe_data_setting)
    this->write_fe_data();
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::assemble_linear_system(
    Stage<dim, Number> *stage) {

  TimerOutput::Scope section(solver_timer, "assemble_linear_system");
  system_matrix = 0;
  system_rhs = 0;

  std::vector<Number> cell_u;
  std::vector<Tensor<1, dim, Number>> cell_u_vecs;

  std::vector<Tensor<1, dim, Number>> spatial_B;

  Tensor<2, dim, Number> F;
  Tensor<2, dim, Number> F_inv_T;
  Number psi;
  SymmetricTensor<2, dim, Number> tau;
  Tensor<3, dim, Number> intermediate;
  Number geom_stiffness;
  SymmetricTensor<4, dim, Number> cc;

  Tensor<1, dim, Number> node_res;
  Tensor<2, dim, Number> node_stiffness;

  FullMatrix<double> cell_matrix;
  Vector<double> cell_rhs;

  for (auto &nbc : stage->get_nbcs())
    nbc->apply(locally_relevant_u, constraints, system_matrix, system_rhs,
               time.current(), time.get_delta_t());

  {
    TimerOutput::Scope section_f(solver_timer,
                                 "assemble_linear_system_update_F");
    for (const CellData<dim, Number> &cell_data :
         this->fe_data.get_cell_data()) {
      std::unique_ptr<MaterialDomain<dim, Number>> &mat_domain =
          this->material_domains.at(cell_data.material_id);
      cell_u.resize(cell_data.n_nodes * dim);
      cell_u_vecs.resize(cell_data.n_nodes);

      locally_relevant_u.extract_subvector_to(cell_data.global_dofs, cell_u);

      for (unsigned int i_node = 0; i_node < cell_data.n_nodes; i_node++)
        for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
          cell_u_vecs.at(i_node)[i_comp] = cell_u[i_node * dim + i_comp];

      for (unsigned int qp = 0; qp < cell_data.n_qps; qp++) {
        F = Physics::Elasticity::StandardTensors<dim>::I;

        for (unsigned int i_node = 0; i_node < cell_data.n_nodes; i_node++)
          F += outer_product(cell_u_vecs.at(i_node),
                             cell_data.B.at(qp).at(i_node));

        mat_domain->update_F(cell_data.id, qp, F);
      }
    }
  }

  {
    TimerOutput::Scope section_c(solver_timer,
                                 "assemble_linear_system_constitutive_update");
    for (auto &[mat_id, mat_domain] : this->material_domains) {
      mat_domain->compute_constitutive_behaviour();
    }
  }

  {
    TimerOutput::Scope section_c(
        solver_timer, "assemble_linear_system_compute_el_contribution");
    for (const CellData<dim, Number> &cell_data :
         this->fe_data.get_cell_data()) {
      std::unique_ptr<MaterialDomain<dim, Number>> &mat_domain =
          this->material_domains.at(cell_data.material_id);
      cell_matrix.reinit(cell_data.n_nodes * dim, cell_data.n_nodes * dim);
      cell_rhs.reinit(cell_data.n_nodes * dim);
      spatial_B.resize(cell_data.n_nodes);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int qp = 0; qp < cell_data.n_qps; qp++) {
        // mat_domain.get_vals(cell_data.id, qp, F, psi, tau, cc);
        mat_domain->get_vals(cell_data.id, qp, F, psi, tau, cc);

        F_inv_T = invert(transpose(F));

        for (unsigned int i = 0; i < cell_data.n_nodes; i++)
          spatial_B.at(i) = F_inv_T * cell_data.B.at(qp).at(i);

        for (unsigned int i = 0; i < cell_data.n_nodes; i++) {

          node_res = tau * spatial_B.at(i) * cell_data.jxw.at(qp);
          for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
            cell_rhs(dim * i + i_comp) += -node_res[i_comp];

          intermediate = spatial_B.at(i) * cc * cell_data.jxw.at(qp);
          for (unsigned int j = 0; j < cell_data.n_nodes; j++) {

            node_stiffness = intermediate * spatial_B.at(j);
            geom_stiffness = spatial_B.at(j) * node_res;

            for (unsigned int i_comp = 0; i_comp < dim; i_comp++) {

              cell_matrix(dim * i + i_comp, dim * j + i_comp) +=
                  geom_stiffness + node_stiffness[i_comp][i_comp];
              for (unsigned int j_comp = i_comp + 1; j_comp < dim; j_comp++) {

                cell_matrix(dim * i + i_comp, dim * j + j_comp) +=
                    node_stiffness[i_comp][j_comp];
                cell_matrix(dim * j + j_comp, dim * i + i_comp) +=
                    node_stiffness[i_comp][j_comp];
              }
            }
          }
        }
      }

      // std::cout << "cell_data.global_dofs";
      // for (const auto &v : cell_data.global_dofs)
      //   std::cout << v;
      // std::cout << std::endl;

      constraints.distribute_local_to_global(
          // cell_matrix, cell_rhs, cell_data.global_dofs(), system_matrix,
          cell_matrix, cell_rhs, cell_data.global_dofs, system_matrix,
          system_rhs,
          /*use_inhomogeneities_for_rhs*/ false);
    }
  }

  {
    TimerOutput::Scope section_compress(solver_timer,
                                        "assemble_linear_system_compress");
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::solve_linear_system_old() {

  TimerOutput::Scope section(solver_timer, "solve_linear_system");
  const Number tol_mult = 1e-6;
  SolverControl solver_control(system_matrix.m(),
                               tol_mult * system_rhs.l2_norm());
  PETScWrappers::SolverCG solver(solver_control);
  PETScWrappers::PreconditionJacobi preconditioner(system_matrix);
  solver.solve(system_matrix, locally_owned_du, system_rhs, preconditioner);
  constraints.distribute(locally_owned_du);
  this->locally_owned_u += locally_owned_du;

  locally_relevant_u = locally_owned_u;
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::solve_linear_system() {

  TimerOutput::Scope section(solver_timer, "solve_linear_system");

  locally_owned_du = 0;
  constraints.distribute(locally_owned_du);

  // std::map<types::global_dof_index, double> boundary_values;
  // for (const auto &line : this->constraints.get_lines())
  // 	boundary_values.insert({line.index, line.inhomogeneity});
  // // VectorTools::interpolate_boundary_values(dof_handler, 0,
  // Functions::ZeroFunction<dim>(dim), boundary_values);
  // // MatrixTools::apply_boundary_values(boundary_values, system_matrix,
  // locally_owned_du, system_rhs, false);

  // this->write_system_to_file("pre-apply-bcs");

  // this->write_system_to_file("pre-solve");
  SolverControl res =
      this->sparse_solver.solve(system_matrix, locally_owned_du, system_rhs);
  // LOGGER.info("Iterations to solve linear system: " +
  // to_string(res.last_step()));

  constraints.distribute(locally_owned_du);
  locally_owned_u += locally_owned_du;

  locally_relevant_u = locally_owned_u;

  // const Number tol_mult = 1e-6;
  // SolverControl solver_control(system_matrix.m(), tol_mult *
  // system_rhs.l2_norm()); PETScWrappers::SolverCG solver(solver_control);
  // PETScWrappers::PreconditionJacobi preconditioner(system_matrix);
  // solver.solve(system_matrix, locally_owned_du, system_rhs, preconditioner);
  // constraints.distribute(locally_owned_du);
  // this->locally_owned_u += locally_owned_du;
  // locally_relevant_u = locally_owned_u;
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::output() {
  if (!output_vtus_setting)
    return;

  std::filesystem::create_directories(
      std::filesystem::path(rel_field_data_pth));

  this->project_outputs();
  TimerOutput::Scope section(solver_timer, "write_output");

  DataOut<dim> data_out;
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);
  std::vector<std::string> solution_names(dim, "u");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      tensor_interpretation(
          dim * dim, DataComponentInterpretation::component_is_part_of_tensor);

  data_out.add_data_vector(df, locally_relevant_u, solution_names,
                           interpretation);

  for (auto &[flag, vec] : this->scalar_outputs) {
    vec.compress(VectorOperation::add);
    data_out.add_data_vector(df_sca, vec, SCALAR_OUTPUT_NAMES.at(flag));
  }

  for (auto &[flag, vec] : this->vector_outputs) {
    vec.compress(VectorOperation::add);
    data_out.add_data_vector(df_vec, vec, VECTOR_OUTPUT_NAMES.at(flag),
                             interpretation);
  }

  for (auto &[flag, vec] : this->tensor_outputs) {
    vec.compress(VectorOperation::add);
    data_out.add_data_vector(df_ten, vec, TENSOR_OUTPUT_NAMES.at(flag),
                             tensor_interpretation);
  }

  vector_fields.ouput_fields(data_out);
  data_out.build_patches(mapping, fe.max_degree(),
                         DataOut<dim>::curved_inner_cells);
  // data_out.build_patches(mapping, 1, DataOut<dim>::curved_inner_cells);

  const unsigned int n_digits = 4;
  const std::string base_vtu_name = "solution";
  std::ostringstream ss;
  ss << std::setw(n_digits) << std::setfill('0') << time.get_timestep();
  const std::string rel_vtu_name = base_vtu_name + "_" + ss.str() + ".pvtu";
  const std::string name = "./" + base_vtu_name;
  data_out.write_vtu_with_pvtu_record(this->rel_field_data_pth + "/",
                                      base_vtu_name, time.get_timestep(),
                                      mpi_communicator, n_digits);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    times_and_names.emplace_back(time.current(), rel_vtu_name);
    const std::string pvd_name = this->rel_field_data_pth + "/solution.pvd";
    std::ofstream pvd_output(pvd_name);
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::nr(Stage<dim, Number> *stage) {
  this->increment_constraints(stage);
  this->assemble_linear_system(stage);
  const double initial_residual = this->system_rhs.l2_norm();

  LOGGER.info("Initial norm: " + format_float(initial_residual));

  this->solve_linear_system();

  unsigned int iteration = 1;
  if (iteration >= max_nr_iterations)
    return;

  this->homogeneous_constraints();
  this->assemble_linear_system(stage);

  double current_residual = this->system_rhs.l2_norm();

  LOGGER.info("Iteration: " + to_string(iteration) +
              "\tR: " + format_float(current_residual) +
              "\tR/R_0: " + format_float(current_residual / initial_residual));

  while (current_residual / initial_residual > this->nr_threshold &&
         iteration < max_nr_iterations) {

    this->solve_linear_system();
    this->assemble_linear_system(stage);
    current_residual = this->system_rhs.l2_norm();
    LOGGER.info("Iteration: " + to_string(iteration) +
                "\tR: " + format_float(current_residual) + "\tR/R_0: " +
                format_float(current_residual / initial_residual));
    iteration++;
  }
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::homogeneous_constraints() {

  this->dummy_constraints.clear();
  this->dummy_constraints.copy_from(this->constraints);
  this->constraints.clear();
  this->constraints.copy_from(this->dummy_constraints);

  for (const auto &line : this->constraints.get_lines())
    this->constraints.set_inhomogeneity(line.index, 0);

  this->constraints.close();
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::write_problem_size() {
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    json problem_size;
    problem_size["n_elements"] = df.get_triangulation().n_global_active_cells();
    problem_size["n_dofs"] = df.n_dofs();

    // TODO Consider refactoring out file name to a global const
    const std::string name = "./problem_size.json";
    // TODO consider_checking if the file already exists and thinking about if
    // it should be overwritten by default
    std::ofstream o_file(name);
    o_file << std::setw(4) << problem_size << std::endl;
  }
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::write_compute_times() {
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    json compute_times_output;

    for (const auto &item :
         solver_timer.get_summary_data(TimerOutput::total_wall_time))
      compute_times_output[item.first] = item.second;

    // const string name = output_path.string() + "/compute_times.json";
    const std::string name = "./compute_times.json";
    std::ofstream o_file(name);
    o_file << std::setw(4) << compute_times_output << std::endl;
  }
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::project_outputs() {

  TimerOutput::Scope section(solver_timer, "project_outputs");
  for (auto &[flag, vec] : this->scalar_outputs) {
    vec = 0;
    // Warning: this compress is very neccesary, see
    // https://dealii.org/current/doxygen/deal.II/classPETScWrappers_1_1MPI_1_1Vector.html
    vec.compress(VectorOperation::add);
  }
  for (auto &[flag, vec] : this->vector_outputs) {
    vec = 0;
    // Warning: this compress is very neccesary, see
    // https://dealii.org/current/doxygen/deal.II/classPETScWrappers_1_1MPI_1_1Vector.html
    vec.compress(VectorOperation::add);
  }
  for (auto &[flag, vec] : this->tensor_outputs) {
    vec = 0;
    // Warning: this compress is very neccesary, see
    // https://dealii.org/current/doxygen/deal.II/classPETScWrappers_1_1MPI_1_1Vector.html
    vec.compress(VectorOperation::add);
  }

  Vector<Number> sca_projection_vector;
  Vector<Number> vec_projection_vector;
  Vector<Number> ten_projection_vector;

  Vector<Number> rhs_vector;

  std::vector<Vector<Number>> vec_storage_vector(dim);
  std::vector<Vector<Number>> ten_storage_vector(dim * dim);

  std::vector<Number> sca_qp_vals;
  std::vector<Tensor<1, dim, Number>> vec_qp_vals;
  std::vector<Tensor<2, dim, Number>> ten_qp_vals;
  unsigned int last_n_qps = 0;
  unsigned int last_n_nodes = 0;

  for (const CellData<dim, Number> &cell_data : this->fe_data.get_cell_data()) {
    std::unique_ptr<MaterialDomain<dim, Number>> &mat_domain =
        this->material_domains.at(cell_data.material_id);
    if (cell_data.n_qps != last_n_qps || cell_data.n_nodes != last_n_nodes) {
      sca_qp_vals.resize(cell_data.n_qps);
      vec_qp_vals.resize(cell_data.n_qps);
      ten_qp_vals.resize(cell_data.n_qps);

      rhs_vector.reinit(cell_data.n_nodes);
      unsigned int count = 0;
      for (unsigned int i = 0; i < dim; i++) {
        vec_storage_vector.at(i).reinit(cell_data.n_nodes);
        for (unsigned int j = 0; j < dim; j++) {
          ten_storage_vector.at(count).reinit(cell_data.n_nodes);
          count++;
        }
      }

      sca_projection_vector.reinit(cell_data.sca_dofs.size());
      vec_projection_vector.reinit(cell_data.vec_dofs.size());
      ten_projection_vector.reinit(cell_data.ten_dofs.size());

      last_n_qps = cell_data.n_qps;
      last_n_nodes = cell_data.n_nodes;
    }

    for (auto &[flag, vec] : this->scalar_outputs) {
      // Get values at qps
      for (unsigned int qp = 0; qp < cell_data.n_qps; qp++)
        sca_qp_vals.at(qp) =
            mat_domain->get_scalar_value(cell_data.id, qp, flag);

      // Project qp values to nodes
      cell_data.project_scalar_values(sca_qp_vals, sca_projection_vector,
                                      rhs_vector);
      //
      // Add element projections to global vector
      for (unsigned int i = 0; i < sca_projection_vector.size(); i++)
        vec[cell_data.sca_dofs[i]] += sca_projection_vector[i];
    }
    for (auto &[flag, vec] : this->vector_outputs) {
      // Get values at qps
      for (unsigned int qp = 0; qp < cell_data.n_qps; qp++)
        vec_qp_vals.at(qp) =
            mat_domain->get_vector_value(cell_data.id, qp, flag);

      // Project qp values to nodes
      cell_data.project_vector_values(vec_qp_vals, vec_projection_vector,
                                      rhs_vector, vec_storage_vector,
                                      fe_vec[cell_data.fe_index]);

      // Add element projections to global vector
      for (unsigned int i = 0; i < vec_projection_vector.size(); i++)
        vec[cell_data.vec_dofs[i]] += vec_projection_vector[i];
    }
    for (auto &[flag, vec] : this->tensor_outputs) {

      // Get values at qps
      for (unsigned int qp = 0; qp < cell_data.n_qps; qp++)
        ten_qp_vals.at(qp) =
            mat_domain->get_tensor_value(cell_data.id, qp, flag);

      // Project qp values to nodes
      cell_data.project_tensor_values(ten_qp_vals, ten_projection_vector,
                                      rhs_vector, ten_storage_vector,
                                      fe_ten[cell_data.fe_index]);

      // Add element projections to global vector
      for (unsigned int i = 0; i < ten_projection_vector.size(); i++)
        vec[cell_data.ten_dofs[i]] += ten_projection_vector[i];
    }
  }
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::write_fe_data() {
  // const string rel_mesh_data_pth = "./mesh_data";

  std::filesystem::create_directories(std::filesystem::path(rel_mesh_data_pth));
  {

    json arr = json::array();

    for (const CellData<dim, Number> &c : this->fe_data.get_cell_data()) {
      auto j = json{{"id", c.id},
                    {"fe_index", c.fe_index},
                    {"material_id", c.material_id},
                    {"n_nodes", c.n_nodes},
                    {"n_qps", c.n_qps},
                    {"global_dofs", c.global_dofs},
                    {"jxw", c.jxw},
                    {"N", c.N},
                    {"B", json::array()}};

      for (const auto &qp_B : c.B) {
        json qp_B_json = json::array();
        for (const auto &node_B : qp_B)
          qp_B_json.push_back(tensor_to_json<dim, Number>(
              node_B)); // relies on Tensor<1,dim,Number> -> json overload
        j["B"].push_back(qp_B_json);
      }

      arr.push_back(j);
    }
    ofstream mesh_out(rel_mesh_data_pth + "/mesh_partition_" + to_string(pid) +
                      ".json");
    mesh_out << arr.dump(2);
    mesh_out.close();
  }
  {

    json arr = json::array();
    std::map<types::global_dof_index, Point<dim>> dof_locations =
        DoFTools::map_dofs_to_support_points(mapping, this->df);

    for (const auto &[dof_id, pt] : dof_locations)
      arr.push_back(json{{"dof_id", dof_id},
                         {"location", tensor_to_json<dim, Number>(pt)}});

    ofstream node_out(rel_mesh_data_pth + "/node_partition_" + to_string(pid) +
                      ".json");
    node_out << arr.dump(2);
    node_out.close();
  }
  this->write_fe_surface_data();
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::write_fe_surface_data() {
  // const string rel_mesh_data_pth = "./mesh_data";

  json arr = json::array();

  hp::FEFaceValues<dim> hp_fe_face_values(
      mapping, fe,
      hp::QCollection<dim - 1>(QGaussSimplex<dim - 1>(fe.max_degree() + 1),
                               QGauss<dim - 1>(fe.max_degree() + 1)),
      update_values | update_gradients | update_JxW_values |
          update_normal_vectors);

  for (const auto &cell : df.active_cell_iterators()) {
    if (cell->is_locally_owned() && cell->at_boundary()) {
      for (unsigned int i_face = 0; i_face < cell->n_faces(); i_face++) {
        const auto &face = cell->face(i_face);

        if (face->at_boundary()) {

          const unsigned int &face_boundary_id = face->boundary_id();

          hp_fe_face_values.reinit(cell, face);
          const auto &fe_face_values =
              hp_fe_face_values.get_present_fe_values();
          const Quadrature<dim - 1> &face_quadrature_formula =
              fe_face_values.get_quadrature();
          const unsigned int n_qps = face_quadrature_formula.size();
          const unsigned int dofs_per_face =
              cell->get_fe().n_dofs_per_face(i_face);
          const unsigned int nodes_per_face = dofs_per_face / 3;

          vector<global_dof_index> local_dof_indices(dofs_per_face);
          vector<Tensor<1, dim, Number>> normals(n_qps);
          vector<vector<Number>> N(n_qps, vector<Number>(nodes_per_face));
          vector<vector<Tensor<1, dim, Number>>> B(
              n_qps, vector<Tensor<1, dim, Number>>(nodes_per_face));

          face->get_dof_indices(local_dof_indices, cell->active_fe_index());

          for (unsigned int qp = 0; qp < n_qps; qp++) {
            normals.at(qp) = fe_face_values.normal_vector(qp);

            for (unsigned int i_node = 0; i_node < nodes_per_face; i_node++) {
              B.at(qp).at(i_node) = fe_face_values.shape_grad(
                  cell->get_fe().face_to_cell_index(i_node * dim, i_face), qp);

              N.at(qp).at(i_node) = fe_face_values.shape_value(
                  cell->get_fe().face_to_cell_index(i_node * dim, i_face), qp);
            }
          }

          auto j = json{
              {"boundary_id", face_boundary_id},
              {"n_qps", n_qps},
              {"global_dofs", local_dof_indices},
              {"jxw", fe_face_values.get_JxW_values()},
              {"normals", json::array()},
              {"N", N},
              {"B", json::array()},
          };

          for (const auto &qp_normal : normals) {
            j["normals"].push_back(tensor_to_json<dim, Number>(qp_normal));
          }

          for (const auto &qp_B : B) {
            json qp_B_json = json::array();
            for (const auto &node_B : qp_B)
              qp_B_json.push_back(tensor_to_json<dim, Number>(
                  node_B)); // relies on Tensor<1,dim,Number> -> json overload
            j["B"].push_back(qp_B_json);
          }

          arr.push_back(j);
        }
      }
    }
  }

  ofstream surface_out(rel_mesh_data_pth + "/boundary_partition_" +
                       to_string(pid) + ".json");
  surface_out << arr.dump(2);
  surface_out.close();
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::write_stage_constraints(
    const unsigned int &stage_number) {
  {

    json arr = json::array();
    for (const auto &line : constraints.get_lines())
      arr.push_back(line.index);

    ofstream constraint_lines(rel_mesh_data_pth + "/constraints_stage_" +
                              to_string(stage_number) + "_partition_" +
                              to_string(pid) + ".json");
    constraint_lines << arr.dump(2);
    constraint_lines.close();
  }
  if (this->pid == 0) {
    json dbc_stages = json::array();

    for (const auto &stage : this->stages) {
      json dbc_stage = json::array();
      for (const auto &dbc : stage->get_dbcs()) {
        dbc_stage.push_back(json{{"boundary_id", dbc->boundary_id},
                                 {"components", dbc->components}});
      }
      dbc_stages.push_back(dbc_stage);
    }
    ofstream dbc_settings_stream(rel_mesh_data_pth + "/dbcs.json");
    dbc_settings_stream << dbc_stages.dump(2);
    dbc_settings_stream.close();
  }
}

template <int dim, typename Number>
void FiniteStrainSolver<dim, Number>::write_nodal_values() {
  system_rhs = 0;

  std::vector<Number> cell_u;
  std::vector<Tensor<1, dim, Number>> cell_u_vecs;

  std::vector<Tensor<1, dim, Number>> spatial_B;

  Tensor<2, dim, Number> F;
  Tensor<2, dim, Number> F_inv_T;
  Number psi;
  SymmetricTensor<2, dim, Number> tau;
  Tensor<3, dim, Number> intermediate;
  SymmetricTensor<4, dim, Number> cc;

  Tensor<1, dim, Number> node_res;
  Tensor<2, dim, Number> node_stiffness;

  FullMatrix<double> cell_matrix;
  Vector<double> cell_rhs;

  {
    // TimerOutput::Scope section_c(solver_timer,
    // "assemble_linear_system_compute_el_contribution");
    for (const CellData<dim, Number> &cell_data :
         this->fe_data.get_cell_data()) {
      std::unique_ptr<MaterialDomain<dim, Number>> &mat_domain =
          this->material_domains.at(cell_data.material_id);
      cell_matrix.reinit(cell_data.n_nodes * dim, cell_data.n_nodes * dim);
      cell_rhs.reinit(cell_data.n_nodes * dim);
      spatial_B.resize(cell_data.n_nodes);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int qp = 0; qp < cell_data.n_qps; qp++) {
        mat_domain->get_vals(cell_data.id, qp, F, psi, tau, cc);

        F_inv_T = invert(transpose(F));

        for (unsigned int i = 0; i < cell_data.n_nodes; i++)
          spatial_B.at(i) = F_inv_T * cell_data.B.at(qp).at(i);

        for (unsigned int i = 0; i < cell_data.n_nodes; i++) {

          node_res = tau * spatial_B.at(i) * cell_data.jxw.at(qp);
          for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
            cell_rhs(dim * i + i_comp) += -node_res[i_comp];
        }
      }
      for (unsigned int i = 0; i < cell_data.global_dofs.size(); ++i)
        system_rhs(cell_data.global_dofs[i]) += cell_rhs(i);
    }
  }

  system_rhs.compress(VectorOperation::add);

  json arr = json::array();
  for (const types::global_dof_index &idx : this->locally_owned_dofs) {
    auto j = json{{"dof_idx", idx},
                  {"u", (double)this->locally_owned_u[idx]},
                  {"reaction", -(double)this->system_rhs[idx]}};
    arr.push_back(j);
  }

  const unsigned int n_digits = 4;
  std::ostringstream ss;
  ss << std::setw(n_digits) << std::setfill('0') << time.get_timestep();
  ofstream u_out(rel_mesh_data_pth + "/nodal_values_time_step_" + ss.str() +
                 "_partition_" + to_string(pid) + ".json");
  u_out << arr.dump(2);
  u_out.close();
}
} // namespace commet_solve

#endif // INCLUDE_COMMET_SOLVE_SOLVER_HPP_
