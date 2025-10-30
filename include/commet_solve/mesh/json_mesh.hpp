#ifndef INCLUDE_MESH_JSON_MESH_HPP_
#define INCLUDE_MESH_JSON_MESH_HPP_

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
// #include <deal.II/>

#include "../config.hpp"
#include "mesh.hpp"
#include <array>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_project.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <sstream>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include "../field/field_manager.hpp"
#include "../field/scalar_field.hpp"
#include "../field/tensor_field.hpp"
#include "../field/vector_field.hpp"

namespace commet_solve
{

using namespace dealii;
using namespace std;
using json = nlohmann::json;

const map<unsigned int, vector<unsigned int>> TET_FACE_TO_VERTS{
	{{0, {0, 1, 2}}, {1, {0, 1, 3}}, {2, {1, 2, 3}}, {3, {0, 2, 3}}}};

const map<unsigned int, vector<unsigned int>> HEX_FACE_TO_VERTS{
	{{0, {0, 2, 4, 6}}, {1, {1, 3, 5, 7}}, {2, {0, 4, 1, 5}}, {3, {2, 6, 3, 7}}, {4, {0, 1, 2, 3}}, {5, {4, 5, 6, 7}}}};

template <int dim>
void project_el_values_to_nodes(Triangulation<dim, dim> &tria)
{
	const int order = 1;
	// const hp::MappingCollection<dim> mapping(MappingFE<dim>(FE_SimplexP<dim>(1)), MappingFE<dim>(FE_Q<dim>(1)));
	const hp::MappingCollection<dim> mapping(MappingFE<dim>(FE_SimplexP<dim>(1)), MappingFE<dim>(FE_Q<dim>(1)));
	// const hp::FECollection<dim> fe(FESystem<dim, dim>(FE_SimplexP<dim>(order), dim),
	// 							   FESystem<dim, dim>(FE_Q<dim>(order), dim));
	const hp::FECollection<dim> fe(FE_DGP<dim>(1), FE_DGQ<dim>(1));
	const hp::QCollection<dim> quadrature_formula(QGaussSimplex<dim>(order + 1), QGauss<dim>(order + 1));

	AffineConstraints<double> constraints;

	DoFHandler<dim> df(tria);

	for (const auto &cell : df.active_cell_iterators())
		if (cell->is_locally_owned())
		{
			if (cell->reference_cell() == ReferenceCells::Tetrahedron)
				cell->set_active_fe_index(0);
			else if (cell->reference_cell() == ReferenceCells::Hexahedron)
				cell->set_active_fe_index(1);
			else
				DEAL_II_NOT_IMPLEMENTED();
		}

	df.distribute_dofs(fe);

	// constraints.open();
	constraints.close();
	// LA::MPI::Vector projected(df.locally_owned_dofs());
	LA::MPI::Vector projected;
	projected.reinit(df.locally_owned_dofs(), MPI_COMM_WORLD);

	VectorTools::project(
		MappingFE<dim>(FE_DGP<dim>(1)),
		df,
		constraints,
		QGaussSimplex<dim>(order),
		[&](const typename DoFHandler<dim>::active_cell_iterator &cell, const unsigned int q) -> double {
			// return qp_data.get_data(cell)[q]->density;
			return 2.0;
		},
		projected);

	DataOut<dim> dout;
	dout.add_data_vector(df, projected, "projected");
	dout.build_patches();
	ofstream dout_stream("projected.vtu");
	dout.write_vtu(dout_stream);
	dout_stream.close();
}

template <int dim>
unsigned int cell_to_n_face_verts(const dealii::CellData<dim> &cell)
{
	switch (cell.vertices.size())
	{
	case 8:
		return 4;
	case 4:
		return 3;
	default:
		throw std::logic_error("Cannot determine number of vertices for face on cell that has " +
							   to_string(cell.vertices.size()) + " vertices.");
	}
}

template <int dim>
vector<unsigned int> cell_and_face_to_local_vertices(const dealii::CellData<dim> &cell,
													 const unsigned int &local_face_number)
{
	switch (cell.vertices.size())
	{
	case 8:
		return HEX_FACE_TO_VERTS.at(local_face_number);
	case 4:
		return TET_FACE_TO_VERTS.at(local_face_number);
	default:
		throw std::logic_error("Cannot determine number of vertices for face on cell that has " +
							   to_string(cell.vertices.size()) + " vertices.");
	}
}

template <int dim, typename tri_type>
void field_test(tri_type *tri)
{

	json inp;
	ifstream("../resources/meshes/fudge_sv_1.json") >> inp;
	unsigned int order = 1;

	vector<Tensor<1, dim>> fibs;
	unsigned int count = 0;
	std::cout << "inp size: " << inp.size() << "\n";
	for (vector<double> vals : inp.get<vector<vector<double>>>())
	{
		fibs.push_back(Tensor<1, dim>({vals[0], vals[1], vals[2]}));
		count++;
	}

	// hp::MappingCollection<dim> mapping(MappingFE<dim>(FE_SimplexP<dim>(order)), MappingFE<dim>(FE_Q<dim>(order)));
	hp::MappingCollection<dim> mapping;
	mapping.push_back(MappingFE<dim>(FE_SimplexP<dim>(order)));
	mapping.push_back(MappingFE<dim>(FE_Q<dim>(order)));
	// const hp::FECollection<dim> fe(FESystem<dim, dim>(FE_SimplexP<dim>(order), dim), FESystem<dim,
	// dim>(FE_Q<dim>(order), dim));
	hp::QCollection<dim> quadrature_formula(QGaussSimplex<dim>(order + 1), QGauss<dim>(order + 1));
	ElementWiseVectorField<dim, double> v_field(tri, mapping, quadrature_formula);
	// v_field.add_field("fibre", fibs);

	// vector<Tensor<1, dim>> qp_vals;
	// std::cout << "Evaluating v field ...\n";
	// // v_field.evaluate_field(*tri->begin_active(), qp_vals, "fibre");
	// count = 0;
	// for (const auto &cell : v_field.df.active_cell_iterators())
	// {
	// 	v_field.evaluate_field(*cell, qp_vals, "fibre");
	// 	std::cout << "count: " << count << "\n";
	// 	for (auto &q_val : qp_vals)
	// 		std::cout << "q_val: " << q_val << "\n";
	// 	count++;
	// 	if (count > 10)
	// 		break;
	// }

	DataOut<dim> data_out;
	DataOutBase::VtkFlags flags;
	flags.write_higher_order_cells = true;
	data_out.set_flags(flags);
	std::vector<std::string> solution_names(dim, "fibre");

	std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(
		dim, DataComponentInterpretation::component_is_part_of_vector);
	data_out.add_data_vector(v_field.df, v_field.fields["fibre"], solution_names, interpretation);
	AnalyticalVectorField<dim, double> an_field("y", "x", "0", tri, mapping, quadrature_formula);
	an_field.ouput_field("an", data_out);

	data_out.build_patches(mapping, 1, DataOut<dim>::curved_inner_cells);

	const unsigned int n_digits = 4;
	const std::string base_vtu_name = "fibres";
	std::ostringstream ss;
	ss << std::setw(n_digits) << std::setfill('0') << 0;
	const std::string rel_vtu_name = base_vtu_name + "_" + ss.str() + ".pvtu";
	const std::string name = "./" + base_vtu_name;
	data_out.write_vtu_with_pvtu_record("./", base_vtu_name, 0, MPI_COMM_WORLD, n_digits);
}

template<int dim>
void triangulation_from_msh(Triangulation<dim, dim> &tria,
							 const std::string &path_to_file)
{

    GridIn<dim, dim> grid_in(tria);
    grid_in.read_msh(path_to_file);
    // output_triangulation_vtk<dim>("msh_tri", tria);

}

void triangulation_from_json(Triangulation<3, 3> &tria,
							 const std::string &path_to_file,
							 std::map<std::string, unsigned int> &b_id_map)
{
	const unsigned int dim = 3;
	json inp;
	ifstream(path_to_file.c_str()) >> inp;
	const unsigned int n_nodes = inp["nodes"].size();
	std::vector<Point<dim>> vertices(n_nodes);
	unsigned int count = 0;
	for (vector<double> vals : inp["nodes"].get<vector<vector<double>>>())
	{
		vertices.at(count) = Point<dim>(vals[0], vals[1], vals[2]);
		count++;
	}

	const unsigned int n_els = inp["elements"].size();
	std::vector<dealii::CellData<dim>> cells(n_els);
	count = 0;
	for (auto &it : inp["elements"])
	{
		const unsigned int n_verts = it["connectivity"].size();
		dealii::CellData<dim> new_cell(n_verts);
		new_cell.vertices = it["connectivity"].get<vector<unsigned int>>();
		new_cell.material_id = it["material_id"].get<unsigned int>();
		cells.at(count) = new_cell;
		count++;
	}
	GridTools::consistently_order_cells(cells);

	SubCellData face_data;
	unsigned int b_id = 1;
	for (auto bound_it = inp["boundaries"].begin(); bound_it != inp["boundaries"].end(); ++bound_it)
	{

		b_id_map[bound_it.key()] = b_id;
		json &bound = bound_it.value();
		// for (auto &it : bound)
		for (auto it = bound.begin(); it != bound.end(); ++it)
		{
			std::string key = it.key();
			const unsigned int element_number = (unsigned int)stoi(key);
			const unsigned int local_face_number = it.value().get<unsigned int>();

			const dealii::CellData<dim> &cell = cells.at(element_number);

			dealii::CellData<dim - 1> face(cell_to_n_face_verts(cell));
			face.boundary_id = b_id;
			// face.manifold_id = b_id;

			// const array<unsigned int, 3> el_verts = face_to_verts.at(val);
			const vector<unsigned int> &el_verts = cell_and_face_to_local_vertices(cell, local_face_number);

			for (unsigned int j = 0; j < el_verts.size(); ++j)
				face.vertices[j] = cell.vertices[el_verts[j]];

			face_data.boundary_quads.push_back(face);
		}

		b_id++;
	}

	tria.create_triangulation(vertices, cells, face_data);
}

void json_to_deal(Triangulation<3, 3> &tria, const std::string &path_to_file)
{
	const unsigned int dim = 3;
	json inp;
	ifstream(path_to_file.c_str()) >> inp;

	const unsigned int n_nodes = inp["nodes"].size();
	std::vector<Point<dim>> vertices(n_nodes);
	// unsigned int count = 0;
	for (auto it = inp["nodes"].begin(); it != inp["nodes"].end(); ++it)
	{
		std::string key = it.key();
		unsigned int key_i = (unsigned int)stoi(key);
		vector<double> vals = it.value().get<vector<double>>();
		vertices.at(key_i) = Point<dim>(vals[0], vals[1], vals[2]);
	}

	const unsigned int n_els = inp["elements"].size();
	std::vector<std::array<int, 4>> connectivity(n_els);
	for (auto it = inp["elements"].begin(); it != inp["elements"].end(); ++it)
	{
		std::string key = it.key();
		unsigned int key_i = (unsigned int)stoi(key);
		vector<int> vals = it.value().get<vector<int>>();

		connectivity.at(key_i) = std::array<int, 4>({{vals[0], vals[1], vals[2], vals[3]}});
	}

	vector<map<unsigned int, unsigned int>> boundaries;
	const map<unsigned int, array<unsigned int, 3>> face_to_verts{
		{1, {{0, 1, 2}}},
		{2, {{0, 1, 3}}},
		{3, {{1, 2, 3}}},
		{4, {{0, 2, 3}}},
	};
	SubCellData face_data;
	unsigned int b_id = 1;
	for (auto bound_it = inp["boundaries"].begin(); bound_it != inp["boundaries"].end(); ++bound_it)
	{
		json &bound = bound_it.value();
		for (auto it = bound.begin(); it != bound.end(); ++it)
		{
			dealii::CellData<dim - 1> face(3);
			face.boundary_id = b_id;
			face.manifold_id = b_id;

			std::string key = it.key();
			unsigned int key_i = (unsigned int)stoi(key);
			unsigned int val = it.value().get<unsigned int>();
			const array<unsigned int, 3> el_verts = face_to_verts.at(val);
			for (unsigned int j = 0; j < 3; ++j)
				face.vertices[j] = connectivity[key_i][el_verts[j]];

			face_data.boundary_quads.push_back(face);
		}

		b_id++;
	}
	// valves: 4, 5, 6, 7
	// outside: 1
	// lv: 2
	// rv: 3

	const unsigned int n_cells = connectivity.size();

	std::vector<dealii::CellData<dim>> cells(n_cells, dealii::CellData<dim>(4));
	for (unsigned int i = 0; i < n_cells; ++i)
	{
		for (unsigned int j = 0; j < connectivity[i].size(); ++j)
			cells[i].vertices[j] = connectivity[i][j];
		cells[i].material_id = 0;
	}

	GridTools::consistently_order_cells(cells);

	tria.create_triangulation(vertices, cells, face_data);
}

} // namespace commet_solve

#endif // INCLUDE_MESH_JSON_MESH_HPP_
