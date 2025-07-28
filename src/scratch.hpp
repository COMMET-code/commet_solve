
void first_test()
{

	std::cout << "Hello!" << std::endl;
	// std::cout << "This is F:" << std::endl;
	commet_solve::NeoHookeanDomain<3, double> nh_domain(1, 1);
	nh_domain.add_entry(0, 0);

	dealii::Tensor<2, 3> F;
	unsigned int dim = 3;

	for (unsigned int i = 0; i < dim; i++)
	{
		F[i][i] = 1;
	}

	F[0][0] = 2;
	nh_domain.update_F(0, 0, F);
	nh_domain.compute_constitutive_behaviour();

	double read_psi;
	dealii::Tensor<2, 3> read_F;
	dealii::SymmetricTensor<2, 3> read_tau;
	dealii::SymmetricTensor<4, 3> read_cc;

	nh_domain.get_vals(0, 0, read_F, read_psi, read_tau, read_cc);

	std::cout << "This is F:" << std::endl;
	std::cout << read_F << std::endl;

	std::cout << "This is psi:" << std::endl;
	std::cout << read_psi << std::endl;

	std::cout << "This is tau:" << std::endl;
	std::cout << read_tau << std::endl;

	std::cout << "This is cc:" << std::endl;
	std::cout << read_cc << std::endl;
}

template <typename Number = double>
bool almost_equals(const Number &v1, const Number &v2, const double &eps = 1e-7)
{
	return fabs(v1 - v2) < eps;
}

// template <unsigned int dim, typename tri_type>
// void set_rectangular_boundary_ids(tri_type &triangulation)
// {
// 	BoundingBox<dim> bounding_box = GridTools::compute_bounding_box(triangulation);

// 	for (const auto &cell : triangulation.active_cell_iterators())
// 	{
// 		if (cell->at_boundary())
// 			for (const auto &face : cell->face_iterators())
// 			{
// 				if (face->at_boundary())
// 					for (unsigned int i = 0; i < dim; i++)
// 					{
// 						if (almost_equals(face->center()[i], bounding_box.lower_bound(i)))
// 							face->set_boundary_id(i + 1);
// 						else if (almost_equals(face->center()[i], bounding_box.upper_bound(i)))
// 							face->set_boundary_id(i + 1 + dim);
// 					}
// 			}
// 	}
// }

void second_test()
{

	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	GridGenerator::hyper_cube(tri);
	tri.refine_global(8);
	commet_solve::set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .2);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);

	solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	solver.initialize();
	solver.setup_system_with_constraints();
	solver.assemble_linear_system();
	solver.solve_linear_system();
	solver.output();
}

void third_test()
{
	const unsigned int dim = 3;
	// const unsigned int batch_size = 3;
	const unsigned int batch_size = 3;
	const unsigned int n_fibres = 0;
	const at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);

	commet_solve::VectorizedMaterialDomain<dim> dom;
	// dom.set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
	dom.load_model(
		"/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
	//
	dom.load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/"
				   "opt-neohookean-micnn-cpu.torchscript");

	torch::Tensor F = torch::zeros({batch_size, dim, dim}, options);
	// for(unsigned int i=0; i<dim; i++){
	//     F.index({torch::indexing::Slice(), i, i}) = 1;
	// }

	using t_Slice = torch::indexing::Slice;
	F.index_put_({t_Slice(), t_Slice(), t_Slice()}, torch::eye(dim));
	F = F + 0.02 * torch::randn({batch_size, dim, dim}, options);
	torch::Tensor structural_tensors = torch::randn({batch_size, n_fibres, dim}, options);

	torch::Tensor strain_energy = torch::zeros({batch_size}, options);
	torch::Tensor k_stress = torch::zeros({batch_size, dim, dim}, options);
	torch::Tensor spatial_tangent = torch::zeros({batch_size, dim, dim, dim, dim}, options);

	dom.evaluate_model(F, structural_tensors, strain_energy, k_stress, spatial_tangent);

	//     dom.set_evaluation_method(commet_solve::NCMEvaluationMethod::USING_F);
	//     dom.evaluate_model(F, structural_tensors, strain_energy, k_stress, spatial_tangent);

	//     dom.set_evaluation_method(commet_solve::NCMEvaluationMethod::USING_C);

	//     torch::Tensor strain_energy_c = torch::zeros({batch_size}, options);
	// torch::Tensor k_stress_c = torch::zeros({batch_size, dim, dim}, options);
	// torch::Tensor spatial_tangent_c = torch::zeros({batch_size, dim, dim, dim, dim}, options);

	//     dom.evaluate_model(F, structural_tensors, strain_energy_c, k_stress_c, spatial_tangent_c);

	// mm.get_methods();
	//
	std::cout << "very nice..." << std::endl;
	std::cout << "spatial_tangent: \n" << spatial_tangent << std::endl;
	std::cout << "k_stress: \n" << k_stress << std::endl;
	std::cout << "strain_energy: \n" << strain_energy << std::endl;

	dealii::Tensor<2, dim, double> tau;
	dealii::SymmetricTensor<2, dim> tau_sim;
	dealii::SymmetricTensor<4, dim> cc;

	switch (commet_solve::determine_tensor_layout(k_stress))
	{
	case commet_solve::TensorLayout::STANDARD: {
		std::cout << "k_stress_layout is standard" << std::endl;
		break;
	}
	case commet_solve::TensorLayout::VOIGT: {
		std::cout << "k_stress_layout is voigt" << std::endl;
		break;
	}
	}

	commet_solve::TensorLayout layout = commet_solve::determine_tensor_layout(k_stress);
	commet_solve::torch_to_deal_tensor<dim, double>(0, k_stress, tau, layout);
	commet_solve::torch_to_deal_tensor<dim, double>(0, k_stress, tau_sim, layout);
	commet_solve::torch_to_deal_tensor<dim, double>(1, spatial_tangent, cc, layout);

	// std::cout << "tau:\n" ;
	// for(unsigned int i=0; i<dim; i++){
	//     for(unsigned int j=0; j<dim; j++){
	//         std::cout << tau[i][j] << ",  ";
	//     }
	//     std::cout << "\n";
	// }
	// std::cout << "tau_sim:\n" ;
	// for(unsigned int i=0; i<dim; i++){
	//     for(unsigned int j=0; j<dim; j++){
	//         std::cout << tau_sim[i][j] << ",  ";
	//     }
	//     std::cout << "\n";
	// }
	std::cout << "tau:\n" << tau << std::endl;
	std::cout << "tau_sim:\n" << tau_sim << std::endl;
	std::cout << "cc:\n" << cc << std::endl;

	double dif = 0;
	for (unsigned int i = 0; i < dim; i++)
	{
		for (unsigned int j = 0; j < dim; j++)
		{
			for (unsigned int k = 0; k < dim; k++)
			{
				for (unsigned int l = 0; l < dim; l++)
				{
					dif += std::pow((double)cc[i][j][k][l] - spatial_tangent[1][i][j][k][l].item<double>(), 2);
				}
			}
		}
	}

	std::cout << "dff: " << dif << std::endl;

	dealii::Tensor<2, dim> F_deal;
	for (unsigned int i = 0; i < dim; i++)
	{
		F_deal[i][i] = 1;
	}

	// torch::Tensor F_torch = torch::zeros({batch_size, dim, dim}, options);

	commet_solve::deal_to_torch_tensor<dim, double>(0, F_deal, F, commet_solve::TensorLayout::STANDARD);

	std::cout << "F:\n" << F << std::endl;

	// std::cout << "very nice..." << std::endl;
	// std::cout << "spatial_tangent: \n" << spatial_tangent << std::endl;
	// std::cout << "spatial_tangent_c :\n" << spatial_tangent_c << std::endl;
	// std::cout << "k_stress: \n" << k_stress << std::endl;
	// std::cout << "k_stress_c :\n" << k_stress_c << std::endl;
	// std::cout << "strain_energy: \n" << strain_energy << std::endl;
	// std::cout << "strain_energy_c: \n" << strain_energy_c << std::endl;

	// std::cout << "(spatial_tangent-spatial_tangent_c).norm() : \n" << (spatial_tangent-spatial_tangent_c).norm() <<
	// std::endl; std::cout << " (k_stress-k_stress_c).norm() : \n" << (k_stress-k_stress_c).norm() << std::endl;
	// std::cout << "(strain_energy-strain_energy_c).norm() : \n" << (strain_energy-strain_energy_c).norm() <<
	// std::endl;

	// std::cout << mm.get_methods();
}

void fourth_test()
{

	std::cout << "Henlloo\n";
	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	GridGenerator::hyper_cube(tri);
	tri.refine_global(3);
	commet_solve::set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .2);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);
	// solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	{

		auto domain = std::make_unique<commet_solve::GloballyVectorizedDomain<dim, double>>();

		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
		domain->set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");
		domain->load_model("../resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");

		solver.add_material_domain(0, std::move(domain));
	}

	solver.initialize();
	solver.setup_system_with_constraints();
	solver.assemble_linear_system();
	solver.solve_linear_system();
	solver.output();
}

void fifth_test(const unsigned int &batch_size = 128)
{

	std::cout << "Henlloo\n";
	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	GridGenerator::hyper_cube(tri);
	tri.refine_global(3);
	commet_solve::set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .2);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);
	// solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	{

		// auto domain = std::make_unique<commet_solve::GloballyVectorizedDomain<dim, double>>();
		auto domain = std::make_unique<commet_solve::BatchVectorizedDomain<dim, double>>(batch_size);

		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
		domain->set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");
		domain->load_model("../resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");

		solver.add_material_domain(0, std::move(domain));
	}

	solver.initialize();
	solver.setup_system_with_constraints();
	solver.assemble_linear_system();
	solver.solve_linear_system();
	solver.output();
}

void sixth_test(const unsigned int &batch_size = 128)
{

	std::cout << "Henlloo\n";
	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	GridGenerator::hyper_cube(tri);
	tri.refine_global(3);
	commet_solve::set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .2);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);
	// solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	{

		// auto domain = std::make_unique<commet_solve::GloballyVectorizedDomain<dim, double>>();
		auto domain = std::make_unique<commet_solve::BatchVectorizedDomain<dim, double>>(batch_size);

		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
		domain->set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");
		domain->load_model("../resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");

		solver.add_material_domain(0, std::move(domain));
	}

	solver.initialize();

	// solver.setup_system_with_constraints();
	// solver.assemble_linear_system();
	// solver.solve_linear_system();
	solver.nr();

	solver.output();
}

void seventh_test(const unsigned int &batch_size = 128)
{

	std::cout << "Henlloo\n";
	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	GridGenerator::hyper_cube(tri);
	tri.refine_global(4);
	commet_solve::set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .2);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);
	// solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	{

		// auto domain = std::make_unique<commet_solve::GloballyVectorizedDomain<dim, double>>();
		auto domain = std::make_unique<commet_solve::BatchVectorizedDomain<dim, double>>(batch_size);

		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
		domain->set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");
		domain->load_model("../resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");

		solver.add_material_domain(0, std::move(domain));
	}

	solver.add_dbc(std::make_unique<commet_solve::FullyDefinedDBC<dim, double>>(
		1, std::vector<unsigned int>({0, 1, 2}), std::vector<double>({0, 0, 0}), 1));
	solver.add_dbc(std::make_unique<commet_solve::FullyDefinedDBC<dim, double>>(
		4, std::vector<unsigned int>({0}), std::vector<double>({1}), 1));
	solver.initialize();
	solver.solve();

	// solver.setup_system_with_constraints();
	// solver.assemble_linear_system();
	// solver.solve_linear_system();
	// solver.nr();

	// solver.output();
}

void eighth_test(const unsigned int &batch_size = 128)
{

	std::cout << "Henlloo\n";
	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	GridGenerator::hyper_cube(tri);
	tri.refine_global(4);
	commet_solve::set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .1);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);
	// solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	{

		// auto domain = std::make_unique<commet_solve::GloballyVectorizedDomain<dim, double>>();
		auto domain = std::make_unique<commet_solve::BatchVectorizedDomain<dim, double>>(batch_size);

		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
		domain->set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");
		domain->load_model("../resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");

		solver.add_material_domain(0, std::move(domain));
	}

	solver.add_dbc(std::make_unique<commet_solve::FullyDefinedDBC<dim, double>>(
		1, std::vector<unsigned int>({0, 1, 2}), std::vector<double>({0, 0, 0}), 1));
	// solver.add_dbc(std::make_unique<commet_solve::FullyDefinedDBC<dim, double>>(
	// 	4, std::vector<unsigned int>({0}), std::vector<double>({1}), 1));

	solver.add_dbc(std::make_unique<commet_solve::RotateBoundaryCondition<dim, double>>(
		4, dealii::Point<dim>({0.5, 0.5, 0.5}), dealii::Tensor<1, dim, double>({1, 0, 0}), 3.14, 1, 1));

	solver.initialize();
	solver.solve();

	// solver.setup_system_with_constraints();
	// solver.assemble_linear_system();
	// solver.solve_linear_system();
	// solver.nr();

	// solver.output();
}

void ninth_test(const unsigned int &batch_size = 128)
{

	std::cout << "Henlloo\n";
	const int dim = 3;
	// Triangulation<dim> tri;
	parallel::distributed::Triangulation<dim> tri(MPI_COMM_WORLD);
	// GridGenerator::hyper_cube(tri);
	GridGenerator::cylinder_shell(tri,
								  1,
								  0.95,
								  1.05,
								  /*n_radial*/ 0,
								  /*n_axial*/ 0,
								  true);

	tri.refine_global(2);
	// set_rectangular_boundary_ids<dim, parallel::distributed::Triangulation<dim>>(tri);
	commet_solve::Time<double> time(1, .05);
	commet_solve::FiniteStrainSolver<dim, double> solver(&tri, time, 1);
	// solver.add_material_domain(0, std::make_unique<commet_solve::NeoHookeanDomain<dim, double>>(1, 1));
	{

		// auto domain = std::make_unique<commet_solve::GloballyVectorizedDomain<dim, double>>();
		auto domain = std::make_unique<commet_solve::BatchVectorizedDomain<dim, double>>(batch_size);

		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/isihara-pmicnn-cpu.torchscript");
		domain->set_evaluation_method(commet_solve::NCMEvaluationMethod::OPT_F);
		// domain->load_model("/home/ben/Dropbox/clean-projects/commet_solve/resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");
		domain->load_model("../resources/torchscripts/opt/opt-neohookean-micnn-cpu.torchscript");

		solver.add_material_domain(0, std::move(domain));
	}

	solver.add_dbc(std::make_unique<commet_solve::FullyDefinedDBC<dim, double>>(
		2, std::vector<unsigned int>({0, 1, 2}), std::vector<double>({0, 0, 0}), 1));
	// solver.add_dbc(std::make_unique<commet_solve::FullyDefinedDBC<dim, double>>(
	// 	4, std::vector<unsigned int>({0}), std::vector<double>({1}), 1));

	solver.add_dbc(std::make_unique<commet_solve::RotateBoundaryCondition<dim, double>>(
		// 3, dealii::Point<dim>({0.5, 0.5, 0.5}), dealii::Tensor<1, dim, double>({0, 0, 1}), 3.14, 1, 1)
		3,
		dealii::Point<dim>({0, 0, 0}),
		dealii::Tensor<1, dim, double>({0, 0, 1}),
		3.14,
		0,
		1));

	solver.initialize();
	solver.solve();

	// solver.setup_system_with_constraints();
	// solver.assemble_linear_system();
	// solver.solve_linear_system();
	// solver.nr();

	// solver.output();
}


