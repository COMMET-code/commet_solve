#ifndef INCLUDE_FE_DATA_FE_DATA_HPP_
#define INCLUDE_FE_DATA_FE_DATA_HPP_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/full_matrix.h>

namespace commet_solve
{

using namespace dealii;

template <int dim, typename Number = double>
struct CellData
{
	CellData(const types::global_cell_index &id,
			 const unsigned int &fe_index,
			 const unsigned int &material_id,
			 const unsigned int &n_nodes,
			 const unsigned int &n_qps,
			 const std::vector<types::global_cell_index> &global_dofs,
			 const std::vector<types::global_cell_index> &sca_dofs,
			 const std::vector<types::global_cell_index> &vec_dofs,
			 const std::vector<types::global_cell_index> &ten_dofs,
			 const std::vector<Number> &jxw,
			 const std::vector<std::vector<Number>> &project_N,
			 const std::vector<std::vector<Number>> &N,
			 const std::vector<std::vector<Tensor<1, dim, Number>>> &B
			 // FENumbering<dim> && numbering
			 )
		: id(id)
		, fe_index(fe_index)
		, material_id(material_id)
		, n_nodes(n_nodes)
		, n_qps(n_qps)
		, global_dofs(global_dofs)
		, jxw(jxw)
		, project_N(project_N)
		, N(N)
		, B(B)
		, sca_dofs(sca_dofs)
		, vec_dofs(vec_dofs)
		, ten_dofs(ten_dofs)
	{
		FullMatrix<Number> M(n_nodes, n_nodes);
		for (unsigned int qp = 0; qp < n_qps; qp++)
			for (unsigned int i = 0; i < n_nodes; i++)
				for (unsigned int j = 0; j < n_nodes; j++)
					M[i][j] += jxw.at(qp) * project_N.at(qp).at(i) * project_N.at(qp).at(j);

		projection_matrix.invert(M);

	};

	void project_scalar_values(const std::vector<Number> &values_at_qps,
                            Vector<Number> &projection,
                            Vector<Number> &rhs) const
	{
        Assert(projection.size()==this->n_nodes, ExcDimensionMismatch(projection.size(), this->n_nodes));
        Assert(rhs.size()==this->n_nodes, ExcDimensionMismatch(projection.size(), this->n_nodes));
		rhs = 0;
		for (unsigned int qp = 0; qp < n_qps; qp++)
			for (unsigned int i = 0; i < n_nodes; i++)
				rhs[i] += values_at_qps.at(qp) * project_N.at(qp).at(i) * jxw.at(qp);

		projection_matrix.vmult(projection, rhs);
	}
	void project_vector_values(const std::vector<Tensor<1, dim, Number>> &values_at_qps,
							   Vector<Number> &projection,
							   Vector<Number> &rhs,
							   std::vector<Vector<Number>>& projection_storage,
							   const FESystem<dim> &fe_sys) const
	{

        Assert(projection.size()==this->vec_dofs.size(),
               ExcDimensionMismatch(projection.size(), this->vec_dofs.size()));
        Assert(rhs.size()==this->n_nodes, ExcDimensionMismatch(rhs.size(), this->n_nodes));
        Assert(projection_storage.size()==dim, ExcDimensionMismatch(projection_storage.size(), dim));
        for(unsigned int i=0; i<dim; i++)
            Assert(projection_storage.at(i).size()==this->n_nodes,
                   ExcDimensionMismatch(projection_storage.at(i).size(), this->n_nodes));


		for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
		{

			rhs = 0;
			for (unsigned int qp = 0; qp < n_qps; qp++)
				for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
					rhs[i_node] += values_at_qps.at(qp)[i_comp] * project_N.at(qp).at(i_node) * jxw.at(qp);

			projection_matrix.vmult(projection_storage.at(i_comp), rhs);

		}

		for (unsigned int j = 0; j < projection.size(); j++)
		{
			const auto &comps = fe_sys.system_to_component_index(j);
			projection[j] = projection_storage.at(comps.first)[comps.second];
		}
	}
	void project_tensor_values(const std::vector<Tensor<2, dim, Number>> &values_at_qps,
							   Vector<Number> &projection,
							   Vector<Number> &rhs,
							   std::vector<Vector<Number>>& projection_storage,
							   const FESystem<dim> &fe_sys) const
	{

        Assert(projection.size()==this->ten_dofs.size(),
               ExcDimensionMismatch(projection.size(), this->ten_dofs.size()));
        Assert(rhs.size()==this->n_nodes, ExcDimensionMismatch(rhs.size(), this->n_nodes));
        Assert(projection_storage.size()==dim*dim, ExcDimensionMismatch(projection_storage.size(), dim*dim));
        for(unsigned int i=0; i<dim*dim; i++)
            Assert(projection_storage.at(i).size()==this->n_nodes,
                   ExcDimensionMismatch(projection_storage.at(i).size(), this->n_nodes));

		unsigned int count = 0;
		for (unsigned int i_comp = 0; i_comp < dim; i_comp++)
			for (unsigned int j_comp = 0; j_comp < dim; j_comp++)
			{

				rhs = 0;
				for (unsigned int qp = 0; qp < n_qps; qp++)
					for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
						rhs[i_node] += values_at_qps.at(qp)[i_comp][j_comp] * project_N.at(qp).at(i_node) * jxw.at(qp);

				projection_matrix.vmult(projection_storage.at(count), rhs);
				count++;
			}

		for (unsigned int j = 0; j < projection.size(); j++)
		{
			const auto &comps = fe_sys.system_to_component_index(j);
			projection[j] = projection_storage.at(comps.first)[comps.second];
		}
	}

	types::global_cell_index id;
	unsigned int fe_index;
	unsigned int material_id;
	unsigned int n_nodes;
	unsigned int n_qps;

	std::vector<types::global_dof_index> global_dofs;

	std::vector<Number> jxw;							// Shape [n_qps]
	std::vector<std::vector<Number>> project_N;			// Shape [n_qps, n_nodes]
	std::vector<std::vector<Number>> N;					// Shape [n_qps, n_nodes]
	std::vector<std::vector<Tensor<1, dim, Number>>> B; // Shape [n_qps, n_nodes, dim]

	FullMatrix<Number> projection_matrix;
	std::vector<types::global_dof_index> sca_dofs;
	std::vector<types::global_dof_index> vec_dofs;
	std::vector<types::global_dof_index> ten_dofs;
};




template <int dim, typename Number = double>
class FEData
{
  public:
	FEData() = default;
	FEData(FEData &&) = delete;
	FEData(const FEData &) = delete;
	FEData &operator=(FEData &&) = delete;
	FEData &operator=(const FEData &) = delete;
	~FEData() = default;

	void add_cell(CellData<dim, Number> &&data)
	{
		cell_data.push_back(data);
	};

	const std::vector<CellData<dim, Number>> &get_cell_data()
	{
		return cell_data;
	}

  private:
	std::vector<CellData<dim, Number>> cell_data;
};

} // namespace commet_solve

#endif // INCLUDE_FE_DATA_FE_DATA_HPP_
