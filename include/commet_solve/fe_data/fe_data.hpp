#ifndef INCLUDE_FE_DATA_FE_DATA_HPP_
#define INCLUDE_FE_DATA_FE_DATA_HPP_


#include <deal.II/base/types.h>

namespace commet_solve{

using namespace dealii;

 template<int dim, typename Number = double>
struct CellData {
    CellData(const types::global_cell_index & id,
    const unsigned int & material_id,
    const unsigned int & n_nodes,
    const unsigned int & n_qps,
    const std::vector<types::global_cell_index> & global_dofs,
    const std::vector<Number> & jxw,
    const std::vector<std::vector<Number>> & N,
    const std::vector<std::vector<Tensor<1, dim, Number>>> & B)
    : id(id)
    , material_id(material_id)
    , n_nodes(n_nodes)
    , n_qps(n_qps)
    , global_dofs(global_dofs)
    , jxw(jxw)
    , N(N)
    , B(B)
    {};

    types::global_cell_index id;
    unsigned int material_id;
    unsigned int n_nodes;
    unsigned int n_qps;

    std::vector<types::global_dof_index> global_dofs;  

    std::vector<Number> jxw;  // Shape [n_qps]
    std::vector<std::vector<Number>> N;  // Shape [n_qps, n_nodes]
    std::vector<std::vector<Tensor<1, dim, Number>>> B;  // Shape [n_qps, n_nodes, dim]
};

 template<int dim, typename Number = double>
class FEData {
public:
    FEData() = default;
    FEData(FEData &&) = delete;
    FEData(const FEData &) = delete;
    FEData &operator=(FEData &&) = delete;
    FEData &operator=(const FEData &) = delete;
    ~FEData() = default;

    void add_cell(CellData<dim, Number> && data){
        cell_data.push_back(data);
    };

    const std::vector<CellData<dim, Number>> & get_cell_data(){
        return cell_data;
    }



private:
    std::vector<CellData<dim, Number>> cell_data;
    
};
            



}

#endif  // INCLUDE_FE_DATA_FE_DATA_HPP_
