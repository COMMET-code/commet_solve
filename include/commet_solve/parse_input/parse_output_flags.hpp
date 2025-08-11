#ifndef INCLUDE_PARSE_INPUT_PARSE_OUTPUT_FLAGS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_OUTPUT_FLAGS_HPP_

#include "../output/output_flags.hpp"
#include "parse_utils.hpp"
#include <stdexcept>

namespace commet_solve::parse{

template <int dim, typename Number>
void parse_scalar_outputs_flags(const std::vector<std::string> & scalar_output_flags_inp, 
                                FiniteStrainSolver<dim, Number>& solver){
    for(const std::string & flag : scalar_output_flags_inp){
        try {
        solver.add_scalar_output(INVERTED_SCALAR_OUTPUT_NAMES.at(flag));
        // } catch (std::out_of_range) {
        } catch (const std::out_of_range & e) {
            throw std::logic_error("The flag '" + flag + "' requested for scalar output is not implemented. Available outputs are: " + string_of_keys<scalar_output_flag>(INVERTED_SCALAR_OUTPUT_NAMES));
        }
    }
    
}


template <int dim, typename Number>
void parse_vector_outputs_flags(const std::vector<std::string> & vector_output_flags_inp, 
                                FiniteStrainSolver<dim, Number>& solver){
    for(const std::string & flag : vector_output_flags_inp){
        try {
        solver.add_vector_output(INVERTED_VECTOR_OUTPUT_NAMES.at(flag));
        // } catch (std::out_of_range) {
        } catch (const std::out_of_range & e) {
            throw std::logic_error("The flag '" + flag + "' requested for vector output is not implemented. Available outputs are: " + string_of_keys<vector_output_flag>(INVERTED_VECTOR_OUTPUT_NAMES));
        }
    }
}

template <int dim, typename Number>
void parse_tensor_outputs_flags(const std::vector<std::string> & tensor_output_flags_inp, 
                                FiniteStrainSolver<dim, Number>& solver){

    for(const std::string & flag : tensor_output_flags_inp){
        try {
        solver.add_tensor_output(INVERTED_TENSOR_OUTPUT_NAMES.at(flag));
        // } catch (std::out_of_range) {
        } catch (const std::out_of_range & e) {
            throw std::logic_error("The flag '" + flag + "' requested for scalar output is not implemented. Available outputs are: " + string_of_keys<tensor_output_flag>(INVERTED_TENSOR_OUTPUT_NAMES));
        }
    }
}


template <int dim, typename Number>
void parse_outputs_flags(const json & output_flags_inp,
                   FiniteStrainSolver<dim, Number>& solver)
{

    parse_scalar_outputs_flags(value_or_default<vector<string>>("scalar_outputs", output_flags_inp, {}), solver);
    parse_vector_outputs_flags(value_or_default<vector<string>>("vector_outputs", output_flags_inp, {}), solver);
    parse_tensor_outputs_flags(value_or_default<vector<string>>("tensor_outputs", output_flags_inp, {}), solver);


}

}

#endif  // INCLUDE_PARSE_INPUT_PARSE_OUTPUT_FLAGS_HPP_
