#ifndef INCLUDE_PARSE_INPUT_PARSE_TIME_FUNCTIONS_HPP_
#define INCLUDE_PARSE_INPUT_PARSE_TIME_FUNCTIONS_HPP_

#include "parse_utils.hpp"

#include "commet_solve/time_functions/time_functions.hpp"

#include <memory>


namespace commet_solve::parse {


enum class time_function_types
{
	linear_ramp,
	data
};


static const map<string, time_function_types> //
	TIME_FUNCTION_TYPES({
    {"linear_ramp", time_function_types::linear_ramp},
    {"data", time_function_types::data}
	});


template <typename ValueType = double, typename TimeType = double>
unique_ptr<TimeFunction<ValueType, TimeType>> parse_time_function(const json &time_func_spec){


	switch (json_key_to_map_value("type", time_func_spec, TIME_FUNCTION_TYPES)){
        case time_function_types::linear_ramp:{
            return make_unique<LinearRamp<ValueType, TimeType>>(
                value_or_default<ValueType>("end_value", time_func_spec, 1),
                value_or_default<TimeType>("end_time", time_func_spec, 1),
                value_or_default<ValueType>("start_value", time_func_spec, 0),
                value_or_default<TimeType>("start_time", time_func_spec, 0));
        }
        case time_function_types::data:{
            return make_unique<DataFunction<ValueType, TimeType>>(
                compulsory_value<vector<pair<TimeType, ValueType>>>("data", time_func_spec)
                );
        }
        default:
            throw std::logic_error("Should not have been able to get to this point in the control flow...");
    }
    throw std::logic_error("Should not have been able to get to this point in the control flow...");
}

}

#endif  // INCLUDE_PARSE_INPUT_PARSE_TIME_FUNCTIONS_HPP_
