set(SRC_FILES
    commet_solve.cpp
)

add_executable(commet_solve ${SRC_FILES})

deal_ii_setup_target(commet_solve)

target_link_libraries(commet_solve 
    nlohmann_json::nlohmann_json 
        fmt::fmt
        ${DEAL_II_LIBRARIES} 
        ${TORCH_LIBRARIES} 
)
