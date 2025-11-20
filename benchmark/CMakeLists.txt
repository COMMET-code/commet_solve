# set(SRC_FILES commet_benchmark.cpp)

find_package(benchmark REQUIRED)

set(NCM_BENCH ncm_vectorization_benchmark)
add_executable(${NCM_BENCH} ncm_vectorization_benchmark.cpp)
deal_ii_setup_target(${NCM_BENCH})
target_link_libraries(${NCM_BENCH}
    nlohmann_json::nlohmann_json 
    fmt::fmt 
    ${DEAL_II_LIBRARIES} 
    ${TORCH_LIBRARIES} 
    benchmark::benchmark
)
