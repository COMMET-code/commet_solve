# Gather source files
set(SRC_FILES
    commet_solve.cpp
)

# # Create the library (or use add_executable if that's your goal)
# add_library(commet_solve ${SRC_FILES})

# Optional: If commet_solve.cpp is a main program, replace above with:
add_executable(commet_solve ${SRC_FILES})

# # Add include directories
# target_include_directories(commet_solve
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/include
#         ${DEAL_II_INCLUDE_DIRS}  # in case target doesn't propagate
# )


deal_ii_setup_target(commet_solve)

# Link deal.II
target_link_libraries(commet_solve 
    nlohmann_json::nlohmann_json 
        ${DEAL_II_LIBRARIES} 
        ${TORCH_LIBRARIES} 
)


