cmake_minimum_required(VERSION 3.14)
project(commet_solve VERSION 0.0.1 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Output directories
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# # Find deal.II (make sure to pass -DDEAL_II_DIR=/path if needed)
# find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR})


# set(JSON_BuildTests OFF CACHE INTERNAL "")
# add_subdirectory(external/json)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)

# set(DEAL_II_STATIC_EXECUTABLE OFF CACHE INTERNAL "")

find_package(deal.II 9.6.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

#
# Are all dependencies fulfilled?
#
if(NOT DEAL_II_WITH_LAPACK) # keep in one line
  message(FATAL_ERROR "
Error! This program requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_LAPACK = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_LAPACK = ${DEAL_II_WITH_LAPACK}
This conflicts with the requirements."
    )
endif()


# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/exprtk)

# Add src directory
add_subdirectory(src)
add_subdirectory(benchmark)

# ========================
# CPack Configuration
# ========================
# include(InstallRequiredSystemLibraries)
# set(CPACK_PACKAGE_NAME "commet_solve")
# set(CPACK_PACKAGE_VENDOR "YourName")
# set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "commet_solve: MPI-based C++ Solver")
# set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
# set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
# set(CPACK_PACKAGE_VERSION_MAJOR "1")
# set(CPACK_PACKAGE_VERSION_MINOR "0")
# set(CPACK_PACKAGE_VERSION_PATCH "0")
# set(CPACK_PACKAGE_CONTACT "you@example.com")
# set(CPACK_GENERATOR "TGZ;DEB")

# include(CPack)

# install(TARGETS commet_solve
#         RUNTIME DESTINATION bin)

# install(PROGRAMS ${CMAKE_SOURCE_DIR}/run.sh DESTINATION .)
# install(FILES ${CMAKE_SOURCE_DIR}/README.md DESTINATION .)

# # If you’re bundling shared libraries:
# install(FILES
# 	/home/ben/deal/deal.II-v9.6.0/lib/libdeal_II.so.9.6.0
# 	/home/ben/deal/libtorch/lib/libtorch.so
# 	/home/ben/deal/libtorch/lib/libc10.so
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libepetra.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libteuchoscore.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libteuchoscomm.so.14
# 	/home/ben/deal/petsc-3.18.6/lib/libpetsc.so.3.18
# 	/lib/x86_64-linux-gnu/libmpi.so.40
# 	/lib/x86_64-linux-gnu/libm.so.6
# 	/home/ben/deal/libtorch/lib/libtorch_cpu.so
# 	/lib/x86_64-linux-gnu/libmpi_cxx.so.40
# 	/lib/x86_64-linux-gnu/libstdc++.so.6
# 	/lib/x86_64-linux-gnu/libgcc_s.so.1
# 	/lib/x86_64-linux-gnu/libc.so.6
# 	/lib/x86_64-linux-gnu/libboost_iostreams.so.1.83.0
# 	/lib/x86_64-linux-gnu/libboost_serialization.so.1.83.0
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libkokkoscore.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libnox.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libml.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libifpack.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libamesos.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libaztecoo.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libepetraext.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libzoltan.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libteuchosparameterlist.so.14
# 	/lib/x86_64-linux-gnu/libumfpack.so.6
# 	/usr/local/lib/libgmsh.so.4.13
# 	/home/ben/deal/hdf5-1.12.2/lib/libhdf5.so.200
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKBool.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKBRep.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKernel.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKG3d.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKGeomAlgo.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKGeomBase.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKMath.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKMesh.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKShHealing.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKTopAlgo.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKXSBase.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKIGES.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKSTEP.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKSTL.so.11
# 	/home/ben/deal/slepc-3.18.3/lib/libslepc.so.3.18
# 	/home/ben/deal/parmetis-4.0.3/lib/libmetis.so
# 	/lib/x86_64-linux-gnu/libz.so.1
# 	/home/ben/deal/sundials-5.7.0/lib/libsundials_idas.so.4
# 	/home/ben/deal/sundials-5.7.0/lib/libsundials_arkode.so.4
# 	/home/ben/deal/sundials-5.7.0/lib/libsundials_kinsol.so.5
# 	/home/ben/deal/symengine-0.8.1/lib/libsymengine.so.0.8
# 	/lib/x86_64-linux-gnu/libgmp.so.10
# 	/lib/x86_64-linux-gnu/liblapack.so.3
# 	/lib/x86_64-linux-gnu/libblas.so.3
# 	/home/ben/deal/p4est-2.3.6/FAST/lib/libp4est.so.1
# 	/home/ben/deal/p4est-2.3.6/FAST/lib/libsc.so.1
# 	/lib64/ld-linux-x86-64.so.2 
# 	/lib/x86_64-linux-gnu/libpthread.so.0
# 	/home/ben/deal/petsc-3.18.6/lib/libHYPRE-2.25.0.so
# 	/home/ben/deal/petsc-3.18.6/lib/libscalapack.so.2.2
# 	/home/ben/deal/parmetis-4.0.3/lib/libparmetis.so
# 	/lib/x86_64-linux-gnu/libmpi_mpifh.so.40
# 	/lib/x86_64-linux-gnu/libgfortran.so.5
# 	/lib/x86_64-linux-gnu/libopen-rte.so.40
# 	/lib/x86_64-linux-gnu/libopen-pal.so.40
# 	/lib/x86_64-linux-gnu/libhwloc.so.15
# 	/lib/x86_64-linux-gnu/librt.so.1
# 	/lib/x86_64-linux-gnu/libdl.so.2
# 	/home/ben/deal/libtorch/lib/libgomp-98b21ff3.so.1
# 	/lib/x86_64-linux-gnu/libbz2.so.1.0
# 	/lib/x86_64-linux-gnu/liblzma.so.5
# 	/lib/x86_64-linux-gnu/libzstd.so.1
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libthyracore.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/librtop.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libteuchosnumerics.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libgaleri-epetra.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libtrilinosss.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libteuchosremainder.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libtriutils.so.14
# 	/home/ben/deal/trilinos-release-14-4-0/lib/libteuchosparser.so.14
# 	/lib/x86_64-linux-gnu/libcholmod.so.5
# 	/lib/x86_64-linux-gnu/libamd.so.3
# 	/lib/x86_64-linux-gnu/libsuitesparseconfig.so.7
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKBO.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKPrim.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKG2d.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKSTEPAttr.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKSTEP209.so.11
# 	/home/ben/deal/oce-OCE-0.18.3/lib/libTKSTEPBase.so.11
# 	/lib/x86_64-linux-gnu/libevent_core-2.1.so.7
# 	/lib/x86_64-linux-gnu/libevent_pthreads-2.1.so.7
# 	/lib/x86_64-linux-gnu/libudev.so.1
# 	/lib/x86_64-linux-gnu/libcolamd.so.3
# 	/lib/x86_64-linux-gnu/libcamd.so.3
# 	/lib/x86_64-linux-gnu/libccolamd.so.3
# 	/lib/x86_64-linux-gnu/libcap.so.2
#   # Add more as needed
#   DESTINATION lib
# )
