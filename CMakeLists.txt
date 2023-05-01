cmake_minimum_required (VERSION 3.9.0)
project (mesh LANGUAGES CXX)


################################################################################
# dependencies
################################################################################

add_subdirectory(util EXCLUDE_FROM_ALL)

include(FetchContent)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG        b9be5b9444772324459989177108a6a65b8b2769 # v2.2.0
)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.0-preview4
)
FetchContent_MakeAvailable(cli11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED) # could be made optional I think


################################################################################
# mesh library
################################################################################

file(GLOB modules "src/util/*.cpp" "src/mesh/*.cpp" "src/scalar/*.cpp" "src/lattice/*.cpp")

add_library(mesh STATIC ${modules})
target_compile_features(mesh PUBLIC cxx_std_20)
set_target_properties(mesh PROPERTIES CXX_EXTENSIONS ON)
target_include_directories(mesh PUBLIC src)
target_link_libraries(mesh util_hdf5 stdc++fs fftw3 Eigen3::Eigen CLI11::CLI11 OpenMP::OpenMP_CXX)
target_compile_options(mesh PUBLIC -g -O3 -march=native -Wall -Wextra -Werror -fno-math-errno)

################################################################################
# executables
################################################################################

set(binaries "group_histogram" "ompbench" "phi4" "sigma" "ising" "ising_correlator" "hmc" "sweep_beta" "poisson" "benchmark" "landau")

foreach(bin ${binaries})
	add_executable(${bin} src/apps/${bin}.cpp)
	target_compile_features(${bin} PUBLIC cxx_std_20)
	set_target_properties(${bin} PROPERTIES CXX_EXTENSIONS ON)
	target_include_directories(${bin} PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(${bin} mesh)
endforeach()


################################################################################
# unittests
################################################################################

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	include(FetchContent)

	FetchContent_MakeAvailable(Catch2)

	#add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	#add_link_options(-fsanitize=address)

	add_executable(tests src/tests/lattice_gauge.cpp)
	target_compile_features(tests PRIVATE cxx_std_20)
	target_link_libraries(tests PRIVATE Catch2::Catch2WithMain mesh)
	target_compile_options(tests PRIVATE -g)

endif()
