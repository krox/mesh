cmake_minimum_required (VERSION 3.9.0)
project (mesh)
include_directories(src)


add_subdirectory(util)
include_directories(util/src)


include(FetchContent)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG        b9be5b9444772324459989177108a6a65b8b2769 # v2.2.0
)
FetchContent_MakeAvailable(cli11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wno-unused-variable -Wno-register -Werror -fno-math-errno -g")

find_package(HDF5 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB modules "src/util/*.cpp" "src/mesh/*.cpp" "src/scalar/*.cpp" "src/lattice/*.cpp")

set(binaries "group_histogram" "phi4" "sigma" "ising" "ising_correlator" "hmc" "test_lattice_gauge" "sweep_beta" "poisson" "benchmark" "landau")

foreach(bin ${binaries})
	add_executable(${bin} src/apps/${bin}.cpp ${modules})
	target_compile_features(${bin} PUBLIC cxx_std_20)
	set_target_properties(${bin} PROPERTIES CXX_EXTENSIONS ON)
	target_include_directories(${bin} PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(${bin} ${HDF5_LIBRARIES} util_hdf5 fmt::fmt stdc++fs fftw3 Eigen3::Eigen CLI11::CLI11)
endforeach()
