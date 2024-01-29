cmake_minimum_required(VERSION 3.9.0)
project(mesh LANGUAGES CXX CUDA)

add_compile_definitions(FMT_USE_NONTYPE_TEMPLATE_ARGS=0)

# add_compile_definitions(FMT_USE_CONSTEXPR=0)

# ###############################################################################
# dependencies
# ###############################################################################
include(cmake/CPM.cmake)

CPMAddPackage(
	NAME cli11
	GITHUB_REPOSITORY CLIUtils/CLI11
	GIT_TAG 291c58789c031208f08f4f261a858b5b7083e8e2 # v2.3.2
)

CPMAddPackage(
	NAME util
	GITHUB_REPOSITORY krox/util
	GIT_TAG 2da9780e930833297f8cd042b77c0035a4df6e37
)

CPMAddPackage(
	Catch2
	GITHUB_REPOSITORY catchorg/Catch2
	GIT_TAG 05e10dfccc28c7f973727c54f850237d07d5e10f # v3.5.2
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED) # could be made optional I think

# ###############################################################################
# mesh library
# ###############################################################################
file(GLOB modules "src/mesh/*.cpp" "src/scalar/*.cpp" "src/lattice/*.cpp" "src/gauge/*.cpp")

add_library(mesh STATIC ${modules})
target_compile_features(mesh PUBLIC cxx_std_20)
set_target_properties(mesh PROPERTIES CXX_EXTENSIONS ON)
target_include_directories(mesh PUBLIC src)
target_link_libraries(mesh util_hdf5 stdc++fs fftw3 Eigen3::Eigen CLI11::CLI11 OpenMP::OpenMP_CXX)
target_compile_options(mesh PUBLIC -g -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-Werror -fno-math-errno> $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -extended-lambda>)

set_source_files_properties(src/apps/hmc.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/gauge/hmc.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/lattice/fft.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/apps/benchmark_lattice.cpp PROPERTIES LANGUAGE CUDA)

# ###############################################################################
# executables
# ###############################################################################
set(binaries "benchmark_lattice" "hmc")

foreach(bin ${binaries})
	add_executable(${bin} src/apps/${bin}.cpp)
	target_compile_features(${bin} PUBLIC cxx_std_20)
	set_target_properties(${bin} PROPERTIES CXX_EXTENSIONS ON)
	target_include_directories(${bin} PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(${bin} mesh)
endforeach()

# ###############################################################################
# unittests
# ###############################################################################
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	include(FetchContent)

	FetchContent_MakeAvailable(Catch2)

	# add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	# add_link_options(-fsanitize=address)
	add_executable(tests src/tests/lattice_gauge.cpp)
	set_source_files_properties(src/tests/lattice_gauge.cpp PROPERTIES LANGUAGE CUDA)
	target_compile_features(tests PRIVATE cxx_std_20)
	target_link_libraries(tests PRIVATE Catch2::Catch2WithMain mesh)
	target_compile_options(tests PRIVATE -g)
endif()
