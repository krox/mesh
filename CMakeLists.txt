cmake_minimum_required (VERSION 3.9.0)
project (mesh)
include_directories(src)

add_subdirectory(CLI11)
add_subdirectory(fmt)
add_subdirectory(util)

include_directories(CLI11/include)
include_directories(util/src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wno-unused-variable -Wno-register -Werror -pedantic")

find_package(HDF5 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB modules "src/util/*.cpp" "src/mesh/*.cpp" "src/scalar/*.cpp")

set(binaries "group_histogram" "phi4" "sigma" "ising" "sigma_mesh")

foreach(bin ${binaries})
	add_executable(${bin} src/${bin}.cpp ${modules})
	target_include_directories(${bin} PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(${bin} ${HDF5_LIBRARIES})
	target_link_libraries(${bin} util util_hdf5)
	target_link_libraries(${bin} fmt)
	target_link_libraries(${bin} stdc++fs)
	target_link_libraries(${bin} fftw3)
	target_link_libraries(${bin} Eigen3::Eigen)
endforeach()
