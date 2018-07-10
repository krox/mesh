cmake_minimum_required (VERSION 3.9.0)
project (mesh)
include_directories(src)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if( supported )
	message(STATUS "IPO / LTO enabled")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported: <${error}>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wno-unused-variable -Wno-register -Werror")

find_package(Boost 1.62.0 COMPONENTS program_options REQUIRED)
find_package(HDF5 REQUIRED)

file(GLOB modules "src/mesh/*.cpp" "src/util/*.cpp")

set(binaries "main" "thermalize" "group_histogram" "generate" "wilson" "gaugefix")

foreach(bin ${binaries})
	add_executable(${bin} src/${bin}.cpp ${modules})
	target_include_directories(${bin} PRIVATE ${Boost_INCLUDE_DIRS})
	target_include_directories(${bin} PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(${bin} ${Boost_LIBRARIES})
	target_link_libraries(${bin} ${HDF5_LIBRARIES})
	target_link_libraries(${bin} fmt)
	target_link_libraries(${bin} stdc++fs)
endforeach()
