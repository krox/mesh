cmake_minimum_required (VERSION 3.0)
project (mesh)
include_directories(src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -Wall -Wextra -Wno-unused-variable -Wno-register -Werror")

find_package(Boost 1.62.0 COMPONENTS program_options REQUIRED)

file(GLOB modules "src/mesh/*.cpp" "src/util/*.cpp")

set(binaries "main" "thermalize")

foreach(bin ${binaries})
	add_executable(${bin} src/${bin}.cpp ${modules})
	target_include_directories(${bin} PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_libraries(${bin} ${Boost_LIBRARIES})
	target_link_libraries(${bin} fmt)
endforeach()
