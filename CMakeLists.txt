cmake_minimum_required (VERSION 3.0)
project (mesh)
include_directories(src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -Wall -Wextra -Wno-unused-variable -Wno-register -Werror")

find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost 1.62.0 COMPONENTS program_options REQUIRED)

file(GLOB sources "src/*.cpp")

add_executable(mesh ${sources})
target_include_directories(mesh PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(mesh PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(mesh ${PYTHON_LIBRARIES})
target_link_libraries(mesh ${Boost_LIBRARIES})

add_custom_target(run COMMAND mesh DEPENDS mesh WORKING_DIRECTORY ${CMAKE_PROJECT_DIR})
