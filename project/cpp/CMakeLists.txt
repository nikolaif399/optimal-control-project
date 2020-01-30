cmake_minimum_required(VERSION 3.10)

# set the project name
project(OptimalControlProject)

find_package(ifopt)

include_directories(include)

# add the executable
add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC ifopt::ifopt_ipopt)
