cmake_minimum_required(VERSION 2.8)
project(oop)
add_executable(main main.cpp)

set_target_properties(
    main PROPERTIES
        COMPILE_OPTIONS "-g;-Wall;-Wextra;-Wpedantic;"
)