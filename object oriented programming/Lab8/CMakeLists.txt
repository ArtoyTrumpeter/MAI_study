cmake_minimum_required(VERSION 2.8)
project(oop)
add_executable(main main.cpp)

find_package(Threads)

set_target_properties(
    main PROPERTIES
        COMPILE_OPTIONS "-g;-Wall;-Wextra;-Wpedantic;"
)
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})