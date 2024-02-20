cmake_minimum_required(VERSION 3.26)
project(Grupal01_pp CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(fmt)

add_executable(Grupal01_pp
        histograma.cu
)

target_include_directories(Grupal01_pp
        PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
        Grupal01_pp fmt::fmt
        ${CUDART_LIBRARY}
)

set_target_properties(Grupal01_pp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
