#
# Kahan Reducer Test CMake config
#
cmake_policy(SET CMP0074 NEW)

#set(CMAKE_C_COMPILER "gcc-14.1.0")
#set(CMAKE_CXX_COMPILER "g++-14.1.0")

set(CMAKE_CXX_STANDARD 23)

project(KahanKokkosReducer)

find_package(Kokkos REQUIRED)

#set(CMAKE_CXX_FLAGS "-L/usr/local/gcc-14.1.0/lib64/ -lstdc++ -lgcc_s")
add_executable(out src/main.cxx)
target_link_libraries(out Kokkos::kokkos)
