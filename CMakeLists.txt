#
# Kahan Reducer Test CMake config
#
cmake_policy(SET CMP0074 NEW)
project(KahanKokkosReducer)

find_package(Kokkos REQUIRED)

add_executable(out src/main.cxx)
target_link_libraries(out Kokkos::kokkos)
