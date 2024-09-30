#
# Kahan Reducer Test CMake config
#
cmake_policy(SET CMP0074 NEW)


#set(CMAKE_C_COMPILER "nvcc")
#set(CMAKE_CXX_COMPILER "nvcc")
project(KahanKokkosReducer)

find_package(Kokkos REQUIRED)
find_package(CUDA REQUIRED)

#set(CMAKE_CXX_FLAGS "-L/usr/local/gcc-14.1.0/lib64/ -lstdc++ -lgcc_s")
add_executable(out src/main.cxx)
target_link_libraries(out Kokkos::kokkos)
