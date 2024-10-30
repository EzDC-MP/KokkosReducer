# EFT Kokkos type
This projects has :
- CompensatedReducer.hpp that contains the Kokkos implementations of EFT
- various bench and tests (main.cxx and other .hpp files)

# Compiling
To use the EFT expansion, just include the CompensatedReducer.hpp file into
your kokkos project.

To compile this project whole, you'll need :
- A valid Kokkos installation to link
/static link into (see [The official Kokkos documentation](https://kokkos.org/kokkos-core-wiki/building.html)
- A CXX23 compliant compiler

Then, just CMake the project, something along the line
`cmake -b build/`
then `make` inside the build folder
