// Enzo De Carvalho Bittencourt <ezdecarvalho@gmail.com>
// Error Free Transform reducer using Kahan Summation for Kokkos
#include <Kokkos_Core.hpp>
#include "KahanReducer.hpp"
#include <iostream>

int main(int argc, char* argv[]) 
{
  Kokkos::initialize(argc, argv);
  
  using Space = Kokkos::HostSpace; 

  printf("KahanReducer implementation\n");
  
  printf("Equality test (sum 0...99)\n-----------\n");
  { 
    float resultStandard;
    float resultKahan;
    int N = 100;

    Kokkos::parallel_reduce("Normal Reduction :", N,
       KOKKOS_LAMBDA(const int i, float &val)
       {val += i;},
       Kokkos::Sum<float>(resultStandard));

    Kokkos::parallel_reduce("Kahan Reduction :", N,
      KOKKOS_LAMBDA(const int i, float &val)
      {val += i; printf("&val : %p, i :%i\n", &val, i);},
      KahanReducer<float,float,Space>(resultKahan));
  
    printf("&resultKahan : %p\n", &resultKahan);    
    printf("Results :\n\tNormal\t: %f\n\tKahan\t: %f\n", resultStandard, resultKahan);
  }

  Kokkos::finalize();
}
