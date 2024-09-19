// Enzo De Carvalho Bittencourt <ezdecarvalho@gmail.com>
// Error Free Transform reducer using Kahan Summation for Kokkos
#include <Kokkos_Core.hpp>
#include "KahanReducer.hpp"
#include <iostream>

#include "Arrays.hpp"
template <typename FloatType, typename ArrayType>
void PerformReduction(ArrayType arr, int N, FloatType& res) // dummy to let the compiler the type returned - horrible botch.
{
  Kokkos::parallel_reduce("Custom Reduction :", N,
     KOKKOS_LAMBDA(const int i, FloatType &val)
     {val += arr[i];}, //add and substract stuff..
     Kokkos::Sum<FloatType>(res));
}


int main(int argc, char* argv[]) 
{
  Kokkos::initialize(argc, argv);
  
  using Space = Kokkos::HostSpace; 

  printf("KahanReducer implementation\n");
  
  goto label; 

  printf("\nScalarhilo EFT test\n--------------\n");
  {
    using Sfloat = Scalarhilo<float>;
    Sfloat a(1, 1e-35);
    Sfloat b(1e-34);

    printf("b.hi : %.e  b.lo : %.e\n", b.hi, b.lo);
    printf("a.hi : %.e  a.lo : %.e\n", a.hi, a.lo);
    a += b;
    printf("After summation : a.hi : %.e  a.lo : %.10e\n", a.hi, a.lo);
  } 
  /*
  printf("\nEquality test (sum 0...99)\n-----------\n");
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
  */
  printf("\nEquality test (sum 0...99)\n-----------\n");
  {
    using Sfloat = Scalarhilo<float>;
    float resultStandard;
    Sfloat resultKahan;
    int N = 100;

    Kokkos::parallel_reduce("Normal Reduction :", N,
       KOKKOS_LAMBDA(const int i, float &val)
       {val += i;},
       Kokkos::Sum<float>(resultStandard));

    Kokkos::parallel_reduce("Kahan Reduction :", N,
      KOKKOS_LAMBDA(const int i, Sfloat &val)
      {val += i; },//printf("&val : %p, i :%i\n", &val, i);},
        Kokkos::Sum<Sfloat>(resultKahan));
  
    printf("Results :\n\tNormal\t: %.10e\n\tKahan\t: %.10e + %.10e = %.10e\n"
    , resultStandard, resultKahan.hi, resultKahan.lo, resultKahan.finalize());

  }

  printf("\nbad summation test\n-----------\n");
  {
    using Sfloat = Scalarhilo<float>;
    using Sbfloat = Scalarhilo<__bf16>;
    using SbfloatFloat = Scalarhilo<__bf16,float>;
    using Sdouble = Scalarhilo<double>;
    using SdoubleBfloat = Scalarhilo<double,__bf16>;

    float resultStandard;
    Sfloat resultKahan;
    Sbfloat resultKahanbfloat;
    SbfloatFloat resultKahanbfloatf;
    double resultStandardDouble;
    Sdouble resultKahanDouble;
    SdoubleBfloat resultKahanDoubleBf;

    const int N = 255;
    const float list[N] = 
{
5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 
6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 
4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 3.0517578125000000e-05, 
6.5536000000000000e+04, 7.6293945312500000e-06, 2.6214400000000000e+05, 1.9073486328125000e-06, 1.0485760000000000e+06, 
4.7683715820312500e-07, 4.1943040000000000e+06, 1.1920928955078125e-07, 1.0000000000000000e+00, 5.0000000000000000e-01, 
4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 6.4000000000000000e+01, 
7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 4.8828125000000000e-04, 
4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 3.0517578125000000e-05, 6.5536000000000000e+04, 
7.6293945312500000e-06, 2.6214400000000000e+05, 1.9073486328125000e-06, 1.0485760000000000e+06, 4.7683715820312500e-07, 
4.1943040000000000e+06, 1.1920928955078125e-07, 1.0000000000000000e+00, 5.0000000000000000e-01, 4.0000000000000000e+00, 
1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 6.4000000000000000e+01, 7.8125000000000000e-03, 
2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 4.8828125000000000e-04, 4.0960000000000000e+03, 
1.2207031250000000e-04, 1.6384000000000000e+04, 3.0517578125000000e-05, 6.5536000000000000e+04, 7.6293945312500000e-06, 
2.6214400000000000e+05, 1.9073486328125000e-06, 1.0485760000000000e+06, 4.7683715820312500e-07, 4.1943040000000000e+06, 
1.1920928955078125e-07, 1.0000000000000000e+00, 5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 
1.6000000000000000e+01, 3.1250000000000000e-02, 6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 
1.9531250000000000e-03, 1.0240000000000000e+03, 4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 
1.6384000000000000e+04, 3.0517578125000000e-05, 6.5536000000000000e+04, 7.6293945312500000e-06, 2.6214400000000000e+05, 
1.9073486328125000e-06, 1.0485760000000000e+06, 4.7683715820312500e-07, 4.1943040000000000e+06, 1.1920928955078125e-07, 
1.0000000000000000e+00, 5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 
3.1250000000000000e-02, 6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 
1.0240000000000000e+03, 4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 
3.0517578125000000e-05, 6.5536000000000000e+04, 7.6293945312500000e-06, 2.6214400000000000e+05, 1.9073486328125000e-06, 
1.0485760000000000e+06, 4.7683715820312500e-07, 4.1943040000000000e+06, 1.1920928955078125e-07, 1.0000000000000000e+00, 
5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 
6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 
4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 3.0517578125000000e-05, 
6.5536000000000000e+04, 7.6293945312500000e-06, 2.6214400000000000e+05, 1.9073486328125000e-06, 1.0485760000000000e+06, 
4.7683715820312500e-07, 4.1943040000000000e+06, 1.1920928955078125e-07, 1.0000000000000000e+00, 5.0000000000000000e-01, 
4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 6.4000000000000000e+01, 
7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 4.8828125000000000e-04, 
4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 3.0517578125000000e-05, 6.5536000000000000e+04, 
7.6293945312500000e-06, 2.6214400000000000e+05, 1.9073486328125000e-06, 1.0485760000000000e+06, 4.7683715820312500e-07, 
4.1943040000000000e+06, 1.1920928955078125e-07, 1.0000000000000000e+00, 5.0000000000000000e-01, 4.0000000000000000e+00, 
1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 6.4000000000000000e+01, 7.8125000000000000e-03, 
2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 4.8828125000000000e-04, 4.0960000000000000e+03, 
1.2207031250000000e-04, 1.6384000000000000e+04, 3.0517578125000000e-05, 6.5536000000000000e+04, 7.6293945312500000e-06, 
2.6214400000000000e+05, 1.9073486328125000e-06, 1.0485760000000000e+06, 4.7683715820312500e-07, 4.1943040000000000e+06, 
1.1920928955078125e-07, 1.0000000000000000e+00, 5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 
1.6000000000000000e+01, 3.1250000000000000e-02, 6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 
1.9531250000000000e-03, 1.0240000000000000e+03, 4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 
1.6384000000000000e+04, 3.0517578125000000e-05, 6.5536000000000000e+04, 7.6293945312500000e-06, 2.6214400000000000e+05, 
1.9073486328125000e-06, 1.0485760000000000e+06, 4.7683715820312500e-07, 4.1943040000000000e+06, 1.1920928955078125e-07, 
1.0000000000000000e+00, 5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 
3.1250000000000000e-02, 6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 
1.0240000000000000e+03, 4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 
3.0517578125000000e-05, 6.5536000000000000e+04, 7.6293945312500000e-06, 2.6214400000000000e+05, 1.9073486328125000e-06, 
1.0485760000000000e+06, 4.7683715820312500e-07, 4.1943040000000000e+06, 1.1920928955078125e-07, 1.0000000000000000e+00, 
5.0000000000000000e-01, 4.0000000000000000e+00, 1.2500000000000000e-01, 1.6000000000000000e+01, 3.1250000000000000e-02, 
6.4000000000000000e+01, 7.8125000000000000e-03, 2.5600000000000000e+02, 1.9531250000000000e-03, 1.0240000000000000e+03, 
4.8828125000000000e-04, 4.0960000000000000e+03, 1.2207031250000000e-04, 1.6384000000000000e+04, 1.0000000000000000e+00
};

    /* This list was generated this way 
    for (int i=0; i < N; i++)
    {
      list[i] = (i % 2 == 0) ? pow(2, (i%24)) : pow(2, -(i%24));
      printf("%.16e, ", list[i]);
      if (!(i % 5)) {printf("\n");} 
    }*/
    
    printf("\n");
   
    Kokkos::parallel_reduce("Normal Reduction :", N,
       KOKKOS_LAMBDA(const int i, float &val)
       {val += list[i]*((i % 4) - 2);}, //add and substract stuff..
       Kokkos::Sum<float>(resultStandard));

    Kokkos::parallel_reduce("EFT Reduction :", N,
      KOKKOS_LAMBDA(const int i, Sfloat &val)
      {val += list[i]*((i % 4) - 2);},//printf("&val : %p, i :%i\n", &val, i);},
      Kokkos::Sum<Sfloat>(resultKahan));
/*
    Kokkos::parallel_reduce("Kahan Reduction bf16:", N,
      KOKKOS_LAMBDA(const int i, Sbfloat &val)
      {val += list[i]*((i % 4) - 2);},//printf("&val : %p, i :%i\n", &val, i);},
      Kokkos::Sum<Sbfloat>(resultKahanbfloat));

    Kokkos::parallel_reduce("Kahan Reduction bf16_f:", N,
      KOKKOS_LAMBDA(const int i, SbfloatFloat &val)
      {val += list[i]*((i % 4) - 2);},//printf("&val : %p, i :%i\n", &val, i);},
      Kokkos::Sum<SbfloatFloat>(resultKahanbfloatf));
*/    
    printf("Results :\n\tNormal\t: %.16e\n\tEft\t: %.16e + %.16e \n\t\t= %.16e\n"
    , resultStandard, resultKahan.hi, resultKahan.lo, resultKahan.finalize());  
/*    
    printf("Kahan_bf16\t: %.16e + %.16e \n\t\t= %.16e\n"
    , resultKahanbfloat.hi, resultKahanbfloat.lo, resultKahanbfloat.finalize());
    
    printf("Kahan_bf16+f\t: %.16e + %.16e \n\t\t= %.16e\n"
    , resultKahanbfloatf.hi, resultKahanbfloatf.lo, resultKahanbfloatf.finalize());
*//*
    //------------

    Kokkos::parallel_reduce("Normal Reduction (double) :", N,
       KOKKOS_LAMBDA(const int i, double &val)
       {val += list[i]*((i % 4) - 2);}, //add and substract stuff..
       Kokkos::Sum<double>(resultStandardDouble));

    Kokkos::parallel_reduce("Kahan Reduction (double):", N,
      KOKKOS_LAMBDA(const int i, Sdouble &val)
      {val += list[i]*((i % 4) - 2);},//printf("&val : %p, i :%i\n", &val, i);},
      Kokkos::Sum<Sdouble>(resultKahanDouble));

    Kokkos::parallel_reduce("Kahan Reduction (double + bf):", N,
      KOKKOS_LAMBDA(const int i, SdoubleBfloat &val)
      {val += list[i]*((i % 4) - 2);},//printf("&val : %p, i :%i\n", &val, i);},
      Kokkos::Sum<SdoubleBfloat>(resultKahanDoubleBf));
  

    printf("Results :\n\tNormal\t: %.16e\n\tKahan\t: %.16e + %.16e \n\t\t= %.16e\n"
    , resultStandard, resultKahan.hi, resultKahan.lo, resultKahan.finalize());

    printf("Results :\n\tNormal (double)\t: %.30e \
        \n\tKahan (double)\t: %.30e + %.30e \n\t\t\t= %.30e \
        \n\tKahan (d+bf16)\t: %.30e + %.30e \n\t\t\t= %.30e \
        \n"
    , resultStandardDouble
    , resultKahanDouble.hi, resultKahanDouble.lo, resultKahanDouble.finalize()
    , resultKahanDoubleBf.hi, resultKahanDoubleBf.lo, resultKahanDoubleBf.finalize());
    */
  }
  
label: 
  using fp32_fp32_t = Scalarhilo<float>;
  using bf16_bf16_t = Scalarhilo<__bf16>;
  using fp32_bf16_t = Scalarhilo<float, __bf16>;
  using bf16_fp32_t = Scalarhilo<__bf16, float>;
  using bf16_fp16_t = Scalarhilo<__bf16, _Float16>;

  printf("\nArray summations\n-----------\n");
  {
    double res_fp64 = 0.0;
    _Float16 res_fp16 = 0.0;
    float res_fp32 = 0.0;
    __bf16 res_bf16 = 0.0;
    fp32_fp32_t res_fp32_fp32 = 0.0;
    bf16_bf16_t res_bf16_bf16 = 0.0;
    fp32_bf16_t res_fp32_bf16 = 0.0;
    bf16_fp32_t res_bf16_fp32 = 0.0;
    bf16_fp16_t res_bf16_fp16 = 0.0;

#define AllReduce(arr, N) \
  PerformReduction(arr, N, res_fp64); \
  PerformReduction(arr, N, res_fp32); \
  PerformReduction(arr, N, res_fp16); \
  PerformReduction(arr, N, res_bf16); \
  PerformReduction(arr, N, res_fp32_fp32); \
  PerformReduction(arr, N, res_bf16_bf16); \
  PerformReduction(arr, N, res_fp32_bf16); \
  PerformReduction(arr, N, res_bf16_fp32); \
  PerformReduction(arr, N, res_bf16_fp16); 

#define AllPrint() \
  printf("Results (ScalarType, AccumulatorType))\n"); \
  printf("fp64 (no EFT)\t: %.64e\n", res_fp64); \
  printf("fp32 (no EFT)\t: %.32e\n", res_fp32); \
  printf("fp16 (no EFT)\t: %.32e\n", res_fp16); \
  printf("bf16 (no EFT)\t: %.32e\n", (double)res_bf16); \
  printf("fp32, fp32\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , res_fp32_fp32.hi, res_fp32_fp32.lo, res_fp32_fp32.finalize()); \
  printf("bf16, bf16\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , (double)res_bf16_bf16.hi, (double)res_bf16_bf16.lo, (double)res_bf16_bf16.finalize()); \
  printf("fp32, bf16\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , res_fp32_bf16.hi, (double)res_fp32_bf16.lo, res_fp32_bf16.finalize()); \
  printf("bf16, fp32\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , (double)res_bf16_fp32.hi, res_bf16_fp32.lo, (double)res_bf16_fp32.finalize()); \
  printf("bf16, fp16\t: %.32e + %.32e\n\t\n"\
      , (double)res_bf16_fp16.hi, (double) res_bf16_fp16.lo, (double)res_bf16_fp16.finalize());

#define ShowDiffFp64()\
   printf("Difference with fp64 result :\n"); \
   printf("fp64 (=0)\t: %.64e\n", res_fp64 - res_fp64); \
   printf("fp32 (no EFT)\t: %.64e\n", res_fp64 - res_fp32); \
   printf("fp16 (no EFT)\t: %.64e\n", res_fp64 - res_fp16); \
   printf("bf16 (no EFT)\t: %.64e\n", res_fp64 - res_bf16); \
   printf("fp32, fp32\t: %.64e\n"\
       , (res_fp64 - res_fp32_fp32.hi) - res_fp32_fp32.lo); \
   printf("bf16, bf16\t: %.64e\n"\
       , (res_fp64 - res_bf16_bf16.hi) - res_bf16_bf16.lo); \
   printf("fp32, bf16\t: %.64e\n"\
       , (res_fp64 - res_fp32_bf16.hi) - res_fp32_bf16.lo); \
   printf("bf16, fp32\t: %.64e\n"\
       , (res_fp64 - res_bf16_fp32.hi) - res_bf16_fp32.lo); \
   printf("bf16, fp16\t: %.64e\n"\
       , (res_fp64 - res_bf16_fp16.hi) - res_bf16_fp16.lo);

  printf("\n\n###Init Type Test (one)###\n");
  float one;
  one=1;
  int  N = 1;
  
  AllReduce(&one,1);
  AllPrint();
  

  printf("\n\n###Sum Test (1+1)###########################################\n");
  float one_one[2] = {1.0,1.0};
  AllReduce(one_one,2);
  AllPrint();

  printf("\n\n###Test Reduction [1,2,3,4,5,6,7,8,9,10] (=55)##############\n");
  float range1_10[10] = {1,2,3,4,5,6,7,8,9,10};
  AllReduce(range1_10, 10);
  AllPrint();  

  printf("\n\n### Reduction on 512 positive floats #######################\n");
  AllReduce(float_512p112_dyebi, 512);
  AllPrint();

  printf("\n\n### Reduction on 512 positive float16 ######################\n");
  AllReduce(_Float16_512p22_wslgv, 512);
  AllPrint();
  ShowDiffFp64();

  printf("\n\n### Reduction on 2048 positive float16 #####################\n");
  AllReduce(_Float16_2048p19_uzwjo, 2048);
  AllPrint();
  ShowDiffFp64();
  
  printf("\n\n### Reduction on 512 float16 ###############################\n");
  AllReduce(_Float16_512p19_zovey, 512);
  AllPrint();
  ShowDiffFp64();

  printf("\n\n### Reduction on 2048 float16 ##############################\n");
  AllReduce(_Float16_2048p19_tzlvw, 2048);
  AllPrint();
  ShowDiffFp64();

  printf("\n\n### Reduction on 16384 float16 ##############################\n");
  AllReduce(_Float16_16384p19_wxcse, 16384);
  AllPrint();
  ShowDiffFp64();
  }

  Kokkos::finalize();
}
