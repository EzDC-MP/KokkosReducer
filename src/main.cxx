// Enzo De Carvalho Bittencourt <ezdecarvalho@gmail.com>
// Error Free Transform reducer using Kahan Summation for Kokkos
#include <Kokkos_Core.hpp>
#include "CompensatedReducer.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

#include "Arrays.hpp"
#include "WorstCaseArrays.hpp"
#include "LargeWorstCaseArrays.hpp" //This loads 1.6MB !
																		
template <typename FloatType, typename ArrayType>
void PerformReduction(ArrayType arr, int N, FloatType& res) 
{
  Kokkos::parallel_reduce("Custom Reduction :", N,
     KOKKOS_LAMBDA(const int i, FloatType &val)
     {val += arr[i];},
     Kokkos::Sum<FloatType>(res));
}

template <typename FloatType, typename ArrayType>
inline void PerformRedSingleThread(ArrayType arr, int N, FloatType& res)
{
  for (long unsigned i = 0; i < N ; i++)
  {res += arr[i];} //add stuff..
}

//returns the number of significand depending on num and err 
inline const int significand(double num, double err)
{
	if ((num == 0) && (err == 0)){return std::numeric_limits<int>::max();}//+inf
	if (num != 0)
	{	
		double relat = fabs(err/num);
		if (relat > 1){return 0;}
		return (int)std::floor(- std::log2(relat));	
	}
	return 0;	
} 

#define quote(x) #x
#define title(text) printf( quote(\n\e[35;4m text \e[0m\n) )
#define subtitle(text) printf( quote(\n\n\e[32;1m text \e[0m\n) )

int main(int argc, char* argv[]) 
{
  Kokkos::initialize();
  //typedef
  using fp32_fp32_t = Scalarhilo<float>;
  using bf16_bf16_t = Scalarhilo<__bf16>;
  using fp32_bf16_t = Scalarhilo<float, __bf16>;
  using fp32_fp16_t = Scalarhilo<float, _Float16>;
	using fp16_fp16_t = Scalarhilo<_Float16, _Float16>;
  using bf16_bf16_t = Scalarhilo<__bf16, __bf16>;
  using bf16_fp16_t = Scalarhilo<__bf16, _Float16>;
	using fp16_bf16_t = Scalarhilo<_Float16, __bf16>;

  title(Array summations);
  {
    double res_fp64 = 0.0;
    _Float16 res_fp16 = 0.0;
    float res_fp32 = 0.0;
    __bf16 res_bf16 = 0.0;
    fp32_fp32_t res_fp32_fp32 = 0.0;
    fp32_bf16_t res_fp32_bf16 = 0.0;
    fp32_fp16_t res_fp32_fp16 = 0.0;
    fp16_fp16_t res_fp16_fp16 = 0.0;
    bf16_bf16_t res_bf16_bf16 = 0.0;
    bf16_fp16_t res_bf16_fp16 = 0.0;
		fp16_bf16_t res_fp16_bf16 = 0.0;

#define AllReduce(arr, N) \
  PerformReduction(arr, N, res_fp64); \
  PerformReduction(arr, N, res_fp32); \
  PerformReduction(arr, N, res_fp16); \
  PerformReduction(arr, N, res_bf16); \
  PerformReduction(arr, N, res_fp32_fp32); \
  PerformReduction(arr, N, res_fp32_bf16); \
  PerformReduction(arr, N, res_fp32_fp16); \
  PerformReduction(arr, N, res_fp16_fp16); \
  PerformReduction(arr, N, res_bf16_bf16); \
  PerformReduction(arr, N, res_bf16_fp16); \
	PerformReduction(arr, N, res_fp16_bf16);

#define AllReduceSingleThread(arr, N) \
  PerformRedSingleThread(arr, N, res_fp64); \
  PerformRedSingleThread(arr, N, res_fp32); \
  PerformRedSingleThread(arr, N, res_fp16); \
  PerformRedSingleThread(arr, N, res_bf16); \
  PerformRedSingleThread(arr, N, res_fp32_fp32); \
  PerformRedSingleThread(arr, N, res_fp32_bf16); \
  PerformRedSingleThread(arr, N, res_fp32_fp16); \
  PerformRedSingleThread(arr, N, res_fp16_fp16); \
  PerformRedSingleThread(arr, N, res_bf16_bf16); \
  PerformRedSingleThread(arr, N, res_bf16_fp16); \
	PerformRedSingleThread(arr, N, res_fp16_bf16); \

#define AllPrint() \
  printf("Results (ScalarType, AccumulatorType))\n"); \
  printf("fp64 (no EFT)\t: %.64e\n", res_fp64); \
  printf("fp32 (no EFT)\t: %.32e\n", res_fp32); \
  printf("fp16 (no EFT)\t: %.32e\n", res_fp16); \
  printf("bf16 (no EFT)\t: %.32e\n", (double)res_bf16); \
  printf("fp32, fp32\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , res_fp32_fp32.hi, res_fp32_fp32.lo, res_fp32_fp32.finalize()); \
  printf("fp32, bf16\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , res_fp32_bf16.hi, (double)res_fp32_bf16.lo, res_fp32_bf16.finalize()); \
  printf("fp32, fp16\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , res_fp32_fp16.hi, (double)res_fp32_fp16.lo, res_fp32_fp16.finalize()); \
	printf("fp16, fp16\t: %.32e + %.32e\n\t\t= %.32e\n" \
      , (double)res_fp16_fp16.hi, (double)res_fp16_fp16.lo, (double)res_fp16_fp16.finalize()); \
  printf("bf16, bf16\t: %.32e + %.32e\n\t\t= %.32e\n" \
      , (double)res_bf16_bf16.hi, (double)res_bf16_bf16.lo, (double)res_bf16_bf16.finalize()); \
  printf("bf16, fp16\t: %.32e + %.32e\n\t\t= %.32e\n" \
      , (double)res_bf16_fp16.hi, (double) res_bf16_fp16.lo, (double)res_bf16_fp16.finalize()); \
  printf("fp16, bf16\t: %.32e + %.32e\n\t\t= %.32e\n"\
      , (double)res_fp16_bf16.hi, (double) res_fp16_bf16.lo, (double)res_fp16_bf16.finalize()); \

#define ShowDiffFp64()\
   printf("Difference with fp64 result :\n"); \
   printf("fp64 (=0)\t: %.64e\n", res_fp64 - res_fp64); \
   printf("fp32 (no EFT)\t: %.64e\n", res_fp64 - res_fp32); \
   printf("fp16 (no EFT)\t: %.64e\n", res_fp64 - res_fp16); \
   printf("bf16 (no EFT)\t: %.64e\n", res_fp64 - res_bf16); \
   printf("fp32, fp32\t: %.64e\n"\
       , (res_fp64 - res_fp32_fp32));\
   printf("fp32, bf16\t: %.64e\n"\
       , (res_fp64 - res_fp32_bf16)); \
   printf("fp32, fp16\t: %.64e\n"\
       , (res_fp64 - res_fp32_fp16)); \
	 printf("fp16, fp16\t: %.64e\n"\
			 , (res_fp64 - res_fp16_fp16)); \
	 printf("bf16, bf16\t: %.64e\n"\
       , (res_fp64 - res_bf16_bf16)); \
   printf("bf16, fp16\t: %.64e\n"\
       , (res_fp64 - res_bf16_fp16)); \
	printf("fp16, bf16\t: %.64e\n"\
       , (res_fp64 - res_fp16_bf16));
goto skip;
  subtitle(Init Type Test (one));
  {
    float one;
    one=1;
    int  N = 1;
  
    AllReduce(&one,1);
    AllPrint();
  }

  subtitle(Sum Test (1+1));
  { 
    float one_one[2] = {1.0,1.0};
    AllReduce(one_one,2);
    AllPrint();
  }

  subtitle(Test Reduction [1 2 3 4 5 6 7 8 9 10] (=55));
  { 
    float range1_10[10] = {1,2,3,4,5,6,7,8,9,10};
    AllReduce(range1_10, 10);
    AllPrint();  
  }

  subtitle(Reduction on 512 positive floats );
  AllReduce(float_512p112_dyebi, 512);
  AllPrint();

  subtitle(Reduction on 512 positive float16 );
  AllReduce(_Float16_512p22_wslgv, 512);
  AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 2048 positive float16 );
  AllReduce(_Float16_2048p19_uzwjo, 2048);
  AllPrint();
  ShowDiffFp64();
  
  subtitle(Reduction on 512 float16 );
  AllReduce(_Float16_512p19_zovey, 512);
  AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 2048 float16 );
  AllReduce(_Float16_2048p19_tzlvw, 2048);
  AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 16384 float16 );
  AllReduce(_Float16_16384p19_wxcse, 16384);
  AllPrint();
  ShowDiffFp64();
  
  /*
  subtitle(Reduction on 512 positive ordered float16 (asc) );
  AllReduce(_Float16_512p19_iaqtn_sorted, 512);
  AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 4096 positive ordered float16 (asc) );
  AllReduce(_Float16_4096p19_bauck_sorted, 4096);
  AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 512 positive ordered float16 (dsc) );
  AllReduce(_Float16_512p19_kjtxi_sorted_reverse, 512);
  AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 4096 positive ordered float16 (dsc) );
  AllReduce(_Float16_4096p19_cetsx_sorted_reverse, 4096);
  AllPrint();
  ShowDiffFp64();
  */

  subtitle(Reduction on 4096 positive float16 );
  AllReduce(_Float16_4096p19_jdygv, 4096);
  //AllPrint();
  ShowDiffFp64();
 
  subtitle(Reduction on 4096 positive float16 (single thread) );
  AllReduceSingleThread(_Float16_4096p19_jdygv, 4096);
  //AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 4096 positive ordered float16 (asc) );
  AllReduce(_Float16_4096p19_jdygv_sorted, 4096);
  //AllPrint();
  ShowDiffFp64();
  
  subtitle(Reduction on 4096 positive ordered float16 (asc single thread));
  AllReduceSingleThread(_Float16_4096p19_jdygv_sorted, 4096);
  //AllPrint();
  ShowDiffFp64();
  
  subtitle(Reduction on 4096 positive ordered float16 (dsc) );
  AllReduce(_Float16_4096p19_jdygv_sorted_rev, 4096);
  //AllPrint();
  ShowDiffFp64();
    
  subtitle(Reduction on 4096 positive ordered float16 (dsc single thread));
  AllReduceSingleThread(_Float16_4096p19_jdygv_sorted_rev, 4096);
  //AllPrint();
  ShowDiffFp64();

  subtitle(Reduction on 4096 positive double (bad) );
  AllReduce(worstCase_4096, 4096);
  AllPrint();
  ShowDiffFp64();

skip:
  subtitle(Reduction on 4194304 positive double (bad) );
  AllReduce(worstCaseSimple_4194304, 4194304);
  AllPrint();
  ShowDiffFp64();

  }
	return 0; // 
///////////////////////////////////////////////////////////////////////////////
  title(Array reduction with error evolution profiling);
  {
    double res_fp64 = 0.0;
    double diff = 0.0;

    _Float16 res_fp16 = 0.0;
    float res_fp32 = 0.0;
    __bf16 res_bf16 = 0.0;
    fp32_fp32_t res_fp32_fp32 = 0.0;
    fp32_bf16_t res_fp32_bf16 = 0.0;
    fp32_fp16_t res_fp32_fp16 = 0.0;
    fp16_fp16_t res_fp16_fp16 = 0.0;
    bf16_bf16_t res_bf16_bf16 = 0.0;
    bf16_fp16_t res_bf16_fp16 = 0.0;

#define RELATIVE_ERROR

#ifdef RELATIVE_ERROR
	#define dist(A, B) fabs((A - B)/A);
	#define dir(x, name) "../csv/" #name "_RELATIVE_ERR/" #x
#else
	#define dist(A, B) fabs(A - B);
	#define dir(x, name) "../csv/" #name "/" #x
#endif

#define IterProfile(arr, N, I, res)\
    {\
      std::fstream file;\
      std::filesystem::create_directory(dir(,));\
      std::filesystem::create_directory(dir(,arr));\
      file.open( dir(arr ## _ ## res ## _ ## I,arr), std::fstream::out);\
      std::cout << "generating " << dir(arr ## _ ## res ## _ ## I, arr) << std::endl;\
      res_fp64 = 0;\
      res = 0;\
      diff = 0;\
      for(int i=0; i < N; i+=I)\
      {\
        PerformRedSingleThread(arr+i, I, res);\
        PerformRedSingleThread(arr+i, I, res_fp64);\
        diff = dist(res_fp64, res);\
        file << diff << "," << std::endl;\
      }\
      file.close();\
    }//end define

#define AllIterSingleThread(arr, N, I)\
  IterProfile(arr, N, I, res_fp16)\
  IterProfile(arr, N, I, res_fp32)\
  IterProfile(arr, N, I, res_bf16)\
  IterProfile(arr, N, I, res_fp32_fp32)\
  IterProfile(arr, N, I, res_fp32_bf16)\
  IterProfile(arr, N, I, res_fp32_fp16)\
  IterProfile(arr, N, I, res_fp16_fp16)\
  IterProfile(arr, N, I, res_bf16_bf16)\
  IterProfile(arr, N, I, res_bf16_fp16)\

subtitle(error profiling);	
  AllIterSingleThread(_Float16_4096p19_jdygv, 4096, 1)
  AllIterSingleThread(_Float16_4096p19_jdygv_sorted, 4096, 1)
  AllIterSingleThread(_Float16_4096p19_jdygv_sorted_rev, 4096, 1)
  AllIterSingleThread(_Float16_16384p19_wxcse, 16384, 5)
  AllIterSingleThread(worstCase_4096, 4096, 1)

subtitle(significant digits profiling);

#define dirsd(x, name) "../csv/" #name "_SIGNIFICANTD/" #x
#define distsd(A, B) (A - B);

#define IterProfileSignificantd(arr, N, I, res)\
    {\
      std::fstream file;\
      std::filesystem::create_directory(dirsd(,));\
      std::filesystem::create_directory(dirsd(,arr));\
      file.open( dirsd(arr ## _ ## res ## _ ## I,arr), std::fstream::out);\
      std::cout << "generating " << dirsd(arr ## _ ## res ## _ ## I, arr) << std::endl;\
      res_fp64 = 0;\
      res = 0;\
      diff = 0;\
      for(int i=0; i < N; i+=I)\
      {\
        PerformRedSingleThread(arr+i, I, res);\
        PerformRedSingleThread(arr+i, I, res_fp64);\
        diff = distsd(res_fp64, res);\
        file << significand(res_fp64, diff) << "," << std::endl;\
      }\
      file.close();\
    }//end define

#define AllIterSingleThreadSD(arr, N, I)\
  IterProfileSignificantd(arr, N, I, res_fp16)\
  IterProfileSignificantd(arr, N, I, res_fp32)\
  IterProfileSignificantd(arr, N, I, res_bf16)\
  IterProfileSignificantd(arr, N, I, res_fp32_fp32)\
  IterProfileSignificantd(arr, N, I, res_fp32_bf16)\
  IterProfileSignificantd(arr, N, I, res_fp32_fp16)\
  IterProfileSignificantd(arr, N, I, res_fp16_fp16)\
  IterProfileSignificantd(arr, N, I, res_bf16_bf16)\
  IterProfileSignificantd(arr, N, I, res_bf16_fp16)\
	
	AllIterSingleThreadSD(_Float16_4096p19_jdygv, 4096, 1)
  AllIterSingleThreadSD(_Float16_4096p19_jdygv_sorted, 4096, 1)
  AllIterSingleThreadSD(_Float16_4096p19_jdygv_sorted_rev, 4096, 1)
  AllIterSingleThreadSD(_Float16_16384p19_wxcse, 16384, 5)
  AllIterSingleThreadSD(worstCase_4096, 4096, 1)

  }

  Kokkos::finalize();
}
