// Enzo De Carvalho Bittencourt <ezdecarvalho@gmail.com>
// Error Free Transform reducer using Kahan Summation for Kokkos
#include <Kokkos_Core.hpp>

template <typename ScalarType, typename AccumulatorType, typename Space>
struct KahanReducer
{
  public:
    // Required
    typedef KahanReducer<ScalarType, AccumulatorType, Space> reducer;
    typedef ScalarType value_type;
    typedef Kokkos::View<value_type, Space> result_view_type;

  private:
    value_type& result; 
    //where the result reside at the end of the reduction. Called at construc.
    //we own nothing here
    AccumulatorType acc;

  public:
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const
    //'val's (second parameters of all functor) are reduced this way
    {
      dest += src; 
      printf("&dest : %p, &src : %p\n", &dest, &src);
    } //TEMPORARY FOR TESTING

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return result;} 

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return result_view_type(&result, 1);}  

    //Optional
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const 
    //val is the second param called in the functor. Initialized by kokkos ig 
    {val = 0; printf("init &res : %p, &val : %p\n", &result, &val);}

    KOKKOS_INLINE_FUNCTION
    void final(value_type& val) const 
    {printf("final &res : %p, &val : %p\n", &result, &val);}

    //Constructors
    KOKKOS_INLINE_FUNCTION
    KahanReducer(value_type& result_) : result(result_) {};

    //KOKKOS_INLINE_FUNCTION
    //Reducer(const result_view_type& value_);
};

template <typename ScalarType, typename AccumulatorType=ScalarType>
struct Scalarhilo
{
  public:
    ScalarType hi;      //high order value
    AccumulatorType lo; //low order value

  //constructors
    KOKKOS_INLINE_FUNCTION
    Scalarhilo(){ hi = 0; lo = 0;}
    
    KOKKOS_INLINE_FUNCTION
    Scalarhilo(ScalarType a){hi = a; lo = 0;}

    KOKKOS_INLINE_FUNCTION
    Scalarhilo(ScalarType a, AccumulatorType b){hi = a; lo = b;}

  //operators (add)
    KOKKOS_INLINE_FUNCTION
    Scalarhilo& operator+=(Scalarhilo x) //add assignement 
    {
      ScalarType sum;
      ScalarType err;
      ScalarType tmp;
      ScalarType tmp_;
      
      sum = hi + x.hi;
      //printf("sum : %.e\n", sum);
      tmp = sum - hi;
      tmp_ = sum - x.hi;
      //printf("tmp : %.e, tmp_ : %.e\n", tmp, tmp_);
      err = (hi - tmp_) + (x.hi - tmp);
      //printf("err: %.e\n", err);

      hi = sum;
      lo += x.lo + (AccumulatorType)err;

      return *this;
    }

  //finalize (number + error)
    KOKKOS_INLINE_FUNCTION
    ScalarType finalize(){return this->hi + (ScalarType)this->lo;} 

};

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<typename ScalarType, typename AccumulatorType>
   struct reduction_identity< Scalarhilo<ScalarType, AccumulatorType> > 
   {
      KOKKOS_FORCEINLINE_FUNCTION static 
      Scalarhilo<ScalarType, AccumulatorType> sum() 
      {return Scalarhilo<ScalarType,AccumulatorType>();}
   };
  
   template <>
   struct reduction_identity< __bf16 > 
   {
      KOKKOS_FORCEINLINE_FUNCTION static 
      __bf16 sum() 
      {return (__bf16)0.0;}
   };
   
   template <>
   struct reduction_identity< _Float16 > 
   {
      KOKKOS_FORCEINLINE_FUNCTION static 
      _Float16 sum() 
      {return (_Float16)0.0;}
   };
}
