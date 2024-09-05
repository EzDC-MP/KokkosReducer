// Enzo De Carvalho Bittencourt <ezdecarvalho@gmail.com>
// Error Free Transform reducer using Kahan Summation for Kokkos
#include <Kokkos_Core.hpp>

template <typename ScalarType, typename AccumulatorType, typename Space>
struct KahanReducer
{
  public:
    // Required
    typedef KahanReducer<ScalarType, AccumulatorType, Space> reducer;
    typedef AccumulatorType value_type;
    typedef Kokkos::View<value_type, Space> result_view_type;

  private:
    value_type& result; 
    //where the result reside at the end of the reduction. Called at construc.
    //we own nothing here

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
