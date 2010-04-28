#ifndef KASTL_INNER_PRODUCT_HH_INCLUDED
# define KASTL_INNER_PRODUCT_HH_INCLUDED



#include <numeric>
#include <functional>
#include <iterator>
#include "kastl/kastl_impl.hh"



namespace kastl
{

namespace impl
{

template
<
  typename SequenceType,
  typename ConstantType,
  typename ResultType,
  typename MacroType,
  typename NanoType,
  typename SplitterType
>
class InnerProductWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef InnerProductWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

public:

  InnerProductWork() : BaseType() {}

  InnerProductWork
  (const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r) {}

  inline void prepare()
  {
    this->_res._is_valid = false;
  }

  inline void reduce_result(const ResultType& res)
  {
    if (this->_res._is_valid == true)
      this->_res._value = (this->_const->_f1)(this->_res._value, res._value);
    else
      this->_res = res;
  }

  inline void compute(SequenceType& seq)
  {
    typename SequenceType::SizeType i = 0;
    if (this->_res._is_valid == false)
    {
      this->_res = ResultType(*seq.begin0());
      i = 1;
    }

    this->_res._value = std::inner_product
    (
     seq.begin0() + i, seq.end0(), seq.begin1(),
     this->_res._value,
     this->_const->_f1, this->_const->_f2
    );

    seq.advance();
  }

  inline void reduce(const BaseType& tw)
  {
    if (tw._res._is_valid == false)
      return ;

    reduce_result(tw._res);
  }

};

template<typename T>
struct InnerProductResult
{
  T _value;
  bool _is_valid;

  InnerProductResult(const T& value)
    : _value(value), _is_valid(true) {}
};

template<typename T1, typename T2>
struct InnerProductConstant
{
  T1 _f1;
  T2 _f2;

  InnerProductConstant(const T1& f1, const T2& f2)
    : _f1(f1), _f2(f2) {}
};

// tunning params
typedef Daouda0TuningParams InnerProductTuningParams;

} // kastl::impl


template
<
  class RandomAccessIterator1,
  class RandomAccessIterator2,
  class ValueType,
  class BinaryFunction1,
  class BinaryFunction2,
  class ParamType
>
ValueType inner_product
(
 RandomAccessIterator1 ipos,
 RandomAccessIterator1 iend,
 RandomAccessIterator2 ipos2,
 ValueType init_value,
 BinaryFunction1 f1,
 BinaryFunction2 f2
)
{
  typedef kastl::impl::In2EqSizedSequence
    <RandomAccessIterator1, RandomAccessIterator2>
    SequenceType;

  typedef typename kastl::impl::make_macro_type
    <ParamType::macro_tag, ParamType, SequenceType>::Type
    MacroType;

  typedef typename kastl::impl::make_nano_type
    <ParamType::nano_tag, ParamType, SequenceType>::Type
    NanoType;

  typedef typename kastl::impl::make_splitter_type
    <ParamType::splitter_tag, ParamType>::Type
    SplitterType;

  typedef kastl::impl::InnerProductResult<ValueType>
    ResultType;

  typedef kastl::impl::InnerProductConstant<BinaryFunction1, BinaryFunction2>
    ConstantType;

  typedef kastl::impl::InnerProductWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  SequenceType sequence(ipos, iend, ipos2);
  ResultType result(init_value);
  ConstantType constant(f1, f2);
  WorkType work(sequence, &constant, result);
  kastl::impl::compute<WorkType>(work);
  return work._res._value;
}


template
<
  class RandomAccessIterator1,
  class RandomAccessIterator2,
  class ValueType,
  class BinaryFunction1,
  class BinaryFunction2
>
ValueType inner_product
(
 RandomAccessIterator1 ipos,
 RandomAccessIterator1 iend,
 RandomAccessIterator2 ipos2,
 ValueType init_value,
 BinaryFunction1 f1,
 BinaryFunction2 f2
)
{
  typedef typename kastl::impl::InnerProductTuningParams
    ParamType;

  return kastl::inner_product
    <
      RandomAccessIterator1,
      RandomAccessIterator2,
      ValueType,
      BinaryFunction1,
      BinaryFunction2,
      ParamType
    >
    (ipos, iend, ipos2,  init_value, f1, f2);
}


template
<
  class RandomAccessIterator1,
  class RandomAccessIterator2,
  class ValueType
>
ValueType inner_product
(
 RandomAccessIterator1 begin,
 RandomAccessIterator1 end,
 RandomAccessIterator2 begin2,
 ValueType init_value
)
{
  typedef std::plus<ValueType> BinaryFunction1;
  typedef std::multiplies<ValueType> BinaryFunction2;
  
  return kastl::inner_product
    (
     begin, end, begin2,
     init_value,
     BinaryFunction1(),
     BinaryFunction2()
    );
}

} // kastl


#endif // ! KASTL_INNER_PRODUCT_HH_INCLUDED
