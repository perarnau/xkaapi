#ifndef KASTL_ACCUMULATE_HH_INCLUDED
# define KASTL_ACCUMULATE_HH_INCLUDED



#include <stddef.h>
#include <algorithm>
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
struct AccumulateWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef AccumulateWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  AccumulateWork() : BaseType() {}

  AccumulateWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
    : BaseType(s, c, r)
  { }

  inline void prepare()
  {
    this->_res._is_valid = false;
  }

  inline void reduce_result(const ResultType& res)
  {
    // assume res is valid

    if (this->_res._is_valid == true)
      this->_res._value = (*this->_const)(this->_res._value, res._value);
    else
      this->_res = res;
  }

  inline void compute(SequenceType& seq)
  {
    // todo: could be removed in prepare()
    typename SequenceType::SizeType i = 0;
    if (this->_res._is_valid == false)
    {
      this->_res = ResultType(*seq.begin(), true);
      i = 1;
    }

    this->_res._value = std::accumulate
      (seq.begin() + i, seq.end(), this->_res._value, *this->_const);

    seq.advance(seq.size());
  }

  inline void reduce(const SelfType& thief_work)
  {
    // todo: see above
    if (thief_work._res._is_valid == false)
      return ;

    reduce_result(thief_work._res);
  }

};

// result type
template<typename T>
struct AccumulateResult
{
  typedef T ValueType;

  T _value;
  bool _is_valid;

  AccumulateResult(const T& value, bool is_valid)
    : _value(value), _is_valid(is_valid) {}
};

// tunning params
typedef Daouda0TuningParams AccumulateTuningParams;

} // kastl::impl


template
<
  typename RandomAccessIterator,
  typename AccumulatorType,
  typename BinaryFunction,
  typename ParamType
>
AccumulatorType accumulate
(
 RandomAccessIterator begin,
 RandomAccessIterator end,
 AccumulatorType val,
 BinaryFunction func
)
{
  typedef kastl::impl::InSequence<RandomAccessIterator>
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

  typedef BinaryFunction ConstantType;

  typedef kastl::impl::AccumulateResult<AccumulatorType> ResultType;

  typedef kastl::impl::AccumulateWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end), &func, ResultType(val, true));
  kastl::impl::compute<WorkType>(work);
  return work._res._value;
}


template
<
  typename RandomAccessIterator,
  typename AccumulatorType,
  typename BinaryFunction
>
AccumulatorType accumulate
(
 RandomAccessIterator begin,
 RandomAccessIterator end,
 AccumulatorType val,
 BinaryFunction func
)
{
  typedef kastl::impl::AccumulateTuningParams ParamType;
  return kastl::accumulate
    <RandomAccessIterator, AccumulatorType, BinaryFunction, ParamType>
    (begin, end, val, func);
}


template
<
 typename RandomAccessIterator,
 typename AccumulatorType
>
AccumulatorType accumulate
(
 RandomAccessIterator begin,
 RandomAccessIterator end,
 AccumulatorType val
)
{
  return kastl::accumulate
    (begin, end, val, std::plus<AccumulatorType>());
}

} // kastl



#endif // ! KASTL_ACCUMULATE_HH_INCLUDED
