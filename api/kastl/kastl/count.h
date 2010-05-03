#ifndef KASTL_COUNT_H_INCLUDED
# define KASTL_COUNT_H_INCLUDED



#include <stddef.h>
#include <algorithm>
#include <iterator>
#include "kastl/kastl_impl.h"



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
struct CountWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef CountWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  CountWork() : BaseType() {}

  CountWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
    : BaseType(s, c, r) { prepare(); }

  inline void prepare()
  {
    this->_res = 0;
  }

  inline void compute(SequenceType& seq)
  {
    this->_res += std::count(seq.begin(), seq.end(), *this->_const);
    seq.advance();
  }

  inline void reduce(const SelfType& thief_work)
  {
    this->_res += thief_work._res;
  }

};

// tunning params
typedef Daouda0TuningParams CountTuningParams;

} // kastl::impl


template
<
  typename ForwardIterator,
  typename ValueType,
  typename ParamType
>
typename std::iterator_traits<ForwardIterator>::difference_type
count
(
 ForwardIterator begin,
 ForwardIterator end,
 const ValueType& val
)
{
  typedef kastl::impl::InSequence<ForwardIterator>
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

  typedef ValueType ConstantType;

  typedef typename
    std::iterator_traits<ForwardIterator>::difference_type
    ResultType;

  typedef kastl::impl::CountWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end), &val, 0);

  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
 typename ForwardIterator,
 typename ValueType
>
typename std::iterator_traits<ForwardIterator>::difference_type
count
(
 ForwardIterator begin,
 ForwardIterator end,
 const ValueType& val
)
{
  typedef kastl::impl::CountTuningParams ParamType;

  return kastl::count
    <ForwardIterator, ValueType, ParamType>
    (begin, end, val);
}

} // kastl



#endif // ! KASTL_COUNT_H_INCLUDED
