#ifndef KASTL_EQUAL_H_INCLUDED
# define KASTL_EQUAL_H_INCLUDED



#include <algorithm>
#include <functional>
#include <iterator>
#include "kastl_impl.h"



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
class EqualWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef EqualWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  typedef typename SequenceType::_Iterator0Type Iterator0Type;
  typedef typename SequenceType::_Iterator1Type Iterator1Type;

public:

  EqualWork() : BaseType() {}

  EqualWork
  (const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r) {}

  inline void prepare()
  {
    this->_res = true;
  }

  inline void compute(SequenceType& seq)
  {
    Iterator0Type pos0 = seq._seq0._beg;
    Iterator1Type pos1 = seq._beg1;

    for (; (pos0 != seq._seq0._end) && (*pos0 == *pos1); ++pos0, ++pos1)
      ;

    seq._seq0._beg = pos0;
    seq._beg1 = pos1;

    if (pos0 == seq._seq0._end)
      return ;

    this->_res = false;
    this->_is_done = true;
  }

  inline void reduce(const BaseType& tw)
  {
    this->_is_done = tw._is_done;
    this->_res = tw._res;
  }
};

// tunning params
struct EqualTuningParams : Daouda1TuningParams
{
  static const enum TuningTag macro_tag = TAG_LINEAR;
  static const size_t macro_min_size = 1024;
  static const size_t macro_max_size = 32768;
  static const size_t macro_step_size = 2048;
};

} // kastl::impl


template
<
  class InputIterator1,
  class InputIterator2,
  class BinaryPredicate,
  class ParamType
>
bool equal
(
 InputIterator1 first1,
 InputIterator1 last1,
 InputIterator2 first2,
 BinaryPredicate pred
)
{
  typedef kastl::impl::In2EqSizedSequence
    <InputIterator1, InputIterator2>
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

  typedef bool ResultType;

  typedef BinaryPredicate ConstantType;

  typedef kastl::impl::EqualWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(first1, last1, first2), &pred, true);
  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
  class InputIterator1,
  class InputIterator2,
  class BinaryPredicate
>
bool equal
(
 InputIterator1 first1,
 InputIterator1 last1,
 InputIterator2 first2,
 BinaryPredicate pred
)
{
  typedef kastl::impl::EqualTuningParams ParamType;

  return kastl::equal
    <InputIterator1, InputIterator2, BinaryPredicate, ParamType>
    (first1, last1, first2, pred);
}


template
<
  class InputIterator1,
  class InputIterator2
>
bool equal
(
 InputIterator1 first1,
 InputIterator1 last1,
 InputIterator2 first2 
)
{
  typedef typename std::iterator_traits
    <InputIterator1>::value_type
    ValueType;

  return kastl::equal(first1, last1, first2, std::equal_to<ValueType>());
}

} // kastl


#endif // ! KASTL_EQUAL_H_INCLUDED
