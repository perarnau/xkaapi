#ifndef KASTL_MISMATCH_HH_INCLUDED
# define KASTL_MISMATCH_HH_INCLUDED



#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include "kastl_impl.hh"



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
class MismatchWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef MismatchWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

public:

  MismatchWork() : BaseType() {}

  MismatchWork
  (const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r) {}

  inline void prepare() {}

  inline void compute(const SequenceType& seq)
  {
    this->_res = std::mismatch
      (seq._seq0._beg, seq._seq0._end, seq._beg1, *this->_const);

    if (this->_res.first == seq._seq0._end)
      return ;

    this->_is_done = true;
  }

  inline void reduce(const BaseType& tw)
  {
    this->_is_done = tw._is_done;
    this->_res = tw._res;
  }
};

// tunning params
struct MismatchTuningParams : Daouda1TuningParams
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
std::pair<InputIterator1, InputIterator2>
mismatch
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

  typedef std::pair<InputIterator1, InputIterator2>
    ResultType;

  typedef BinaryPredicate ConstantType;

  typedef kastl::impl::MismatchWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(first1, last1, first2), &pred, ResultType(first1, first2));
  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
  class InputIterator1,
  class InputIterator2,
  class BinaryPredicate
>
std::pair<InputIterator1, InputIterator2>
mismatch
(
 InputIterator1 first1,
 InputIterator1 last1,
 InputIterator2 first2,
 BinaryPredicate pred
)
{
  typedef kastl::impl::MismatchTuningParams ParamType;

  return kastl::mismatch
    <InputIterator1, InputIterator2, BinaryPredicate, ParamType>
    (first1, last1, first2, pred);
}


template
<
  class InputIterator1,
  class InputIterator2
>
std::pair<InputIterator1, InputIterator2>
mismatch
(
 InputIterator1 first1,
 InputIterator1 last1,
 InputIterator2 first2 
)
{
  typedef typename std::iterator_traits
    <InputIterator1>::value_type
    ValueType;

  return kastl::mismatch
    (first1, last1, first2, std::equal_to<ValueType>());
}

} // kastl


#endif // ! KASTL_MISMATCH_HH_INCLUDED
