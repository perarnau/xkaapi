#ifndef KASTL_SWAP_RANGES_H_INCLUDED
# define KASTL_SWAP_RANGES_H_INCLUDED



#include <algorithm>
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
class SwapRangesWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef SwapRangesWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

public:

  SwapRangesWork() : BaseType() {}

  SwapRangesWork
  (const SequenceType& s, const ResultType& r)
  : BaseType(s, NULL, r) {}

  inline void compute(SequenceType& seq)
  {
    this->_res = std::swap_ranges(seq.begin0(), seq.end0(), seq.begin1());
    seq.advance();
  }

  inline void reduce(const BaseType& tw)
  {
    this->_res = tw._res;
  }

};

// tunning params
typedef Daouda0TuningParams SwapRangesTuningParams;

} // kastl::impl


template
<
  class ForwardIterator1,
  class ForwardIterator2,
  class ParamType
>
ForwardIterator2 swap_ranges
(
 ForwardIterator1 ipos,
 ForwardIterator1 iend,
 ForwardIterator2 ipos2
)
{
  typedef kastl::impl::In2EqSizedSequence
    <ForwardIterator1, ForwardIterator2>
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

  typedef ForwardIterator2 ResultType;

  typedef kastl::impl::InvalidConstantType ConstantType;

  typedef kastl::impl::SwapRangesWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;
  
  WorkType work(SequenceType(ipos, iend, ipos2), ipos2);
  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
  class ForwardIterator1,
  class ForwardIterator2
>
ForwardIterator2 swap_ranges
(
 ForwardIterator1 ipos,
 ForwardIterator1 iend,
 ForwardIterator2 ipos2
)
{
  typedef kastl::impl::SwapRangesTuningParams ParamType;
  return kastl::swap_ranges
    <ForwardIterator1, ForwardIterator2, ParamType>
    (ipos, iend, ipos2);
}

} // kastl


#endif // ! KASTL_SWAP_RANGES_H_INCLUDED
