#ifndef KASTL_MERGE_H_INCLUDED
# define KASTL_MERGE_H_INCLUDED



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
class MergeWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef MergeWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  BaseType;

  typedef typename SequenceType::_InputIterator0Type InputIterator0Type;
  typedef typename SequenceType::_InputIterator1Type InputIterator1Type;
  typedef typename SequenceType::_OutputIteratorType OutputIteratorType;

  typedef ConstantType ComparatorType;

  static void reduce_merged_sequences
  (
   OutputIteratorType pos,
   OutputIteratorType mid,
   OutputIteratorType end,
   const ComparatorType& cmp
  )
  {
    typedef typename std::iterator_traits
      <OutputIteratorType>::value_type OutputValueType;

    // look for range not verifying pred then inplace_merge

    if ((pos == mid) || (mid == end))
      return ;

    const OutputIteratorType last0 = mid - 1;

    OutputIteratorType pos0;
    OutputIteratorType pos1;

    OutputValueType value = *mid;
    for (pos0 = last0; !cmp(*pos0, value); --pos0)
      ;

    value = *last0;
    for (pos1 = mid; !cmp(value, *pos1); ++pos1)
      ;

    if (pos0 != last0)
      std::inplace_merge(pos0, mid, pos1, cmp);
  }

public:

  MergeWork() : BaseType() {}

  MergeWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r) {}

  inline void prepare()
  {
    this->_res = this->_seq._obeg;
  }

  inline void compute(SequenceType& seq)
  {
    // init algo state
    InputIterator0Type ipos0 = seq._iseq0._beg;
    InputIterator0Type iend0 = seq._iseq0._end;
    InputIterator1Type ipos1 = seq._iseq1._beg;
    InputIterator1Type iend1 = seq._iseq1._end;
    OutputIteratorType opos = this->_res;

    if (ipos0 == iend0)
    {
      opos = std::copy(ipos1, iend1, opos);
      ipos1 = iend1;
    }
    else if (ipos1 == iend1)
    {
      opos = std::copy(ipos0, iend0, opos);
      ipos0 = iend0;
    }
    else // 2 input sequences are valid
    {
      while (ipos1 != iend1)
      {
	while ((ipos0 != iend0) && (*ipos0 < *ipos1))
	  *opos++ = *ipos0++;

	if (ipos0 == iend0)
	{
	  // terminate since we dont know if the
	  // remaining of this sequence has to
	  // be processed by this iteration
	  break;
	}

	while ((ipos1 != iend1) && !(*ipos0 < *ipos1))
	  *opos++ = *ipos1++;
      }
    }

    // store algo state

    seq._iseq0._beg = ipos0;
    seq._iseq1._beg = ipos1;

    this->_res = opos;
  }

  inline void reduce(const BaseType& tw)
  {
    reduce_merged_sequences
      (this->_seq._obeg, this->_res, tw._res, *this->_const);

    this->_res = tw._res;
  }

};

// tunning params
struct MergeTuningParams : Daouda0TuningParams
{
  static const enum TuningTag nano_tag = TAG_MERGE;
  static const size_t nano_size = 1024;
};

} // kastl::impl


template
<
  class InputIterator1,
  class InputIterator2,
  class OutputIterator,
  class CompareType,
  class ParamType
>
OutputIterator merge
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::In2OutSequence
    <InputIterator1, InputIterator2, OutputIterator>
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

  typedef OutputIterator ResultType;

  typedef kastl::impl::MergeWork
    <SequenceType, CompareType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  SequenceType sequence(ipos1, iend1, ipos2, iend2, opos);
  WorkType work(sequence, &comp, opos);

  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator,
 class CompareType 
>
OutputIterator merge
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::MergeTuningParams ParamType;

  return kastl::merge
    <InputIterator1, InputIterator2, OutputIterator, CompareType, ParamType>
    (ipos1, iend1, ipos2, iend2, opos, comp);
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator
>
OutputIterator merge
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos
)
{
  typedef typename std::iterator_traits
    <InputIterator1>::value_type ValueType;

  typedef std::less<ValueType> CompareType;

  return kastl::merge
    <InputIterator1, InputIterator2, OutputIterator, CompareType>
    (ipos1, iend1, ipos2, iend2, opos, CompareType());
}

} // kastl


#endif // ! KASTL_MERGE_H_INCLUDED
