#ifndef KASTL_SET_UNION_HH_INCLUDED
# define KASTL_SET_UNION_HH_INCLUDED



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
class SetUnionWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef SetUnionWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  typedef typename SequenceType::_InputIterator0Type InputIterator0Type;
  typedef typename SequenceType::_InputIterator1Type InputIterator1Type;
  typedef typename SequenceType::_OutputIteratorType OutputIteratorType;

  typedef typename std::iterator_traits<OutputIteratorType>::value_type OutputValueType;

  typedef ConstantType ComparatorType;

  // reduce union results
  static OutputIteratorType reduce_union_sequences
  (
   OutputIteratorType beg0, OutputIteratorType end0,
   OutputIteratorType beg1, OutputIteratorType end1,
   const ComparatorType& cmp
  )
  {
    if ((beg0 == end0) || (beg1 == end1))
      return end1;

    const OutputIteratorType last0 = end0 - 1;
    OutputIteratorType mid0;
    OutputIteratorType mid1;

    OutputValueType value = *beg1;
    for (mid0 = last0; (mid0 != beg0) && !cmp(*mid0, value); --mid0)
      ;

    value = *last0;
    for (mid1 = beg1; (mid1 != end1) && !cmp(value, *mid1); ++mid1)
      ;

    const OutputIteratorType old_mid1 = mid1;

    if (end0 != beg1)
      mid1 = std::copy(beg1, mid1, end0);

    std::inplace_merge(mid0, end0, mid1, cmp);
    mid1 = std::unique(mid0, mid1);
    end1 = std::copy(old_mid1, end1, mid1);

    return end1;
  }

  template<typename IteratorType> static void skip_value
  (
   IteratorType& pos, const IteratorType& end,
   typename std::iterator_traits<IteratorType>::value_type & value
  )
  {
    for (; (pos != end) && (*pos == value); ++pos)
      ;
  }

public:

  SetUnionWork() : BaseType() {}

  SetUnionWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
    : BaseType(s, c, r) {}

  inline void prepare()
  {
    this->_res._obeg = this->_seq._obeg;
    this->_res._opos = this->_seq._obeg;
    this->_res._has_last_value = false;
  }

  inline void compute(SequenceType& seq)
  {
    InputIterator0Type pos0 = seq._iseq0._beg;
    InputIterator0Type end0 = seq._iseq0._end;
    InputIterator1Type pos1 = seq._iseq1._beg;
    InputIterator1Type end1 = seq._iseq1._end;
    OutputIteratorType opos = this->_res._opos;

    OutputValueType last_value;

    if (pos1 > end1)
    {
      printf("compute >>\n"); fflush(stdout); exit(-1);
    }

    // prepare last value
    if (this->_res._has_last_value == false)
    {
      if (pos0 == end0)
	last_value = *pos1;
      else if (pos1 == end1)
	last_value = *pos0;
      else if ((*this->_const)(*pos0, *pos1))
	last_value = *pos0;
      else // if (!cmp(pos0, pos1))
	last_value = *pos1;

      this->_res._has_last_value = true;
      this->_res._last_value = last_value;
      *opos++ = last_value;
    }
    else
    {
      last_value = this->_res._last_value;
    }

    if ((pos0 == end0) && (pos1 != end1))
    {
      skip_value(pos1, end1, last_value);
      opos = std::unique_copy(pos1, end1, opos);
      last_value = *(opos - 1);
      pos1 = end1;
    }
    else if ((pos1 == end1) && (pos0 != end0))
    {
      skip_value(pos0, end0, last_value);
      opos = std::unique_copy(pos0, end0, opos);
      last_value = *(opos - 1);
      pos0 = end0;
    }
    else // seq0.size(), seq1.size()
    {
      const ConstantType cmp = *this->_const;

      while (pos1 != end1)
      {
	for (; (pos0 != end0) && cmp(*pos0, *pos1); ++pos0)
	{
	  if (last_value != *pos0)
	  {
	    *opos++ = *pos0;
	    last_value = *pos0;
	  }
	}

	if (pos0 == end0)
	  break;

	for (; (pos1 != end1) && !cmp(*pos0, *pos1); ++pos1)
	{
	  if (last_value != *pos1)
	  {
	    *opos++ = *pos1;
	    last_value = *pos1;
	  }
	}
      }
    }

    // store state

    seq._iseq0._beg = pos0;
    seq._iseq1._beg = pos1;

    if (pos1 > end1)
    {
      printf("compute <<\n"); fflush(stdout); exit(-1);
    }

    this->_res._last_value = last_value;
    this->_res._opos = opos;
  }

  inline void reduce(const BaseType& tw)
  {
    if (tw._res._obeg > tw._res._opos)
    {
      printf("reduce >>\n"); fflush(stdout); exit(-1);
    }

    this->_res._opos = reduce_union_sequences
    (
     this->_res._obeg,
     this->_res._opos,
     tw._res._obeg,
     tw._res._opos,
     *this->_const
    );

    if (this->_res._obeg > this->_res._opos)
    {
      printf("reduce <<\n"); fflush(stdout); exit(-1);
    }

    this->_res._has_last_value = tw._res._has_last_value;
    this->_res._last_value = tw._res._last_value;
  }

};


// result type
template<typename IteratorType>
struct SetUnionResult
{
  IteratorType _obeg;
  IteratorType _opos;

  bool _has_last_value;
  typename std::iterator_traits<IteratorType>::value_type _last_value;

  SetUnionResult(const IteratorType& obeg)
    : _obeg(obeg), _opos(obeg), _has_last_value(false) {}
};


// tunning params
struct SetUnionTuningParams : Daouda0TuningParams
{
  static const enum TuningTag nano_tag = TAG_MERGE;
  static const size_t nano_size = 512;
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
OutputIterator set_union
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

  typedef kastl::impl::SetUnionResult<OutputIterator> ResultType;

  typedef kastl::impl::SetUnionWork
    <SequenceType, CompareType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  SequenceType sequence(ipos1, iend1, ipos2, iend2, opos);
  WorkType work(sequence, &comp, ResultType(opos));
  kastl::impl::compute<WorkType>(work);
  return work._res._opos;
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator,
 class CompareType 
>
OutputIterator set_union
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::SetUnionTuningParams ParamType;

  return kastl::set_union
    <InputIterator1, InputIterator2, OutputIterator, CompareType, ParamType>
    (ipos1, iend1, ipos2, iend2, opos, comp);
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator
>
OutputIterator set_union
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

  return kastl::set_union
    <InputIterator1, InputIterator2, OutputIterator, CompareType>
    (ipos1, iend1, ipos2, iend2, opos, CompareType());
}

} // kastl


#endif // ! KASTL_SET_UNION_HH_INCLUDED
