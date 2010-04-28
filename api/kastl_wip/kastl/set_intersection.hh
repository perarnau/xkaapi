#ifndef KASTL_SET_INTERSECTION_HH_INCLUDED
# define KASTL_SET_INTERSECTION_HH_INCLUDED



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
class SetIntersectionWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef SetIntersectionWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  inline void reduce_result(const ResultType& res)
  {
    // bug: cannot handle the following case:
    // [0 9][10 20]
    // [0 8][ 9 20]
    // solved by the ExclusiveOrderedSequence:
    // [0 9][10 20]
    // [0 9][10 20]

    if (this->_res.second != res.first)
    {
      typename SequenceType::OutputIterator ipos = this->_res.second;
      typename SequenceType::OutputIterator jpos = res.first;
      typename SequenceType::OutputIterator jend = res.second;

      for (; jpos != jend; ++jpos, ++ipos)
	*ipos = *jpos;

      this->_res.second = ipos;
    }
    else
    {
      this->_res.second = res.second;
    }
  }

public:

  SetIntersectionWork() : BaseType() {}

  SetIntersectionWork
  (const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r) {}

  inline void prepare()
  {
    this->_res = ResultType(this->_seq._obeg, this->_seq._obeg);
  }

  inline void compute(const SequenceType& seq)
  {
    ResultType res;

    res.first = this->_res.second;

    res.second = std::set_intersection
    (
     seq._iseq0._beg,
     seq._iseq0._end,
     seq._iseq1._beg,
     seq._iseq1._end,
     this->_res.second,
     *this->_const
    );

    reduce_result(res);
  }

  inline void reduce(const BaseType& tw)
  {
    reduce_result(tw._res);
  }

};

// tunning params
struct SetIntersectionTuningParams : Daouda0TuningParams
{
  static const size_t nano_size = 512;
};


// temporary fix for the abovementionned case
template
<
  typename InputIterator0Type,
  typename InputIterator1Type,
  typename OutputIteratorType,
  typename ComparatorType
>
class ExclusiveOrderedSequence
: public OrderedSequence
<
  InputIterator0Type,
  InputIterator1Type,
  OutputIteratorType,
  ComparatorType
>
{
public:

  typedef ExclusiveOrderedSequence
  <
  InputIterator0Type,
  InputIterator1Type,
  OutputIteratorType,
  ComparatorType
  >
  SequenceType;

  typedef OrderedSequence
  <
  InputIterator0Type,
  InputIterator1Type,
  OutputIteratorType,
  ComparatorType
  >
  BaseType;
  
  typedef typename std::iterator_traits
  <InputIterator0Type>::value_type InputValue0Type;
  
  typedef typename std::iterator_traits
  <InputIterator1Type>::value_type InputValue1Type;
  
  typedef typename std::iterator_traits
  <OutputIteratorType>::value_type OutputValueType;
  
  typedef typename std::iterator_traits
  <InputIterator0Type>::difference_type SizeType;

  typedef OutputIteratorType OutputIterator;

  ExclusiveOrderedSequence()
    : BaseType() {}

  ExclusiveOrderedSequence
  (
   const InputIterator0Type& ipos0,
   const InputIterator0Type& iend0,
   const InputIterator1Type& ipos1,
   const InputIterator1Type& iend1,
   const OutputIteratorType& obeg,
   const ComparatorType& comp
  ) : BaseType(ipos0, iend0, ipos1, iend1, obeg, comp) {}

  void rsplit(SequenceType& oseq, size_t size)
  {
    // size is the size of the largest sequence
    swap_if_greater(BaseType::_iseq1, BaseType::_iseq0);

    InputIterator0Type mid0 = BaseType::_iseq0._beg + size - 1;
    InputIterator1Type mid1 = find_with_pred
      (BaseType::_iseq1._beg, BaseType::_iseq1._end, *mid0, BaseType::_comp);

    // give [mid, end[
    oseq._iseq0 = BasicSequence<InputIterator0Type>(mid0, BaseType::_iseq0._end);
    oseq._iseq1 = BasicSequence<InputIterator1Type>(mid1, BaseType::_iseq1._end);

    oseq._obeg = BaseType::_obeg + std::distance(BaseType::_iseq0._beg, mid0) +
      std::distance(BaseType::_iseq1._beg, mid1);

    // keep [beg, mid[
    BaseType::_iseq0._end = mid0;
    BaseType::_iseq1._end = mid1;
  }

  void split(SequenceType& oseq, size_t size)
  {
    // size is the size of the largest sequence
    swap_if_greater(BaseType::_iseq1, BaseType::_iseq0);

    InputIterator0Type mid0 = BaseType::_iseq0._beg + size;
    InputIterator1Type mid1 = find_with_pred
      (BaseType::_iseq1._beg, BaseType::_iseq1._end, *(mid0 - 1), BaseType::_comp);

    const InputValue0Type value = *(mid0 - 1);
    for (; (mid1 != BaseType::_iseq1._end) && (*mid1 == value); ++mid1)
      ;

    // give [beg, mid]
    oseq._obeg = BaseType::_obeg;
    oseq._iseq0 = BasicSequence<InputIterator0Type>(BaseType::_iseq0._beg, mid0);
    oseq._iseq1 = BasicSequence<InputIterator1Type>(BaseType::_iseq1._beg, mid1);

    // keep ]mid, end[
    BaseType::_obeg += std::distance(BaseType::_iseq0._beg, mid0) +
      std::distance(BaseType::_iseq1._beg, mid1);
    BaseType::_iseq0 = BasicSequence<InputIterator0Type>
      (mid0, BaseType::_iseq0._end);
    BaseType::_iseq1 = BasicSequence<InputIterator1Type>
      (mid1, BaseType::_iseq1._end);
  }
    
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
OutputIterator set_intersection
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::ExclusiveOrderedSequence
    <InputIterator1, InputIterator2, OutputIterator, CompareType>
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

  typedef std::pair<OutputIterator, OutputIterator> ResultType;

  typedef kastl::impl::SetIntersectionWork
    <SequenceType, CompareType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  SequenceType sequence(ipos1, iend1, ipos2, iend2, opos, comp);
  WorkType work(sequence, &comp, ResultType(opos, opos));
  kastl::impl::compute<WorkType>(work);
  return work._res.second;
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator,
 class CompareType 
>
OutputIterator set_intersection
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::SetIntersectionTuningParams ParamType;

  return kastl::set_intersection
    <InputIterator1, InputIterator2, OutputIterator, CompareType, ParamType>
    (ipos1, iend1, ipos2, iend2, opos, comp);
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator
>
OutputIterator set_intersection
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

  return kastl::set_intersection
    <InputIterator1, InputIterator2, OutputIterator, CompareType>
    (ipos1, iend1, ipos2, iend2, opos, CompareType());
}

} // kastl


#endif // ! KASTL_SET_INTERSECTION_HH_INCLUDED
