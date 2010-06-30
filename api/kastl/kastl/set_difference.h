#ifndef KASTL_SET_DIFFERENCE_H_INCLUDED
# define KASTL_SET_DIFFERENCE_H_INCLUDED



#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
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
class SetDifferenceWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef SetDifferenceWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  inline void reduce_result(const ResultType& res)
  {
    typename SequenceType::OutputIterator res_first = res.first;
    typename SequenceType::OutputIterator this_second = this->_res.second;

    if ((this->_res.first != this->_res.second) && (res.first != res.second))
    {
      // handle this case:
      // this->_res: [0 - 9][10 - 20]
      // res:        [0 - 8][ 9 - 20]
      // 9 should be excluded from both seqs

      if (!(*this->_const)(*(this_second - 1), *res_first))
      {
	++res_first;
	--this_second;
      }
    }

    if (this_second != res_first)
    {
      typename SequenceType::OutputIterator ipos = this_second;
      typename SequenceType::OutputIterator jpos = res_first;
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

  SetDifferenceWork() : BaseType() {}

  SetDifferenceWork
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

    res.second = std::set_difference
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
struct SetDifferenceTuningParams : Daouda0TuningParams
{
  static const size_t nano_size = 4096;
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
OutputIterator set_difference
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::OrderedSequence
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

  typedef kastl::impl::SetDifferenceWork
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
OutputIterator set_difference
(
 InputIterator1 ipos1,
 InputIterator1 iend1,
 InputIterator2 ipos2,
 InputIterator2 iend2,
 OutputIterator opos,
 CompareType comp
)
{
  typedef kastl::impl::SetDifferenceTuningParams ParamType;

  return kastl::set_difference
    <InputIterator1, InputIterator2, OutputIterator, CompareType, ParamType>
    (ipos1, iend1, ipos2, iend2, opos, comp);
}


template
<
 class InputIterator1,
 class InputIterator2,
 class OutputIterator
>
OutputIterator set_difference
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

  return kastl::set_difference
    <InputIterator1, InputIterator2, OutputIterator, CompareType>
    (ipos1, iend1, ipos2, iend2, opos, CompareType());
}

} // kastl


#endif // ! KASTL_SET_DIFFERENCE_H_INCLUDED
