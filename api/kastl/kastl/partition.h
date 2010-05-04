#ifndef KASTL_PARTITION_H_INCLUDED
# define KASTL_PARTITION_H_INCLUDED



#include <stddef.h>
#include <algorithm>
#include <iterator>
#include <functional>
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
class PartitionWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef PartitionWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  inline void reduce_result(const ResultType& res)
  {
    // [l0 - g0, l1 - g1[ -> [l0 - l1, g0 - g1[

    if (res._pivot != res._seq._beg)
    {
      typename SequenceType::_IteratorType const gend = this->_res._seq._end;
      typename SequenceType::_IteratorType const lend = res._seq._beg - 1;

      typename SequenceType::_IteratorType gi = this->_res._pivot;
      typename SequenceType::_IteratorType li = res._pivot - 1;
      typename SequenceType::_IteratorType prev_li = res._pivot;

      for (; (gi != gend) && (li != lend); ++gi, --li)
      {
	std::swap(*gi, *li);
	prev_li = li;
      }

      if (gi != gend)
	this->_res._pivot = gi;
      else // li != lend
	this->_res._pivot = prev_li;
    }

    this->_res._seq._end = res._seq._end;
  }

public:

  PartitionWork() : BaseType() {}

  PartitionWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
    : BaseType(s, c, r) {}

  inline void prepare()
  {
    this->_res = ResultType(this->_seq._beg);
  }

  inline void compute(const SequenceType& seq)
  {
    ResultType res;
    res._seq = seq;
    res._pivot = std::partition(seq._beg, seq._end, *this->_const);
    reduce_result(res);
  }

  inline void reduce(const BaseType& tw)
  {
    reduce_result(tw._res);
  }

};


template<typename BidirectionalIterator>
struct PartitionResult
{
  BasicSequence<BidirectionalIterator> _seq;
  BidirectionalIterator _pivot;

  PartitionResult() {}

  PartitionResult(const BidirectionalIterator& i)
    : _seq(BasicSequence<BidirectionalIterator>(i, i)), _pivot(i)
  {}

};


// tunning params
typedef Daouda0TuningParams PartitionTuningParams;

} // kastl::impl


template
<
  typename BidirectionalIterator,
  typename Predicate,
  typename ParamType
>
BidirectionalIterator partition
(
 BidirectionalIterator begin,
 BidirectionalIterator end,
 Predicate pred
)
{
  typedef kastl::impl::BasicSequence<BidirectionalIterator>
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

  typedef Predicate ConstantType;

  typedef kastl::impl::PartitionResult<BidirectionalIterator> ResultType;

  typedef kastl::impl::PartitionWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end), &pred, ResultType(begin));
  kastl::impl::compute<WorkType>(work);
  return work._res._pivot;
}


template
<
  typename BidirectionalIterator,
  typename Predicate
>
BidirectionalIterator partition
(
 BidirectionalIterator begin,
 BidirectionalIterator end,
 Predicate pred
)
{
  typedef kastl::impl::PartitionTuningParams ParamType;

  return kastl::partition
    <BidirectionalIterator, Predicate, ParamType>
    (begin, end, pred);
}

} // kastl



#endif // ! KASTL_PARTITION_H_INCLUDED
