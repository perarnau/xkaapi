#include <algorithm>
#include <iterator>
#include "kastl_impl.hh"



namespace kastl
{
namespace impl
{

// partial_sum

template<typename Value, typename OutputIterator>
struct PartialSumResult
{
  bool _has_value;
  Value _value;
  OutputIterator _opos;

  PartialSumResult(const OutputIterator& opos)
    : _has_value(false), _opos(opos) {}
};


template
<
  typename SequenceType,
  typename ConstantType,
  typename ResultType,
  typename MacroType,
  typename NanoType,
  typename SplitterType
>
struct PartialSumWork : BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef PartialSumWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  BaseType;

  typedef typename SequenceType::_InputIteratorType InputIteratorType;
  typedef typename SequenceType::_OutputIteratorType OutputIteratorType;

  PartialSumWork() {}

  void prepare(const WorkType& w, const SequenceType& s)
  {
    // prepare a thief task

    w._res._has_value = false;
    w._res._opos = s._opos;
  }

  void compute(SequenceType& seq)
  {
    typedef typename std::iterator_traits
      <OutputIteratorType>::value_type ValueType;

    InputIteratorType ipos = seq._iseq._beg;
    InputIteratorType iend = seq._iseq._end;
    OutputIteratorType opos = seq._opos;

    ValueType value;

    if (this->_res._has_value == false)
    {
      this->_res._has_value = true;
      this->_res._value = *ipos;
      *opos++ = *ipos++;
    }

    value = this->_res._value;

    for (; ipos != iend; ++ipos, ++opos)
    {
      value = (*this->_const)(value, *ipos);
      *opos = value;
    }

    this->_res._value = value;

    seq._iseq._beg = ipos;
    seq._opos = opos;
  }

  static inline bool has_reducer()
  {
    return true;
  }

  void reduce(SequenceType& seq)
  {
    // reduce the sequence

    if (seq._beg == seq._end)
      return ;

    IteratorType pos = seq._iseq._beg;

    const Value old_value = this->_const->_op(this->_res._value, *pos);
    Value new_value = old_value;

    for (; pos != seq._end; ++pos)
    {
      new_value = this->_const->_op(old_value, *pos);
      *pos = new_value;
    }

    this->_res._value = new_value;
  }

  void preempt(WorkType& twork, SequenceType& tseq)
  {
    this->_res._value = this->_const->_op(this->_res._value, twork._res._value);
  }
};


static void preempt_thief(vwork, twork, tseq, vseq)
{
  // for a finalization task if needed
  // and continue the unprocessed work

  if (vwork->has_reducer())
    fork_reducer_task(vwork->_res, tseq.processed_seq());

  WorkType::preempt(vwork, twork);

  vseq = tseq.remaining_seq();
}


static void thief_task(const SequenceType& tseq)
{
  // tseq the thief sequence

  while (1)
  {
    kaapi_stealpoint(tseq);

    if (nanoloop.next(tseq, nseq) == false)
      return ;

    if (kaapi_preemptpoint(preempt_thief, tseq))
      return ;
  }
}


static void main_task(const SequenceType& seq)
{
  // mseq the macro sequence
  SequenceType mseq;

  // process macro loop
  while (macroloop.next(seq, mseq))
  {
    thief_task(work, mseq);
  }

  // macroloop done, reduce thieves
  while (kaapi_preemptthief(reducer, mseq, tseq, preempter))
  {
    if (WorkType::has_reducer == false)
      continue ;
      
    fork_reduce_task();

    reduce_task(mseq, rseq);
  }
}

// partial_sum stl entry

struct PartialSumTuningParams : Daouda1TuningParams
{
  static const enum TuningTag macro_tag = TAG_BACKOFF;
  static const size_t macro_min_size = 8192;
  static const size_t macro_max_size = 65536;
  static const size_t nano_size = 2048;
};

} // impl


template
<
  class InputIterator,
  class OutputIterator,
  class BinOp,
  class ParamType
>
OutputIterator partial_sum
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos,
 BinOp op
)
{
  typedef kastl::impl::InOutSequence<InputIterator, OutputIterator>
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

  typedef BinOp ConstantType;

  typedef kastl::impl::PartialSumResult<OutputIterator> ResultType;

  typedef kastl::impl::PartialSumWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  SequenceType seq(ipos, iend, opos);
  WorkType work(seq, &op, ResultType(res));
  kastl::impl::compute(work);
  return work._res._oend;
}


template
<
  class InputIterator,
  class OutputIterator,
  class BinOp
>
OutputIterator partial_sum
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos,
 BinOp op
)
{
  typedef kastl::impl::PartialSumTuningParams ParamType;

  return kastl::partial_sum
    <InputIterator, OutputIterator, BinOp, ParamType>
    (ipos, iend, opos, op);
}


template
<
  typename InputIterator,
  typename OutputIterator
>
OutputIterator partial_sum
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos
)
{
  typedef typename std::iterator_traits
    <OutputIterator>::value_type ValueType;

  typedef std::plus<ValueType> BinOp;

  return kastl::partial_sum
    <InputIterator, OutputIterator, BinOp>
    (ipos, iend, opos, BinOp());
}

} // kastl
