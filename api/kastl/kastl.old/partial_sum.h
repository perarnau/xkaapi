#ifndef KASTL_PARTIAL_SUM_H_INCLUDED
# define KASTL_PARTIAL_SUM_H_INCLUDED



#include <algorithm>
#include <functional>
#include <numeric>
#include <iterator>
#include "kastl_impl.h"


namespace kastl
{

namespace impl
{
// partial sum result

template<typename OutputIteratorType>
struct PartialSumResult
{
  typedef typename std::iterator_traits
  <OutputIteratorType>::value_type ValueType;

  bool _has_sum;
  OutputIteratorType _obeg;
  OutputIteratorType _oend;
  ValueType _sum;

  PartialSumResult(const OutputIteratorType& obeg)
    : _has_sum(false), _obeg(obeg), _oend(obeg) {}

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
class PartialSumWork : public BaseWork
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

public:

  PartialSumWork() : BaseType() {}

  PartialSumWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r)
  {
    prepare();
  }

  inline void prepare()
  {
    this->_res._has_sum = false;
    this->_res._obeg = this->_seq._opos;
    this->_res._oend = this->_seq._opos;
  }

  inline void compute(SequenceType& seq)
  {
    // do the partial sum ourselves
    // to keep track of the sum

    typedef typename std::iterator_traits
      <OutputIteratorType>::value_type ValueType;

    InputIteratorType ipos = seq._iseq._beg;
    OutputIteratorType opos = seq._opos;
    OutputIteratorType oend = this->_res._oend;

#if KASTL_DEBUG
    typedef typename std::iterator_traits
      <OutputIteratorType>::difference_type OSizeType;
    const OSizeType oend_index = std::distance(this->_ori_seq._opos, oend);

    typedef typename std::iterator_traits
      <InputIteratorType>::difference_type ISizeType;
    const ISizeType ipos_index = std::distance(this->_ori_seq._iseq._beg, ipos);

    if (oend_index != ipos_index)
      printf(" [!] %c indices: %lu, %lu\n",
	     this->_is_master ? 'm' : 's',
	     oend_index, ipos_index);
#endif

    ValueType sum;

    if (this->_res._has_sum == false)
    {
      this->_res._has_sum = true;
      this->_res._sum = *ipos;
      *opos++ = *ipos++;
      ++oend;
    }

    sum = this->_res._sum;

    for (; ipos != seq._iseq._end; ++opos, ++ipos, ++oend)
    {
      sum = (*this->_const)(sum, *ipos);
      *opos = sum;
    }

    this->_res._oend = oend;
    this->_res._sum = sum;

    seq._iseq._beg = ipos;
  }

  inline void reduce(const BaseType& tw)
  {
    // nothing computed
    if (tw._res._obeg == tw._res._oend)
      return ;

    // propagate sum on thief results
    InputIteratorType pos;
    for (pos = tw._res._obeg; pos != tw._res._oend; ++pos)
      *pos = (*this->_const)(this->_res._sum, *pos);

    if (tw._res._has_sum == true)
    {
      if (this->_res._has_sum == true)
	this->_res._sum = (*this->_const)(this->_res._sum, tw._res._sum);
      else
	this->_res._sum = tw._res._sum;
      this->_res._has_sum = true;
    }

    this->_res._oend = tw._res._oend;
  }

};

// tuning params

#if 1
struct PartialSumTuningParams : Daouda1TuningParams
{
  static const enum TuningTag macro_tag = TAG_BACKOFF;
  static const size_t macro_min_size = 8192;
  static const size_t macro_max_size = 32768;
  static const size_t nano_size = 2048;
};
#else
# warning TESTING
struct PartialSumTuningParams : Daouda0TuningParams
{
  static const enum TuningTag macro_tag = TAG_IDENTITY;
  static const size_t macro_max_size = 8192;
  static const enum TuningTag nano_tag = TAG_STATIC;
  static const size_t nano_size = 512;
};
#endif


} // kastl::impl


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
  WorkType work(seq, &op, ResultType(opos));
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


#if 0 // KASTL_DEBUG

struct checked_plus
{
  unsigned int operator()(const unsigned int& a, const unsigned int& b) const
  {
    if ((a + b) < a)
      printf("[!] overflow\n");
    return a + b;
  }
};

#endif


template
<
  class InputIterator,
  class OutputIterator
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


#endif // ! KASTL_PARTIAL_SUM_H_INCLUDED
