#ifndef KASTL_REVERSE_HH_INCLUDED
# define KASTL_REVERSE_HH_INCLUDED



#include <stddef.h>
#include <algorithm>
#include <iterator>
#include <functional>
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
class ReverseWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef ReverseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

public:

  ReverseWork() : BaseType() {}

  ReverseWork(const SequenceType& s)
  : BaseType(s, NULL, kastl::impl::InvalidResultType())
  { }

  inline void compute(const SequenceType& seq)
  {
    typename SequenceType::_IteratorType li = seq._lseq._beg;
    typename SequenceType::_IteratorType ri = seq._rseq._end - 1;

    for (; li != seq._lseq._end; ++li, --ri)
      std::swap(*li, *ri);
  }

};


template<typename IteratorType>
struct ReverseSequence : public LockableSequence

{
  // [lseq.beg - lseq.end[ + ... + [rseq.beg - rseq.end[

  typedef ReverseSequence<IteratorType> SequenceType;
  typedef IteratorType _IteratorType;
  typedef typename std::iterator_traits
  <IteratorType>::difference_type SizeType;

  BasicSequence<IteratorType> _lseq;
  BasicSequence<IteratorType> _rseq;

  inline ReverseSequence() {}

  inline ReverseSequence(const IteratorType& beg, const IteratorType& end)
  {
    const SizeType midsize = std::distance(beg, end) / 2;
    _lseq = BasicSequence<IteratorType>(beg, beg + midsize);
    _rseq = BasicSequence<IteratorType>(end - midsize, end);
  }

  inline SizeType size() const
  {
    return _lseq.size();
  }

  inline void split(SequenceType& seq, SizeType size)
  {
    _lseq.split(seq._lseq, size);
    _rseq.rsplit(seq._rseq, size);
  }

  inline void rsplit(SequenceType& seq, SizeType size)
  {
    _lseq.rsplit(seq._lseq, size);
    _rseq.split(seq._rseq, size);
  }

  inline void empty_seq(SequenceType& seq) const
  {
    _lseq.empty_seq(seq._lseq);
    _rseq.empty_seq(seq._rseq);
  }

  inline bool is_empty() const
  {
    return _lseq.is_empty();
  }

#if KASTL_DEBUG
  typedef typename BasicSequence<IteratorType>::RangeType RangeType;
  static RangeType get_range(const SequenceType& a, const SequenceType& b)
  {
    return BasicSequence<IteratorType>::get_range(a._lseq, b._lseq);
  }
#endif
  
};


// tunning params
typedef Daouda0TuningParams ReverseTuningParams;

} // kastl::impl


template
<
  typename BidirectionalIterator,
  typename ParamType
>
void reverse
(
 BidirectionalIterator begin,
 BidirectionalIterator end
)
{
  typedef kastl::impl::ReverseSequence<BidirectionalIterator>
    SequenceType;

  static_assert(ParamType::macro_tag == kastl::impl::TuningTag::TAG_IDENTITY, "invalid macro tag");

  typedef typename kastl::impl::make_macro_type
    <ParamType::macro_tag, ParamType, SequenceType>::Type
    MacroType;

  typedef typename kastl::impl::make_nano_type
    <ParamType::nano_tag, ParamType, SequenceType>::Type
    NanoType;

  typedef typename kastl::impl::make_splitter_type
    <ParamType::splitter_tag, ParamType>::Type
    SplitterType;

  typedef kastl::impl::InvalidConstantType ConstantType;

  typedef kastl::impl::InvalidResultType ResultType;

  typedef kastl::impl::ReverseWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end));

  kastl::impl::compute<WorkType>(work);
}


template<typename BidirectionalIterator>
void reverse
(
 BidirectionalIterator begin,
 BidirectionalIterator end
)
{
  typedef kastl::impl::ReverseTuningParams ParamType;

  return kastl::reverse
    <BidirectionalIterator, ParamType>
    (begin, end);
}

} // kastl



#endif // ! KASTL_REVERSE_HH_INCLUDED
