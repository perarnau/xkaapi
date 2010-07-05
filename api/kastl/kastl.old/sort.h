#ifndef KASTL_SORT_H_INCLUDED
# define KASTL_SORT_H_INCLUDED



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
class SortWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef SortWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  BaseType;

#if 0 // unused
  static bool is_sorted(unsigned int* pos, unsigned int* end)
  {
    if ((end - pos) <= 1)
      return true;

    for (pos = pos + 1; pos != end; ++pos)
      if (*pos < *(pos - 1))
	return false;

    return true;
  }

  static bool is_sorted(const SequenceType& seq)
  {
    return is_sorted(seq._beg, seq._end);
  }
#endif

  inline void reduce_result(const SequenceType& seq)
  {
    std::inplace_merge
    (
     this->_res._beg,
     seq._beg,
     seq._end,
     *(this->_const)
    );

    this->_res._end = seq._end;
  }

public:

  SortWork() : BaseType()
  { }

  SortWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r)
  { prepare(); }

  inline void prepare()
  {
    this->_res._beg = this->_seq._beg;
    this->_res._end = this->_seq._beg;
  }

  inline void compute(const SequenceType& seq)
  {
    std::sort(seq.begin(), seq.end(), *(this->_const));
    reduce_result(seq);
  }

  inline void reduce(const BaseType& tw)
  {
    // tw the thief work
    reduce_result(tw._res);
  }

};

// tunning params
struct SortTuningParams : Daouda0TuningParams
{
  static const size_t nano_size = 8192;
};


} // kastl::impl


template
<
  class IteratorType,
  class ComparatorType,
  class ParamType
>
void sort(IteratorType pos, IteratorType end, ComparatorType cmp)
{
  typedef kastl::impl::BasicSequence<IteratorType>
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

  typedef ComparatorType ConstantType;
  typedef SequenceType ResultType;

  typedef kastl::impl::SortWork
    <SequenceType, ComparatorType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(pos, end), &cmp, ResultType());

  kastl::impl::compute(work);
}


template<class IteratorType, class ComparatorType>
void sort(IteratorType pos, IteratorType end, ComparatorType cmp)
{
  typedef kastl::impl::SortTuningParams ParamType;

  kastl::sort
    <IteratorType, ComparatorType, ParamType>
    (pos, end, cmp);
}


template<class IteratorType>
void sort(IteratorType pos, IteratorType end)
{
  typedef typename std::iterator_traits
    <IteratorType>::value_type ValueType;

  typedef std::less<ValueType> ComparatorType;

  kastl::sort<IteratorType, ComparatorType>(pos, end, ComparatorType());
}

} // kastl


#endif // ! KASTL_SORT_H_INCLUDED
