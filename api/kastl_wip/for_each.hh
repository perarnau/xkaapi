#ifndef KASTL_FOR_EACH_HH_INCLUDED
# define KASTL_FOR_EACH_HH_INCLUDED



#include <stddef.h>
#include <algorithm>
#include <iterator>
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
struct ForEachWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef ForEachWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  ForEachWork() : BaseType() {}

  ForEachWork(const SequenceType& s, const ConstantType* c)
    : BaseType(s, c, kastl::impl::InvalidResultType()) {}

  inline void compute(SequenceType& seq)
  {
    std::for_each(seq._seq._beg, seq._seq._end, *this->_const);
    seq._seq._beg = seq._seq._end;
  }
};

// tunning params
typedef Daouda0TuningParams ForEachTuningParams;

} // kastl::impl


template
<
  typename InputIterator,
  typename Function,
  typename ParamType
>
void for_each
(
 InputIterator begin,
 InputIterator end,
 Function func
)
{
  typedef kastl::impl::InSequence<InputIterator>
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

  typedef Function ConstantType;

  typedef kastl::impl::InvalidResultType ResultType;

  typedef kastl::impl::ForEachWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end), &func);
  kastl::impl::compute<WorkType>(work);
}


template
<
 typename InputIterator,
 typename Function
>
void for_each
(
 InputIterator begin,
 InputIterator end,
 Function func
)
{
  // real return type is Function but
  // not possible for us so we voidify

  typedef kastl::impl::ForEachTuningParams ParamType;

  kastl::for_each
  <InputIterator, Function, ParamType>
  (begin, end, func);
}

} // kastl



#endif // ! KASTL_FOR_EACH_HH_INCLUDED
