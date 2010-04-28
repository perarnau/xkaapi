#ifndef KASTL_FILL_HH_INCLUDED
# define KASTL_FILL_HH_INCLUDED



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
struct FillWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef FillWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  FillWork() : BaseType() {}

  FillWork(const SequenceType& s, const ConstantType* c)
    : BaseType(s, c, ResultType()) {}

  inline void compute(const SequenceType& seq)
  {
    std::fill(seq.begin(), seq.end(), *this->_const);
  }
};

// tunning params
typedef Daouda0TuningParams FillTuningParams;

} // kastl::impl


template
<
  typename ForwardIterator,
  typename ValueType,
  typename ParamType
>
void fill
(
 ForwardIterator begin,
 ForwardIterator end,
 const ValueType& val
)
{
  typedef kastl::impl::BasicSequence<ForwardIterator>
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

  typedef ValueType ConstantType;

  typedef kastl::impl::InvalidResultType ResultType;

  typedef kastl::impl::FillWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end), &val);
  kastl::impl::compute<WorkType>(work);
}


template
<
 typename ForwardIterator,
 typename ValueType
>
void fill
(
 ForwardIterator begin,
 ForwardIterator end,
 const ValueType& val
)
{
  typedef kastl::impl::FillTuningParams ParamType;

  return kastl::fill
    <ForwardIterator, ValueType, ParamType>
    (begin, end, val);
}

} // kastl



#endif // ! KASTL_FILL_HH_INCLUDED
