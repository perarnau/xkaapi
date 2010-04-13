#ifndef KASTL_REPLACE_HH_INCLUDED
# define KASTL_REPLACE_HH_INCLUDED



#include <stddef.h>
#include <algorithm>
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
struct ReplaceWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef ReplaceWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  ReplaceWork() : BaseType() {}

  ReplaceWork(const SequenceType& s, const ConstantType* c)
    : BaseType(s, c, InvalidResultType()) { }

  inline void compute(const SequenceType& seq)
  {
    std::replace
    (
     seq.begin(), seq.end(),
     this->_const->first,
     this->_const->second
    );
  }

};

// tunning params
typedef Daouda0TuningParams ReplaceTuningParams;

} // kastl::impl


template
<
  typename ForwardIterator,
  typename ValueType,
  typename ParamType
>
void replace
(
 ForwardIterator begin,
 ForwardIterator end,
 const ValueType& ref_val,
 const ValueType& new_val
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

  typedef std::pair<ValueType, ValueType>
    ConstantType;

  typedef kastl::impl::InvalidResultType
    ResultType;

  typedef kastl::impl::ReplaceWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  ConstantType constant(ref_val, new_val);
  WorkType work(SequenceType(begin, end), &constant);
  kastl::impl::compute<WorkType>(work);
}


template
<
 typename ForwardIterator,
 typename ValueType
>
void replace
(
 ForwardIterator begin,
 ForwardIterator end,
 const ValueType& ref_val,
 const ValueType& new_val
)
{
  typedef kastl::impl::ReplaceTuningParams ParamType;

  return kastl::replace
    <ForwardIterator, ValueType, ParamType>
    (begin, end, ref_val, new_val);
}

} // kastl



#endif // ! KASTL_REPLACE_HH_INCLUDED
