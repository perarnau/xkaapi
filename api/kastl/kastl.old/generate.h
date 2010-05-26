#ifndef KASTL_GENERATE_H_INCLUDED
# define KASTL_GENERATE_H_INCLUDED



#include <stddef.h>
#include <algorithm>
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
struct GenerateWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef GenerateWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  GenerateWork() : BaseType() {}

  GenerateWork(const SequenceType& s, const ConstantType* c)
    : BaseType(s, c, kastl::impl::InvalidResultType()) {}

  inline void compute(const SequenceType& seq)
  {
    std::generate(seq.begin(), seq.end(), *this->_const);
  }
};

// tunning params
typedef Daouda0TuningParams GenerateTuningParams;

} // kastl::impl


template
<
  typename ForwardIterator,
  typename Generator,
  typename ParamType
>
void generate
(
 ForwardIterator begin,
 ForwardIterator end,
 Generator gen
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

  typedef Generator ConstantType;

  typedef kastl::impl::InvalidResultType ResultType;

  typedef kastl::impl::GenerateWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(begin, end), &gen);
  kastl::impl::compute<WorkType>(work);
}


template
<
 typename ForwardIterator,
 typename Generator
>
void generate
(
 ForwardIterator begin,
 ForwardIterator end,
 Generator gen
)
{
  typedef kastl::impl::GenerateTuningParams ParamType;

  kastl::generate
  <ForwardIterator, Generator, ParamType>
  (begin, end, gen);
}

} // kastl



#endif // ! KASTL_GENERATE_H_INCLUDED
