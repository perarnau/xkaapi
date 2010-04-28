#ifndef KASTL_TRANSFORM_HH_INCLUDED
# define KASTL_TRANSFORM_HH_INCLUDED



#include <stddef.h>
#include <algorithm>
#include <iterator>
#include "kastl/kastl_impl.hh"



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
struct TransformWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef TransformWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  TransformWork() : BaseType() {}

  TransformWork(const SequenceType& s, const ConstantType* c)
    : BaseType(s, c, kastl::impl::InvalidResultType()) {}

  inline void compute(SequenceType& seq)
  {
    std::transform(seq.input_begin(), seq.input_end(), seq.opos(), *this->_const);
    seq.advance();
  }
};

// tunning params
typedef Daouda0TuningParams TransformTuningParams;

} // kastl::impl



template
<
  class InputIterator,
  class OutputIterator,
  class UnaryOperator,
  class ParamType
>
void transform
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos,
 UnaryOperator op
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

  typedef UnaryOperator ConstantType;

  typedef kastl::impl::InvalidResultType ResultType;

  typedef kastl::impl::TransformWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(ipos, iend, opos), &op);
  kastl::impl::compute<WorkType>(work);
}


template
<
 class InputIterator,
 class OutputIterator,
 class UnaryOperator
>
void transform
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos,
 UnaryOperator op
)
{
  typedef kastl::impl::TransformTuningParams ParamType;

  return kastl::transform
    <InputIterator, OutputIterator, UnaryOperator, ParamType>
    (ipos, iend, opos, op);
}

} // kastl



#endif // ! KASTL_TRANSFORM_HH_INCLUDED
