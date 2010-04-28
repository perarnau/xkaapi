#ifndef KASTL_COPY_HH_INCLUDED
# define KASTL_COPY_HH_INCLUDED



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
struct CopyWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef CopyWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  CopyWork() : BaseType() {}

  CopyWork(const SequenceType& s, const ResultType& r)
    : BaseType(s, &r, r) {}

  inline void prepare()
  {
    this->_res = *this->_const;
  }

  inline void compute(SequenceType& seq)
  {
    this->_res = std::copy(seq._iseq._beg, seq._iseq._end, seq._opos);
    seq._iseq._beg = seq._iseq._end;
    seq._opos = this->_res;
  }

  inline void reduce(const BaseType& tw)
  {
    this->_res = tw._res;
  }
};

// tunning params
typedef Daouda0TuningParams CopyTuningParams;

} // kastl::impl



template
<
  class InputIterator,
  class OutputIterator,
  class ParamType
>
OutputIterator copy
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos
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

  typedef OutputIterator ConstantType;

  typedef OutputIterator ResultType;

  typedef kastl::impl::CopyWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(ipos, iend, opos), opos);
  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
 class InputIterator,
 class OutputIterator
>
OutputIterator copy
(
 InputIterator ipos,
 InputIterator iend,
 OutputIterator opos
)
{
  typedef kastl::impl::CopyTuningParams ParamType;
  return kastl::copy<InputIterator, OutputIterator, ParamType>
    (ipos, iend, opos);
}

} // kastl



#endif // ! KASTL_COPY_HH_INCLUDED
