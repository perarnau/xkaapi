#ifndef KASTL_FIND_IF_H_INCLUDED
# define KASTL_FIND_IF_H_INCLUDED



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
  typename IteratorType,
  typename PredicateType
>
struct FindIfConstant
{
  IteratorType _bad_res;
  PredicateType _pred;

  FindIfConstant(const IteratorType& bad_res, const PredicateType& pred)
  : _bad_res(bad_res), _pred(pred) {}
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
class FindIfWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef FindIfWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  BaseType;

  inline void reduce_result(const ResultType& res)
  {
    this->_res = res;
    this->_is_done = true;
  }

public:

  FindIfWork() : BaseType() {}

  FindIfWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r) { prepare(); }

  inline void prepare()
  { this->_res = this->_const->_bad_res; }

  inline void compute(SequenceType& seq)
  {
    ResultType res = std::find_if(seq.begin(), seq.end(), this->_const->_pred);

    seq.advance();

    if (res == seq.end())
      return ;

    reduce_result(res);
  }

  inline void reduce(const BaseType& tw)
  {
    // result already found
    if (this->_res != this->_const->_bad_res)
      return ;

    // thief got a result
    if (tw._res != tw._const->_bad_res)
      reduce_result(tw._res);
  }

};

// tuning params

struct FindIfTuningParams : Daouda1TuningParams
{
  static const enum TuningTag macro_tag = TAG_LINEAR;
  static const size_t macro_min_size = 1024;
  static const size_t macro_max_size = 32768;
  static const size_t macro_step_size = 1024;
};

} // kastl::impl


template
<
  class InputIterator,
  class PredicateType,
  class ParamType
>
InputIterator find_if
(
 InputIterator beg,
 InputIterator end,
 PredicateType pred
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

  typedef kastl::impl::FindIfConstant<InputIterator, PredicateType> ConstantType;
  ConstantType constant(end, pred);

  typedef InputIterator ResultType;

  typedef kastl::impl::FindIfWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  WorkType work(SequenceType(beg, end), &constant, end);
  kastl::impl::compute(work);
  return work._res;
}


template
<
  class InputIterator,
  class Predicate
>
InputIterator find_if
(
 InputIterator begin,
 InputIterator end,
 Predicate pred
)
{
  typedef kastl::impl::FindTuningParams ParamType;

  return kastl::find_if
    <InputIterator, Predicate, ParamType>
    (begin, end, pred);
}

} // kastl


#endif // ! KASTL_FIND_IF_H_INCLUDED
