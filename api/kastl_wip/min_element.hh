#ifndef KASTL_MIN_ELEMENT_HH_INCLUDED
# define KASTL_MIN_ELEMENT_HH_INCLUDED



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
struct MinElementWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef MinElementWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

  MinElementWork() : BaseType() {}

  MinElementWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
    : BaseType(s, c, r) { prepare(); }

  inline void prepare()
  {
    this->_res = this->_const->_invalid_res;
  }

  inline void reduce_result(const ResultType& res)
  {
    // assume res is valid
    if (this->_res == this->_const->_invalid_res)
      this->_res = res;
    else if (this->_const->_binary_pred(*res, *this->_res))
      this->_res = res;
  }

  inline void compute(SequenceType& seq)
  {
    reduce_result(std::min_element(seq.begin(), seq.end(), this->_const->_binary_pred));
    seq.advance(seq.size());
  }

  inline void reduce(const SelfType& thief_work)
  {
    if (thief_work._res == thief_work._const->_invalid_res)
      return ;
    reduce_result(thief_work._res);
  }

};


// constant type
template<typename IteratorType, typename BinaryPredicate>
struct MinElementConstant
{
  IteratorType _invalid_res;
  BinaryPredicate _binary_pred;

  MinElementConstant(const IteratorType& invalid_res,
		     const BinaryPredicate& binary_pred)
    : _invalid_res(invalid_res),
      _binary_pred(binary_pred)
  {}

};


// tunning params
typedef Daouda0TuningParams MinElementTuningParams;

} // kastl::impl


template
<
  typename ForwardIterator,
  typename BinaryPredicate,
  typename ParamType
>
ForwardIterator min_element
(
 ForwardIterator begin,
 ForwardIterator end,
 BinaryPredicate pred
)
{
  typedef kastl::impl::InSequence<ForwardIterator>
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

  typedef kastl::impl::MinElementConstant<ForwardIterator, BinaryPredicate>
    ConstantType;

  typedef ForwardIterator ResultType;

  typedef kastl::impl::MinElementWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
    WorkType;

  ConstantType constant(end, pred);
  WorkType work(SequenceType(begin, end), &constant, end);
  kastl::impl::compute<WorkType>(work);
  return work._res;
}


template
<
  typename ForwardIterator,
  typename BinaryPredicate
>
ForwardIterator min_element
(
 ForwardIterator begin,
 ForwardIterator end,
 BinaryPredicate pred
)
{
  typedef kastl::impl::MinElementTuningParams ParamType;
  return kastl::min_element
    <ForwardIterator, BinaryPredicate, ParamType>
    (begin, end, pred);
}


template<typename ForwardIterator>
ForwardIterator min_element
(
 ForwardIterator begin,
 ForwardIterator end
)
{
  typedef typename std::iterator_traits
    <ForwardIterator>::value_type ValueType;
  return kastl::min_element(begin, end, std::less<ValueType>());
}

} // kastl



#endif // ! KASTL_MIN_ELEMENT_HH_INCLUDED
