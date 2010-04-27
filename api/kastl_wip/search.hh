#ifndef KASTL_SEARCH_HH_INCLUDED
# define KASTL_SEARCH_HH_INCLUDED



#include <algorithm>
#include <functional>
#include <iterator>
#include "kastl_impl.hh"


namespace kastl
{

namespace impl
{


template
<
  typename Iterator0Type,
  typename Iterator1Type,
  typename PredicateType
>
struct SearchConstant
{
  typedef typename std::iterator_traits
  <Iterator1Type>::difference_type
  SizeType;

  Iterator0Type _bad_res;
  Iterator1Type _beg;
  Iterator1Type _end;
  PredicateType _pred;

  SearchConstant
  (
   const Iterator1Type& beg,
   const Iterator1Type& end,
   const PredicateType& pred
  ) : _bad_res(end), _beg(beg),
     _end(end), _pred(pred)
  { }

  inline SizeType win_size() const
  { return std::distance(_beg, _end); }
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
class SearchWork : public BaseWork
<SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
{
  typedef SearchWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  SelfType;

  typedef BaseWork
  <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  BaseType;

public:

  SearchWork() : BaseType() {}

  SearchWork(const SequenceType& s, const ConstantType* c, const ResultType& r)
  : BaseType(s, c, r)
  { prepare(); }

  inline void prepare()
  { this->_res = this->_const->_bad_res; }

  inline void compute(SequenceType& seq)
  {
    const typename SequenceType::_IteratorType seq_end =
      seq.end() + this->_const->win_size();

    ResultType result = std::search
    (
     seq.begin(), seq_end,
     this->_const->_beg,
     this->_const->_end,
     this->_const->_pred
    );

    seq.advance();

    if (result == seq_end)
      return ;

    this->_res = result;

    // result found, terminate
    this->_is_done = true;
  }

  inline void reduce(const BaseType& tw)
  {
    // result already found
    if (this->_res != this->_const->_bad_res)
      return ;

    // thief got a result
    if (tw._res != tw._const->_bad_res)
    {
      this->_res = tw._res;
      this->_is_done = true;
    }

    // otherwise continue
  }

};

// tuning params

struct SearchTuningParams : Daouda1TuningParams
{
#if 1
  static const enum TuningTag macro_tag = TAG_LINEAR;
  static const size_t macro_min_size = 1024;
  static const size_t macro_max_size = 32768;
  static const size_t macro_step_size = 2048;
#else
  static const enum TuningTag macro_tag = TAG_STATIC;
  static const size_t macro_min_size = 32768;
  static const size_t macro_max_size = 32768;
#endif
};

} // kastl::impl


template
<
  class Iterator0Type,
  class Iterator1Type,
  class PredicateType,
  class ParamType
>
Iterator0Type search
(
 Iterator0Type beg0,
 Iterator0Type end0,
 Iterator1Type beg1,
 Iterator1Type end1,
 PredicateType pred
)
{
  typedef kastl::impl::InSequence<Iterator0Type>
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

  typedef kastl::impl::SearchConstant
    <Iterator0Type, Iterator1Type, PredicateType>
    ConstantType;

  typedef typename std::iterator_traits
  <Iterator1Type>::difference_type
  SizeType;

  const SizeType win_size = std::distance(beg1, end1);
  if (std::distance(beg0, end0) < win_size)
    return end0;

  ConstantType constant(beg1, end1, pred);

  typedef Iterator0Type ResultType;

  typedef kastl::impl::SearchWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType>
  WorkType;

  SequenceType seq(beg0, end0 - win_size);
  WorkType work(seq, &constant, ResultType(end0));

  kastl::impl::compute(work);

  return work._res;
}


template
<
  class Iterator0Type,
  class Iterator1Type,
  class PredicateType
>
Iterator0Type search
(
 Iterator0Type beg0,
 Iterator0Type end0,
 Iterator1Type beg1,
 Iterator1Type end1,
 PredicateType pred
)
{
  typedef kastl::impl::SearchTuningParams ParamType;

  return kastl::search
    <Iterator0Type, Iterator1Type, PredicateType, ParamType>
    (beg0, end0, beg1, end1, pred);
}


template
<
  class Iterator0Type,
  class Iterator1Type
>
Iterator0Type search
(
 Iterator0Type beg0,
 Iterator0Type end0,
 Iterator1Type beg1,
 Iterator1Type end1
)
{
  typedef typename std::iterator_traits
    <Iterator0Type>::value_type ValueType;

  typedef std::equal_to<ValueType> PredicateType;

  return kastl::search
    <Iterator0Type, Iterator1Type, PredicateType>
    (beg0, end0, beg1, end1, PredicateType());
}

} // kastl


#endif // ! KASTL_SEARCH_HH_INCLUDED
