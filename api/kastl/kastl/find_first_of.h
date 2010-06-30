#ifndef KASTL_FIND_FIRST_OF_H_INCLUDED
# define KASTL_FIND_FIRST_OF_H_INCLUDED


#include <iterator>
#include <algorithm>
#include <functional>
#include "kastl_impl.h"
#include "find_if.h"


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
  struct isInRangePredicate
  {
    typedef typename std::iterator_traits
    <Iterator0Type>::value_type Value0Type;

    typedef typename std::iterator_traits
    <Iterator1Type>::value_type Value1Type;

    Iterator1Type _beg;
    Iterator1Type _end;

    PredicateType _pred;

    isInRangePredicate
    (
     const Iterator1Type& beg,
     const Iterator1Type& end,
     PredicateType pred
    )
      : _beg(beg), _end(end), _pred(pred)
    {}

    bool operator()(const Value0Type& v)
    {
      Iterator1Type pos = _beg;

      for (; (pos != _end) && !_pred(*pos, v); ++pos)
	;

      return !(pos == _end);
    }
  };
} // kastl::impl


template
<
  class ForwardIterator1,
  class ForwardIterator2,
  class BinaryPredicate,
  class ParamType
>
ForwardIterator1 find_first_of
(
 ForwardIterator1 pos1,
 ForwardIterator1 end1,
 ForwardIterator2 pos2,
 ForwardIterator2 end2,
 BinaryPredicate pred
)
{
  kastl::impl::isInRangePredicate
    <ForwardIterator1,
    ForwardIterator2,
    BinaryPredicate>
    range_pred(pos2, end2, pred);

  return kastl::find_if(pos1, end1, range_pred);
}


template
<
  class ForwardIterator1,
  class ForwardIterator2,
  class BinaryPredicate
>
ForwardIterator1 find_first_of
(
 ForwardIterator1 pos1,
 ForwardIterator1 end1,
 ForwardIterator2 pos2,
 ForwardIterator2 end2,
 BinaryPredicate pred
)
{
  typedef kastl::impl::FindIfTuningParams ParamType;

  return kastl::find_first_of
    <ForwardIterator1, ForwardIterator2, BinaryPredicate, ParamType>
    (pos1, end1, pos2, end2, pred);
}


template
<
  class ForwardIterator1,
  class ForwardIterator2
>
ForwardIterator1 find_first_of
(
 ForwardIterator1 pos1,
 ForwardIterator1 end1,
 ForwardIterator2 pos2,
 ForwardIterator2 end2
)
{
  typedef typename std::iterator_traits
    <ForwardIterator1>::value_type Value1Type;

  typedef typename std::iterator_traits
    <ForwardIterator2>::value_type Value2Type;

  return kastl::find_first_of
  (
   pos1, end1, pos2, end2,
   &kastl::impl::isEqualBinaryPredicate<Value1Type, Value2Type>
  );
}

} // kastl


#endif // ! KASTL_FIND_FIRST_OF_H_INCLUDED
