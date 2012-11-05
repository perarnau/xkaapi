/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 ** 
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 ** 
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 ** 
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 ** 
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 ** 
 */


#ifndef KASTL_PARTIAL_SUM_H_INCLUDED
# define KASTL_PARTIAL_SUM_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

#if 0 // todo
template<typename Sequence>
struct partial_sum_result
{
  typedef typename Sequence::range_type::iterator1_type iterator1_type;
  typedef typename Sequence::range_type::iterator2_type iterator2_type;
  typedef typename std::iterator_traits<iterator2_type>::value_type value_type;

  iterator2_type _begin;
  Sequence& _seq;
  value_type _value;

  partial_sum_result(Sequence& seq)
    : _begin(seq.begin2()), _seq(seq)
  {
    // sequence size > 0

    _value = *seq.begin1();
    *seq.begin2() = _value;

    typename Sequence::range_type dummy_range;
    seq.pop_safe(dummy_range, 1);
  }
};
#endif // todo

template<typename Iterator0, typename Iterator1, typename Operator>
struct partial_sum_body
{
  typedef kastl::rts::Sequence<Iterator0, Iterator1> sequence_type;
  typedef typename std::iterator_traits<Iterator1>::value_type value_type;
#if 0 // todo
  typedef partial_sum_result<sequence_type> result_type;
#else
  typedef kastl::impl::dummy_type result_type;
#endif

  Operator _op;

  partial_sum_body(const Operator& op)
    : _op(op)
  {}

  void operator()(result_type& sum, const Iterator0& ipos, Iterator1& opos)
  {
#if 0 // todo
    sum._value = _op(sum._value, *ipos);
    *opos = sum._value;
#endif
  }

  void reduce(result_type& lhs, result_type& rhs)
  {
#if 0 // todo

    // _begin was saved during result init
    // begin2() points to the first non processed
    // iterator, thus we apply result on the range
    // [_begin, begin2()[

    Iterator1 pos = rhs._begin;
    Iterator1 end = rhs._seq.begin2();

    for (; pos != end; ++pos)
      *pos = _op(lhs._value, *pos);
    lhs._value = _op(lhs._value, rhs._value);
#endif
  }

};

template<typename Iterator0, typename Iterator1,
	 typename Operator, typename Settings>
Iterator1 partial_sum
(Iterator0 first0, Iterator0 last0,
 Iterator1 first1, Operator op, const Settings& settings)
{
  typedef typename std::iterator_traits<Iterator0>::difference_type size_type;
  typedef typename std::iterator_traits<Iterator1>::value_type value_type;
  typedef kastl::rts::Sequence<Iterator0, Iterator1> sequence_type;
#if 0 // todo
  typedef partial_sum_result<sequence_type> result_type;
#endif

  const size_type size = std::distance(first0, last0);

  if (size == 0)
    return first1;

  sequence_type seq(first0, first1, size);
#if 0 // todo
  result_type res(seq);
#endif

  partial_sum_body<Iterator0, Iterator1, Operator> body(op);
#if 0
  kastl::impl::foreach_reduce_loop(res, seq, body, settings);
#else
  kastl::impl::foreach_loop(seq, body, settings);
#endif

  return first1 + size;
}

template<typename Iterator0, typename Iterator1, typename Operator>
Iterator1 partial_sum
(Iterator0 first0, Iterator0 last0, Iterator1 first1, Operator op)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::partial_sum(first0, last0, first1, op, settings);
}

template<typename Iterator0, typename Iterator1>
Iterator1 partial_sum
(Iterator0 first0, Iterator0 last0, Iterator1 first1)
{
  typedef typename std::iterator_traits<Iterator0>::value_type value_type;
  return kastl::partial_sum
    (first0, last0, first1, kastl::impl::add<value_type>());
}

} // kastl::


#endif // ! KASTL_PARTIAL_SUM_H_INCLUDED
