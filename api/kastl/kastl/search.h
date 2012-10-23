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


#ifndef KASTL_SEARCH_H_INCLUDED
# define KASTL_SEARCH_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator0, typename Iterator1, typename Predicate>
struct search_body
{
  typedef kastl::rts::Sequence<Iterator0> sequence_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::touched_algorithm_result<Iterator0> result_type;

  Iterator1 _first;
  Iterator1 _last;
  Predicate _pred;

  search_body
  (const Iterator1& first, const Iterator1& last, const Predicate& pred)
    : _first(first), _last(last), _pred(pred)
  {}

  bool operator()(result_type& res, const Iterator0& const_pos)
  {
    Iterator0 pos = const_pos;

    // terminate if pred holds for all
    for (Iterator1 first = _first; first != _last; ++first, ++pos)
    {
      if (_pred(*pos, *first) == false)
	return false;
    }

    res.set_iter(const_pos);
    return true;
  }

  bool reduce(result_type& lhs, const result_type& rhs)
  {
    // terminate if touched
    if ((lhs._is_touched == false) && (rhs._is_touched == true))
      lhs.set_iter(rhs._iter);
    return lhs._is_touched == true;
  }
};

template<typename Iterator0, typename Iterator1,
	 typename Predicate, typename Settings>
Iterator0 search
(Iterator0 first0, Iterator0 last0, Iterator1 first1, Iterator1 last1,
 Predicate pred, const Settings& settings)
{
  typedef typename std::iterator_traits<Iterator0>::difference_type size_type;

  const size_type size0 = std::distance(first0, last0);
  const size_type size1 = std::distance(first1, last1);

  if (size0 < size1)
    return last0;

  kastl::rts::Sequence<Iterator0> seq(first0, (size0 - size1) + 1);
  kastl::impl::touched_algorithm_result<Iterator0> res(last0);

  search_body<Iterator0, Iterator1, Predicate> body(first1, last1, pred);
  kastl::impl::while_reduce_loop(res, seq, body, settings);
  return res._iter;
}

template<typename Iterator0, typename Iterator1, typename Predicate>
Iterator0 search
(Iterator0 first0, Iterator0 last0,
 Iterator1 first1, Iterator1 last1,
 Predicate pred)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::search(first0, last0, first1, last1, pred, settings);
}

template<typename Iterator0, typename Iterator1>
Iterator0 search
(Iterator0 first0, Iterator0 last0, Iterator1 first1, Iterator1 last1)
{
  typedef typename std::iterator_traits<Iterator0>::value_type value_type;
  return kastl::search(first0, last0, first1, last1, eq<value_type>());
}

} // kastl::


#endif // ! KASTL_SEARCH_H_INCLUDED
