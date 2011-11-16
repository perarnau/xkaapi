/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
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


#ifndef KASTL_MISMATCH_H_INCLUDED
# define KASTL_MISMATCH_H_INCLUDED


#include <utility>
#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator0, typename Iterator1>
struct mismatch_result
{
  typedef std::pair<Iterator0, Iterator1> pair_type;

  bool _is_touched;
  pair_type _iters;

  mismatch_result(const Iterator0&)
    : _is_touched(false)
  {}

  mismatch_result(const Iterator0& iter0, const Iterator1& iter1)
    : _is_touched(false), _iters(pair_type(iter0, iter1))
  {}

  void set_iters(const Iterator0& iter0, const Iterator1& iter1)
  {
    _is_touched = true;
    _iters = pair_type(iter0, iter1);
  }
};

template<typename Iterator0, typename Iterator1, typename Predicate>
struct mismatch_body
{
  typedef kastl::rts::Sequence<Iterator0, Iterator1> sequence_type;
  typedef typename sequence_type::range_type range_type;
  typedef mismatch_result<Iterator0, Iterator1> result_type;

  Predicate _pred;

  mismatch_body(const Predicate& pred)
    : _pred(pred)
  {}

  bool operator()
  (result_type& res, const Iterator0& pos0, const Iterator1& pos1)
  {
    // terminate if pred is false
    if (_pred(*pos0, *pos1) == true)
      return false;
    res.set_iters(pos0, pos1);
    return true;
  }

  bool reduce(result_type& lhs, const result_type& rhs)
  {
    // terminate if touched
    if ((lhs._is_touched == false) && (rhs._is_touched == true))
      lhs.set_iters(rhs._iters.first, rhs._iters.second);
    return lhs._is_touched == true;
  }
};

template<typename Iterator0, typename Iterator1,
	 typename Predicate, typename Settings>
std::pair<Iterator0, Iterator1> mismatch
(Iterator0 first0, Iterator0 last, Iterator1 first1,
 Predicate pred, const Settings& settings)
{
  typedef typename std::iterator_traits<Iterator0>::difference_type size_type;

  const size_type size = std::distance(first0, last);

  kastl::rts::Sequence<Iterator0, Iterator1> seq
    (first0, first1, size);

  mismatch_result<Iterator0, Iterator1> res(last, first1 + size);

  mismatch_body<Iterator0, Iterator1, Predicate> body(pred);
  kastl::impl::while_reduce_loop(res, seq, body, settings);
  return res._iters;
}

template<typename Iterator0, typename Iterator1, typename Predicate>
std::pair<Iterator0, Iterator1> mismatch
(Iterator0 first0, Iterator0 last, Iterator1 first1, Predicate pred)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::mismatch(first0, last, first1, pred, settings);
}

template<typename Iterator0, typename Iterator1>
std::pair<Iterator0, Iterator1> mismatch
(Iterator0 first0, Iterator0 last, Iterator1 first1)
{
  typedef typename std::iterator_traits<Iterator0>::value_type value_type;
  return kastl::mismatch(first0, last, first1, eq<value_type>());
}

} // kastl::


#endif // ! KASTL_MISMATCH_H_INCLUDED
