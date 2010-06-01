/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
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


#ifndef KASTL_MIN_ELEMENT_H_INCLUDED
# define KASTL_MIN_ELEMENT_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator, typename Predicate>
struct min_element_body
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  typedef Iterator result_type;

  Predicate _pred;

  min_element_body(const Predicate& pred)
    : _pred(pred)
  {}

  bool operator()(result_type& res, const Iterator& pos)
  {
    // result is already touched

    if (!_pred(*res, *pos))
      res = pos;

    return false;
  }

  bool reduce(result_type& lhs, const result_type& rhs)
  {
    if (!_pred(*lhs, *rhs))
      lhs = rhs;

    return false;
  }

};

template<typename Value>
struct lt
{
  bool operator()(const Value& lhs, const Value& rhs)
  {
    return lhs < rhs;
  }
};

template<typename Iterator, typename Predicate, typename Settings>
Iterator min_element
(Iterator first, Iterator last, Predicate pred, const Settings& settings)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  min_element_body<Iterator, Predicate> body(pred);
  Iterator res(first);
  kastl::impl::reduce_unrolled_loop::run(res, seq, body, settings);
  return res;
}

template<typename Iterator, typename Predicate>
Iterator min_element
(Iterator first, Iterator last, Predicate pred)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::min_element(first, last, pred, settings);
}

template<typename Iterator>
Iterator min_element(Iterator first, Iterator last)
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  return kastl::min_element(first, last, kastl::lt<value_type>());
}

} // kastl::



#endif // KASTL_MIN_ELEMENT_H_INCLUDED
