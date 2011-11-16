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


#ifndef KASTL_FIND_FIRST_OF_H_INCLUDED
# define KASTL_FIND_FIRST_OF_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"
#include "find_if.h"


namespace kastl
{

template<typename Value>
struct eq
{
  bool operator()(const Value& lhs, const Value& rhs)
  {
    return lhs == rhs;
  }
};

template<typename Iterator, typename Predicate>
struct ffo_predicate
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  Iterator _first;
  Iterator _last;

  Predicate _pred;

  ffo_predicate(const Iterator& first, const Iterator& last, const Predicate& pred)
    : _first(first), _last(last), _pred(pred)
  {}

  bool operator()(const value_type& value)
  {
    for (Iterator pos = _first; pos != _last; ++pos)
      if (_pred(*pos, value))
	return true;
    return false;
  }
};

template<typename Iterator0, typename Iterator1, typename Predicate>
Iterator0 find_first_of
(Iterator0 first0, Iterator0 last0,
 Iterator1 first1, Iterator1 last1,
 Predicate pred)
{
  ffo_predicate<Iterator1, Predicate> ffo_pred(first1, last1, pred);
  // call kastl::find_if
  return kastl::find_if(first0, last0, ffo_pred);
}

template<typename Iterator0, typename Iterator1>
Iterator0 find_first_of
(Iterator0 first0, Iterator0 last0,
 Iterator1 first1, Iterator1 last1)
{
  typedef typename std::iterator_traits<Iterator0>::value_type value_type;
  return kastl::find_first_of
    (first0, last0, first1, last1, kastl::eq<value_type>());
}

} // kastl::



#endif // KASTL_FIND_FIRST_OF_H_INCLUDED
