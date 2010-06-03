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


#ifndef KASTL_ADJACENT_DIFFERENCE_H_INCLUDED
# define KASTL_ADJACENT_DIFFERENCE_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator0, typename Iterator1, typename Operator>
struct adjacent_difference_body
{
  typedef kastl::rts::Sequence<Iterator0, Iterator1> sequence_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::dummy_type result_type;

  Operator _op;

  adjacent_difference_body(const Operator& op)
    : _op(op)
  {}

  bool operator()
  (result_type&, const Iterator0& ipos, Iterator1& opos)
  {
    *opos = _op(*ipos, *(ipos - 1));
    return false;
  }
};

template<typename Iterator0, typename Iterator1,
	 typename Operator, typename Settings>
Iterator1 adjacent_difference
(Iterator0 first0, Iterator0 last0,
 Iterator1 first1, Operator op, const Settings& settings)
{
  typedef typename std::iterator_traits<Iterator0>::difference_type size_type;
  const size_type size = std::distance(first0, last0);

  if (size == 0)
    return first1;

  *first1 = *first0;

  if (size > 1)
  {
    kastl::rts::Sequence<Iterator0, Iterator1> seq(first0, first1, size);
    adjacent_difference_body<Iterator0, Iterator1, Operator> body(op);
    kastl::impl::parallel_loop::run(seq, body, settings);
  }

  return first1 + size;
}

template<typename Iterator0, typename Iterator1, typename Operator>
Iterator1 adjacent_difference
(Iterator0 first0, Iterator0 last0, Iterator1 first1, Operator op)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::adjacent_difference(first0, last0, first1, op, settings);
}

template<typename Iterator0, typename Iterator1>
Iterator1 adjacent_difference
(Iterator0 first0, Iterator0 last0, Iterator1 first1)
{
  typedef typename std::iterator_traits<Iterator0>::value_type value_type;
  return kastl::adjacent_difference
    (first0, last0, first1, kastl::impl::sub<value_type>());
}

} // kastl::


#endif // ! KASTL_ADJACENT_DIFFERENCE_H_INCLUDED
