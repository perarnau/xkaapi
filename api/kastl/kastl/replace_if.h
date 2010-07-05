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


#ifndef KASTL_REPLACE_IF_H_INCLUDED
# define KASTL_REPLACE_IF_H_INCLUDED


#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator, typename Predicate, typename Value>
struct replace_if_body
{
  typedef kastl::impl::dummy_type result_type;

  Predicate _pred;
  const Value& _value;

  replace_if_body(Predicate pred, const Value& value)
    : _pred(pred), _value(value)
  {}

  void operator()(result_type&, Iterator& pos)
  {
    if (_pred(*pos))
      *pos = _value;
  }
};

template
<typename Iterator, typename Predicate, typename Value, typename Settings>
void replace_if
(Iterator first, Iterator last,
 Predicate pred, const Value& value,
 const Settings& settings)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  replace_if_body<Iterator, Predicate, Value> body(pred, value);
  kastl::impl::foreach_loop(seq, body, settings);
}

template<typename Iterator, typename Predicate, typename Value>
void replace_if
(Iterator first, Iterator last, Predicate pred, const Value& value)
{
  kastl::impl::static_settings settings(512, 512);
  kastl::replace_if(first, last, pred, value, settings);
}

} // kastl::


#endif // ! KASTL_REPLACE_IF_H_INCLUDED
