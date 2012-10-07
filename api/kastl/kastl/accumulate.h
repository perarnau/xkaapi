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


#ifndef KASTL_ACCUMULATE_H_INCLUDED
# define KASTL_ACCUMULATE_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{
template<typename Iterator, typename Value, typename Function>
struct accumulate_body
{
  typedef kastl::impl::numeric_result<Iterator, Value> result_type;

  Function _op;

  accumulate_body(const Function& op)
    : _op(op)
  {}

  void operator()(result_type& res, const Iterator& pos)
  {
    res._value = _op(res._value, *pos);
  }

  void reduce(result_type& lhs, const result_type& rhs)
  {
    lhs._value = _op(lhs._value, rhs._value);
  }
};

template<typename Iterator, typename Value, typename Function, typename Settings>
Value accumulate(Iterator first, Iterator last, Value init, Function op, const Settings& settings)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  accumulate_body<Iterator, Value, Function> body(op);
  kastl::impl::numeric_result<Iterator, Value> result(init);
  kastl::impl::foreach_reduce_loop(result, seq, body, settings);
  return result._value;
}

template<typename Iterator, typename Value, typename Function>
Value accumulate(Iterator first, Iterator last, Value init, Function op)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::accumulate(first, last, init, op, settings);
}

template<typename Iterator, typename Value>
Value accumulate(Iterator first, Iterator last, Value init)
{
  return kastl::accumulate(first, last, init, kastl::impl::add<Value>());
}

} // kastl::


#endif // ! KASTL_ACCUMULATE_H_INCLUDED
