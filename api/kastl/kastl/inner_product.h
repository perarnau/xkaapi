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


#ifndef KASTL_INNER_PRODUCT_H_INCLUDED
# define KASTL_INNER_PRODUCT_H_INCLUDED


#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{
template<typename Iterator0, typename Iterator1,
	 typename Value,
	 typename Operator0, typename Operator1>
struct inner_product_body
{
  typedef kastl::rts::Sequence<Iterator0, Iterator1> sequence_type;
  typedef typename sequence_type::iterator_type iterator_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::numeric_result<Iterator0, Value> result_type;

  Operator0 _op0;
  Operator1 _op1;

  inner_product_body
  (const Operator0& op0, const Operator1& op1)
    : _op0(op0), _op1(op1)
  {}

  void operator()(result_type& res, Iterator0& ipos0, Iterator1& ipos1)
  {
    res._value = _op0(res._value, _op1(*ipos0, *ipos1));
  }

  void reduce(result_type& lhs, const result_type& rhs)
  {
    lhs._value = _op0(lhs._value, rhs._value);
  }
};

template
<typename Iterator0, typename Iterator1,
 typename Operator0, typename Operator1,
 typename Value, typename Settings>
Value inner_product
(Iterator0 ifirst0, Iterator0 ilast, Iterator1 ifirst1,
 Value init,
 Operator0 op0, Operator1 op1, const Settings& settings)
{
  kastl::rts::Sequence<Iterator0, Iterator1> seq
    (ifirst0, ifirst1, ilast - ifirst0);

  inner_product_body
    <Iterator0, Iterator1, Value, Operator0, Operator1>
    body(op0, op1);

  kastl::impl::numeric_result<Iterator0, Value> res(init);
  kastl::impl::foreach_reduce_loop(res, seq, body, settings);
  return res._value;
}

template<typename Iterator0, typename Iterator1, typename Value,
	 typename Operator0, typename Operator1>
Value inner_product
(Iterator0 ifirst0, Iterator0 ilast, Iterator1 ifirst1,
 Value init,
 Operator0 op0, Operator1 op1)
{
  kastl::impl::static_settings settings(512, 512);

  return kastl::inner_product
    (ifirst0, ilast, ifirst1, init, op0, op1, settings);
}

template<typename Iterator0, typename Iterator1, typename Value>
Value inner_product
(Iterator0 ifirst0, Iterator0 ilast, Iterator1 ifirst1, Value init)
{
  return kastl::inner_product
    (ifirst0, ilast, ifirst1, init,
     kastl::impl::add<Value>(),
     kastl::impl::mul<Value>());
}

} // kastl::


#endif // ! KASTL_INNER_PRODUCT_H_INCLUDED
