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


#ifndef KASTL_TRANSFORM_H_INCLUDED
# define KASTL_TRANSFORM_H_INCLUDED


#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator0, typename Iterator1, typename Operation>
struct transform_body
{
  typedef kastl::rts::Sequence<Iterator0, Iterator1> sequence_type;
  typedef typename sequence_type::iterator_type iterator_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::dummy_type result_type;

  Operation _op;

  transform_body(const Operation& op)
    : _op(op)
  {}

  bool operator()(result_type&, Iterator0& ipos, Iterator1& opos)
  {
    *opos = _op(*ipos);
    return false;
  }
};

// second version body
template<typename Iterator0, typename Iterator1,
	 typename Iterator2, typename Operation>
struct transform2_body
{
  typedef kastl::rts::Sequence<Iterator0, Iterator1, Iterator2> sequence_type;
  typedef typename sequence_type::iterator_type iterator_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::dummy_type result_type;

  Operation _op;

  transform2_body(const Operation& op)
    : _op(op)
  {}

  bool operator()
  (result_type&, Iterator0& ipos0, Iterator1& ipos1, Iterator2& opos)
  {
    *opos = _op(*ipos0, *ipos1);
    return false;
  }
};

// first version
template<typename Iterator0, typename Iterator1,
	 typename Operation, typename Settings>
Iterator1 transform
(Iterator0 ifirst, Iterator0 ilast,
 Iterator1 ofirst, Operation op,
 const Settings& settings)
{
  kastl::rts::Sequence<Iterator0, Iterator1> seq
    (ifirst, ofirst, ilast - ifirst);

  transform_body<Iterator0, Iterator1, Operation> body(op);
  kastl::impl::parallel_loop::run(seq, body, settings);
  return ofirst + (ilast - ifirst);
}

template<typename Iterator0, typename Iterator1, typename Operation>
Iterator1 transform
(Iterator0 ifirst, Iterator0 ilast, Iterator1 ofirst, Operation op)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::transform(ifirst, ilast, ofirst, op, settings);
}

// second version
template<typename Iterator0, typename Iterator1,
	 typename Iterator2, typename Operation,
	 typename Settings>
Iterator2 transform
(Iterator0 ifirst0, Iterator0 ilast,
 Iterator1 ifirst1, Iterator2 ofirst,
 Operation op, const Settings& settings)
{
  kastl::rts::Sequence<Iterator0, Iterator1, Iterator2> seq
    (ifirst0, ifirst1, ofirst, ilast - ifirst0);

  transform2_body<Iterator0, Iterator1, Iterator2, Operation> body(op);
  kastl::impl::parallel_loop::run(seq, body, settings);
  return ofirst + (ilast - ifirst0);
}

#if 0 // unused due to proto conflict
template<typename Iterator0, typename Iterator1,
	 typename Iterator2, typename Operation>
Iterator2 transform
(Iterator0 ifirst0, Iterator0 ilast,
 Iterator1 ifirst1, Iterator2 ofirst,
 Operation op)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::transform
    (ifirst0, ilast, ifirst1, ofirst, op, settings);
}
#endif // unused due to proto conflict

} // kastl::


#endif // ! KASTL_TRANSFORM_H_INCLUDED
