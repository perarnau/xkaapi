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
#ifndef KASTL_FOR_EACH_H_INCLUDED
#define KASTL_FOR_EACH_H_INCLUDED

#include "kastl/kastl_impl.h"

namespace kastl {

namespace rts {
/* -------------------------------------------------------------------- */
/* for_each algorithm                                                   */
/* -------------------------------------------------------------------- */
template<typename input_terator_type, typename Operation>
struct BodyForEach {
  typedef rts::Sequence<input_terator_type> sequence_type;

  BodyForEach( Operation& o ) : op(o) {}
  Operation op;
  bool operator()( input_terator_type& result, typename sequence_type::range_type& r )
  {
    input_terator_type first = r.begin();
    input_terator_type last  = r.end();

    while (first != last)
       op(*first++);
    result = last;
    return (first != last);
  }
};

}// rts

template<typename input_terator_type, typename Operation, typename Settings>
Operation for_each( 
  input_terator_type first, 
  input_terator_type last, 
  Operation op, 
  Settings settings
)
{
  typedef rts::Sequence<input_terator_type> sequence_type;

  if (first == last) return op;
  input_terator_type result = first;
  sequence_type seq(first, last-first);
  
  rts::MacroLoop< rts::NoReduce_MacroLoop_tag >::doit( 
    result,                                                     /* output: the result */
    seq,                                                        /* input: the sequence */
    rts::BodyForEach<input_terator_type, Operation>(op),        /* the body == NanoLoop */
    rts::empty_reducer<input_terator_type>,                     /* merge with a thief: do nothing */
    settings                                                    /* output: the result */
  );
  return op;
}

template<typename input_terator_type, typename Operation>
Operation for_each( 
  input_terator_type first, 
  input_terator_type last, 
  Operation op
)
{ 
  return for_each( first, last, op, rts::DefaultSetting() );
}


} // kastl

#endif // ! KASTL_FOR_EACH_H_INCLUDED
