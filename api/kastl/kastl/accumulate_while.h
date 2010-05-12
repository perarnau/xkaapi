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
#ifndef KASTL_ACCUMULATE_WHILE_H_INCLUDED
# define KASTL_ACCUMULATE_WHILE_H_INCLUDED
#include "kastl/kastl_impl.h"

namespace kastl {

namespace rts {
/* -------------------------------------------------------------------- */
/* accumulate_while algorithm                                                       */
/* -------------------------------------------------------------------- */
template<typename T, typename input_iterator_type>
struct BodyAccumulateWhile {
  BodyAccumulateWhile( const T& init ) : result(init) {}
  T result;
  void operator()( input_iterator_type& dummy, input_iterator_type& current )
  {
    result += *current;
  }
};

}// rts

template <typename T, typename input_iterator_type, typename Body, typename Inserter, typename Settings>
size_t accumulate_while( T& value, 
                         input_iterator_type first, input_iterator_type last, 
                         Body& body, 
                         Inserter& acc, 
                         const Settings& settings
                        )
{
  size_t cnt = 0;
  for (; (first != last); ++first, ++cnt)
  {
    if (acc(value, body(*first))) return 1+cnt;
  }
  return cnt;
}

template <typename T, typename input_iterator_type, typename Body, typename Inserter>
size_t accumulate_while( T& value, 
                         input_iterator_type first, input_iterator_type last, 
                         Body& body, 
                         Inserter& acc
                        )
{
  return accumulate_while(value, first, last, body, acc, rts::DefaultSetting() );
}

} // kastl

#endif // ! KASTL_ACCUMULATE_H_INCLUDED
