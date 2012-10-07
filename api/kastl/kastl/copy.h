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


#ifndef KASTL_COPY_H_INCLUDED
# define KASTL_COPY_H_INCLUDED


#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator0, typename Iterator1>
struct copy_body
{
  typedef kastl::impl::dummy_type result_type;

  void operator()(result_type&, Iterator0& ipos, Iterator1& opos)
  {
    *opos = *ipos;
  }
};

template<typename Iterator0, typename Iterator1, typename Settings>
Iterator1 copy
(Iterator0 ifirst, Iterator0 ilast, Iterator1 ofirst, const Settings& settings)
{
  kastl::rts::Sequence<Iterator0, Iterator1> seq
    (ifirst, ofirst, ilast - ifirst);
  copy_body<Iterator0, Iterator1> body;
  kastl::impl::foreach_loop(seq, body, settings);
  return ofirst + (ilast - ifirst);
}

template<typename Iterator0, typename Iterator1>
Iterator1 copy(Iterator0 ifirst, Iterator1 ilast, Iterator1 ofirst)
{
  kastl::impl::static_settings settings(512, 512);
  return kastl::copy(ifirst, ilast, ofirst, settings);
}

} // kastl::


#endif // ! KASTL_COPY_H_INCLUDED
