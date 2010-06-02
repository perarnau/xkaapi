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


#ifndef KASTL_REPLACE_H_INCLUDED
# define KASTL_REPLACE_H_INCLUDED


#include "kastl_loop.h"
#include "kastl_sequences.h"


namespace kastl
{

template<typename Iterator, typename Value>
struct replace_body
{
  typedef kastl::impl::dummy_type result_type;

  const Value& _old_value;
  const Value& _new_value;

  replace_body(const Value& old_value, const Value& new_value)
    : _old_value(old_value), _new_value(new_value)
  {}

  bool operator()(result_type&, Iterator& pos)
  {
    if (*pos == _old_value)
      *pos = _new_value;
    return false;
  }
};

template<typename Iterator, typename Value, typename Settings>
void replace
(Iterator first, Iterator last,
 const Value& old_value, const Value& new_value,
 const Settings& settings)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  replace_body<Iterator, Value> body(old_value, new_value);
  kastl::impl::unrolled_loop::run(seq, body, settings);
}

template<typename Iterator, typename Value>
void replace
(Iterator first, Iterator last,
 const Value& old_value, const Value& new_value)
{
  kastl::impl::static_settings settings(512, 512);
  kastl::replace(first, last, old_value, new_value, settings);
}

} // kastl::


#endif // ! KASTL_REPLACE_H_INCLUDED
