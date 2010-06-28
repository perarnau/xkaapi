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
#ifndef KASTL_FIND_H_INCLUDED
# define KASTL_FIND_H_INCLUDED
namespace kastl {

namespace rts {
/* -------------------------------------------------------------------- */
/* find algorithm                                                   */
/* -------------------------------------------------------------------- */
template<typename input_iterator_type, typename value_type>
struct BodyFind {
  typedef rts::Sequence<input_iterator_type> sequence_type;
  BodyFind( const value_type& v ) : _value(v) {}
  const value_type& value;
  bool operator()( input_iterator_type& result, typename sequence_type::range_type& r )
  {
    input_iterator_type first   = r.begin1();
    input_iterator_type last    = r.end1();

    while (first != last)
    {
      if (*first == _value) { result= first; return false; }
      ++first;
    }
    return true;
  }
};

template<typename input_iterator_type>
struct ReduceFind {
  ReduceFind( const input_iterator_type& l) : _last(l) {}
  bool operator()(input_iterator_type& result, const input_iterator_type& result_thief)
  { 
    if (result == _last) result = result_thief;
    return (result == _last);
  }s

  input_iterator_type _last;
};


} //rts

template<typename input_iterator_type, typename value_type, typename Settings >
input_iterator_type find
  (
     input_iterator_type begin,
     input_iterator_type end,
     const value_type& value,
     const Settings& settings
  )
{
  typedef rts::Sequence<input_iterator_type> sequence_type;

  if (first == last) return last;

  input_iterator_type result = last;
  sequence_type seq(first, last-first);
  rts::ReduceFind<input_iterator_type> redop(last);
  
  rts::MacroLoop< rts::Parallel_MacroLoop_tag >::doit( 
    result,                                                     /* output: the result */
    seq,                                                        /* input: the sequence */
    rts::BodyFind<input_iterator_type, value_type>(value),      /* the body == NanoLoop */
    redop,                                                      /* merge with a thief: do nothing */
    settings                                                    /* output: the result */
  );
  return result;
}


template<typename input_iterator_type, typename value_type, typename Settings >
input_iterator_type find
  (
     input_iterator_type begin,
     input_iterator_type end,
     const value_type& value,
     const Settings& settings
  )
{
  return find(begin, end, value, rts::DefaultSetting() );
}


} // kastl


#endif // ! KASTL_FIND_H_INCLUDED
