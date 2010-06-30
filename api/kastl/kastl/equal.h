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
#ifndef KASTL_EQUAL_H_INCLUDED
# define KASTL_EQUAL_H_INCLUDED
#include "kastl/kastl_impl.h"

namespace kastl

namespace rts {
/* -------------------------------------------------------------------- */
/* equal algorithm                                                   */
/* -------------------------------------------------------------------- */
template<typename sequence_type>
struct BodyEqual1 {
  bool operator()( bool& result, typename sequence_type::range_type& r )
  {
    sequence_type::input_iterator_type1 first1 = r.begin1();
    sequence_type::input_iterator_type1 last1  = r.end1();
    sequence_type::input_iterator_type2 first2 = r.begin2();

    while (first1 != last1)
    {
      if (*first1 != *first2)
      {
        result = false;
        return false;
      }
      ++first1; ++first2;
    }
    return true;
  }
};

template<typename sequence_type, typename BinaryPredicate>
struct BodyEqual2 {
  BodyEqual2(BodyEqual2& p) : pred(p) {}
  bool operator()( bool& result, typename sequence_type::range_type& r )
  {
    sequence_type::input_iterator_type1 first1 = r.begin1();
    sequence_type::input_iterator_type1 last1  = r.end1();
    sequence_type::input_iterator_type2 first2 = r.begin2();

    while (first1 != last1)
    {
      if ( !pred(*first1, *first2) )
      {
        result = false;
        return false;
      }
      ++first1; ++first2;
    }
    return true;
  }
  BinaryPredicate& pred;
};


struct reducerBodyEqual {  
  bool operator()(bool& result, const bool& result_thief)
  { 
    if (!result) return false;
    if (!result_thief) return result = false;
  }
};

}// rts

template <typename input_iterator_type1, typename input_iterator_type2, typename Settings = rts::DefaultSetting>
bool
   equal( input_iterator_type1 first1, input_iterator_type1 last1, 
          input_iterator_type2 first2, 
          const Settings& settings = rts::DefaultSetting())
{
  typedef rts::Sequence<input_iterator_type1,input_iterator_type2> sequence_type;

  if (first == last) return true;
  sequence_type seq(first1, first2, last1-first1);
  bool result = true;
  rts::BodyEqual1<sequence_type> be;
  
  rts::MacroLoop< rts::Sequential_MacroLoop_tag >( 
    result,                                                     /* output: the result */
    seq,                                                        /* input: the sequence */
    be,                                                         /* the body == NanoLoop */
    rts::reducerBodyEqual                                       /* merge with a thief: do nothing */
    settings                                                    /* output: the result */
  );
  return result;
}


template <typename input_iterator_type1, typename input_iterator_type2, typename BinaryPredicate, typename Settings = rts::DefaultSetting>
bool
   equal( input_iterator_type1 first1, input_iterator_type1 last1, 
          input_iterator_type2 first2, 
          BinaryPredicate pred,
          const Settings& settings = rts::DefaultSetting())
{
  typedef rts::Sequence<input_iterator_type1,input_iterator_type2> sequence_type;

  if (first == last) return true;
  sequence_type seq(first1, first2, last1-first1);
  bool result = true;
  rts::BodyEqual2<sequence_type,BinaryPredicate> be(pred);
  
  rts::MacroLoop< rts::Sequential_MacroLoop_tag >( 
    result,                                                     /* output: the result */
    seq,                                                        /* input: the sequence */
    be,                                                         /* the body == NanoLoop */
    rts::reducerBodyEqual                                       /* merge with a thief: do nothing */
    settings                                                    /* output: the result */
  );
  return result;
}

} // kastl

#endif // ! KASTL_EQUAL_H_INCLUDED
