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
#ifndef  KASTL_IMPL_H_INCLUDED
#define  KASTL_IMPL_H_INCLUDED

#include "kastl/kastl_sequences.h"


namespace kastl {

namespace rts {

/* dummy code to get a pointer to a template static method */
template<typename T>
T get_my_splitter(T t) { return t; }

/* -------------------------------------------------------------------- */
/* Empty splitter                                                       */
/* -------------------------------------------------------------------- */
template<typename sequence_type>
class EmptySplitter {
};

template<typename sequence_type>
class LinearSplitter {
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    sequence_type* seq = static_cast<sequence_type*>(_seq);
    DefaultExtractor* extractor = static_cast<DefaultExtractor*> (argsplitter);
    size_t size = seq.size();     /* upper bound */
    if (size ==0) return 0;

    /* take at most count*_pargrain items per thief */
    size_t blocsize = extractor->_unitsize;
    size_t size_max = (blocsize * count);
    typename sequence_type::range_type r;

    /* */
    if ( !seq->steal(r, size_max )) return 0;
    kaapi_assert_debug (!r.is_empty());
    
    /* equally distribute the work load in r or only distribution by bloc of size at most unitsize ? */
    for (; request_count > 0; ++request)
    {
      if (!kaapi_request_ok(request))
        continue;

      kaapi_thread_t* thief_thread = kaapi_request_getthread(request);
      kaapi_task_t* thief_task = kaapi_thread_toptask(thief_thread);
      WorkType* thief_work = static_cast<WorkType*>
              (kaapi_thread_pushdata_align(thief_thread, sizeof(WorkType), 8));
    }

    return false;
  }
};

/* -------------------------------------------------------------------- */
/* Sequential Extractor / Preemptor                                     */
/* -------------------------------------------------------------------- */
struct SequentialExtractor {
  template<typename sequence_type, typename Settings>
  SequentialExtractor( sequence_type& seq, const Settings& settings )
   : _seq(&seq)
  {}
  template<typename sequence_type>
  bool extract( sequence_type& seq, typename sequence_type::range_type& r)
  {
    kaapi_assert_debug(_seq == (void*)&seq);
    return seq.pop_safe( r, seq.size() );
  }

  template<typename sequence_type>
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
    sequence_type* seq __attribute__((unused)) = static_cast<sequence_type*>(_seq);
    return 0;
  }

  template<typename sequence_type>
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    SequentialExtractor* object = static_cast<SequentialExtractor*>(argsplitter);
    return object->splitter<sequence_type>(sc, count, request);
  }

  template<typename sequence_type>
  static kaapi_task_splitter_t get_static_splitter( const sequence_type& )
  {
    return &static_splitter<sequence_type>;
  }


  template<
      typename result_type, 
      typename sequence_type, 
      typename settings_type, 
      typename reduce_body_type
  >
  bool preempt(result_type&  result, sequence_type& seq, reduce_body_type reduce, settings_type settings)
  { return false;}

protected:
  void* _seq;
};



/* -------------------------------------------------------------------- */
/* Default Extractor / Splittor / Preemptor                              */
/* -------------------------------------------------------------------- */
struct DefaultExtractor {
  template<typename sequence_type, typename Settings>
  DefaultExtractor( sequence_type& seq, const Settings& settings )
   : _seq(&seq), _unitsize(settings.unitsize)
  {}
  template<typename sequence_type>
  bool extract( sequence_type& seq, typename sequence_type::range_type& r)
  {
    kaapi_assert_debug(_seq == (void*)&seq);
    return seq.pop( r, _unitsize );
  }

  template<typename sequence_type>
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    SequentialExtractor* object = static_cast<SequentialExtractor*>(argsplitter);
    return object->splitter<sequence_type>(sc, count, request);
  }

  template<typename result_type, typename reduce_body_type>
  static int static_reducer( void* thief_arg, void* pthief_result, size_t thief_size,
                             result_type* result, reduce_body_type& reducer )
  {
    result_type* thief_result = static_cast<result_type*> (pthief_result);
    reducer( *result, *thief_result);
    return 0; 
  }
  
  template<
      typename result_type, 
      typename sequence_type, 
      typename settings_type, 
      typename reduce_body_type
  >
  bool preempt( kaapi_stealcontext_t* const stc, 
                result_type&  result, 
                sequence_type& seq, 
                reduce_body_type reducer, 
                settings_type settings )
  {
    int (*ptr_reducer)( kaapi_stealcontext_t*, void*, void*, size_t,
                        result_type*, reduce_body_type* ) 
        = static_reducer<result_type, reduce_body_type>;
    kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head( stc );
    while (ktr !=0)
    {
      if (kaapi_preempt_thief(stc, ktr, &result, ptr_reducer, &result, reducer))
        return true;
      ktr = kaapi_get_nextthief_head( stc, ktr );
    }
    return false;
  }
protected:
  void* _seq;
  size_t _unitsize;
};


/* -------------------------------------------------------------------- */
/* Window sized Extractor / Splittor / Preemptor                         */
/* -------------------------------------------------------------------- */
struct WindowExtractor {
  template<typename sequence_type, typename Settings>
  WindowExtractor( const Settings& settings )
   : _unitsize(settings.unitsize)
  {}
  template<typename sequence_type>
  bool extract( sequence_type& seq, typename sequence_type::range_type& r)
  {
    return seq.pop( r, _unitsize );
  }

  template<typename sequence_type>
  bool splitter( sequence_type& seq)
  {
    return false;
  }

  template<typename result_type, typename reduce_body_type>
  int static_reducer( kaapi_stealcontext_t* sc, void* thief_arg, void* thief_result, size_t thief_size,
                      result_type* result, reduce_body_type* reducer )
  {
    return 0; 
  }

  template<
      typename result_type, 
      typename sequence_type, 
      typename settings_type, 
      typename reduce_body_type
  >
  bool preempt( kaapi_stealcontext_t* const stc, 
                result_type&  result, 
                sequence_type& seq, 
                reduce_body_type reducer, 
                settings_type settings )
  {
    int (*ptr_reducer)( kaapi_stealcontext_t*, void*, void*, size_t,
                        result_type*, reduce_body_type* ) 
        = static_reducer<result_type, reduce_body_type>;
    kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head( stc );
    while (ktr !=0)
    {
      if (kaapi_preempt_thief(stc, ktr, 0, ptr_reducer, &result, &reducer))
        return true;
      ktr = kaapi_get_nextthief_head( stc, ktr );
    }
    return false;
  }
protected:
  size_t _unitsize;
};




/* -------------------------------------------------------------------- */
/* empty reducer                                                        */
/* -------------------------------------------------------------------- */
template<typename iterator_type>
void empty_reducer(const iterator_type& result_thief) 
{
}

/* -------------------------------------------------------------------- */
/* Settings                                                             */
/* -------------------------------------------------------------------- */
struct DefaultSetting {
  typedef SequentialExtractor macro_extractor_type;
};


/* -------------------------------------------------------------------- */
/* sequential MacroLoop                                                 */
/* -------------------------------------------------------------------- */
struct Sequential_MacroLoop_tag{};

/* -------------------------------------------------------------------- */
/* Parallel MacroLoop, without reduction                                */
/* -------------------------------------------------------------------- */
struct NoReduce_MacroLoop_tag{};

/* -------------------------------------------------------------------- */
/* Parallel MacroLoop, general case with reduction                      */
/* -------------------------------------------------------------------- */
struct Parallel_MacroLoop_tag{};


/* -------------------------------------------------------------------- */
/* MacroLoop                                                            */
/* -------------------------------------------------------------------- */
template<
    typename TAG
>
struct MacroLoop {
  /* should have 'do' method:
    template<
        typename result_type, 
        typename sequence_type, 
        typename nano_body_type, 
        typename settings_type, 
        typename reduce_body_type = dummy_type
    >
    void do( result_type& result, sequence_type& seq, nano_body_type nano_loop, reduce_body_type reduce, settings_type settings );
  */
};


/* -------------------------------------------------------------------- */
/* MacroLoop: specialisation for sequential tag                         */
/* -------------------------------------------------------------------- */
template<>
struct MacroLoop<Sequential_MacroLoop_tag> {
  template<
      typename result_type, 
      typename sequence_type, 
      typename nano_body_type, 
      typename settings_type, 
      typename reduce_body_type
  >
  static void doit(
    result_type& result, 
    sequence_type& seq, 
    nano_body_type nano_loop, 
    reduce_body_type reduce, 
    settings_type settings 
  )
  {
    typename settings_type::macro_extractor_type extractor(settings);
    typename sequence_type::range_type range;

    while ( extractor.extract(seq, range) && nano_loop(result, range) )
     ;
  }
};

/* -------------------------------------------------------------------- */
/* MacroLoop: specialisation for NoReduce tag                           */
/* -------------------------------------------------------------------- */
template<>
struct MacroLoop<NoReduce_MacroLoop_tag> {
  template<
      typename result_type, 
      typename sequence_type, 
      typename nano_body_type, 
      typename settings_type, 
      typename reduce_body_type
  >
  static void doit(
    result_type& result, 
    sequence_type& seq, 
    nano_body_type nano_loop, 
    reduce_body_type reduce, 
    settings_type settings 
  )
  {
    typename settings_type::macro_extractor_type extractor(seq, settings);
    typename sequence_type::range_type range;

    int (*splitter)(struct kaapi_stealcontext_t*, int, struct kaapi_request_t*, void*) 
      = settings_type::macro_extractor_type::get_static_splitter(seq);

    /* push adaptive task */
    kaapi_thread_t* thread = kaapi_self_thread();
    kaapi_stealcontext_t* const stc = kaapi_thread_pushstealcontext (
        thread, 
        KAAPI_STEALCONTEXT_DEFAULT, 
        splitter,
        &extractor,
        0
    );

    /* execute sequential computation */
    while ( extractor.extract(seq, range) && nano_loop(result, range) )
     ;
    
    kaapi_steal_finalize(stc);
  }
};


/* -------------------------------------------------------------------- */
/* MacroLoop: specialisation for Parallel tag                           */
/* -------------------------------------------------------------------- */
template<>
struct MacroLoop<Parallel_MacroLoop_tag> {
  template<
      typename result_type, 
      typename sequence_type, 
      typename nano_body_type, 
      typename settings_type, 
      typename reduce_body_type
  >
  static void doit(
    result_type& result, 
    sequence_type& seq, 
    nano_body_type nano_loop, 
    reduce_body_type reduce, 
    settings_type settings 
  )
  {
    typename settings_type::macro_extractor_type extractor(seq, settings);
    typename sequence_type::range_type range;

    int (*splitter)(struct kaapi_stealcontext_t*, int, struct kaapi_request_t*, void*) 
      = settings_type::macro_extractor_type::get_static_splitter(seq);

    /* push adaptive task */
    kaapi_thread_t* thread = kaapi_self_thread();
    kaapi_stealcontext_t* const stc = kaapi_thread_pushstealcontext (
        thread, 
        KAAPI_STEALCONTEXT_DEFAULT, 
        splitter,
        &extractor,
        0
    );

  redo_compute:
    /* execute sequential computation */
    while ( extractor.extract(seq, range) && nano_loop(result, range) )
     ;

    /* preempt and merge result with all the thiefs 
       if preempt return true, the local sequential computation is restarted.
    */
    if (extractor.preempt(stc, seq, result, reduce, settings)) goto redo_compute;
    
    
    kaapi_steal_finalize(stc);
  }
};



/* -------------------------------------------------------------------- */
/* reduce algorithm, used by algorithm with associative operator        */
/* operator takes args as (Iterator&, Iterator)                         */
/* -------------------------------------------------------------------- */
template<typename input_iterator_type, typename Operation>
struct __reduce_nanoloop {
  typedef rts::Sequence<input_iterator_type> sequence_type;
  
  __reduce_nanoloop( const Operation& o) : op(o) {}
  bool operator()(input_iterator_type& result, typename sequence_type::range_type& r)
  { 
    input_iterator_type first = r.begin();
    input_iterator_type last  = r.end();
    
    while (first != last)
    {
      op(result, first);
      ++first;
    }
    return false;
  }

  Operation op;
};


template<typename input_iterator_type, typename Operation>
struct __reduce_reducer {
  __reduce_reducer( const Operation& o) : op(o) {}
  bool operator()(input_iterator_type& result, const input_iterator_type& result_thief)
  { 
    op(result, result_thief);
    return false; /* do not continue with local sequence */
  }

  Operation op;
};


template<typename input_iterator_type, typename Operation, typename Settings>
input_iterator_type ReduceLoop( input_iterator_type first, input_iterator_type last, Operation op, const Settings& settings )
{
  typedef rts::Sequence<input_iterator_type> sequence_type;

  input_iterator_type result = first++;
  if (first == last) return result;
  
  sequence_type seq(first, typename sequence_type::size_type(last-first));
  rts::MacroLoop< rts::Sequential_MacroLoop_tag >( 
    result,                                                     /* output: the result */
    seq,                                                        /* input: the sequence */
    __reduce_nanoloop<input_iterator_type,Operation>(op),       /* the body == NanoLoop */
    __reduce_reducer<input_iterator_type,Operation>(op),        /* merge with a thief */
    settings                                                    /* output: the result */
  );
  return result;
}


} // namespace rts

} // kastl


#endif // ! KASTL_IMPL_H_INCLUDED
