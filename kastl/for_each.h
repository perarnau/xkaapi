/*
 *  test_for_each.cpp
 *  ckaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_FOR_EACH_STRUCT_H
#define _CKAAPI_FOR_EACH_STRUCT_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator, class Function>
void for_each ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
            Function f);

/** Stucture of a work for for_each
*/
template<class RandomAccessIterator, class Function>
class ForEachStruct {
public:
  /* cstor */
  ForEachStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Function f
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _f(f) 
  {}
  
  /* do for_each */
  void doit();

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;

protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Function _f;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    ForEachStruct<RandomAccessIterator, Function>* w = (ForEachStruct<RandomAccessIterator, 
              Function>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, 
                        Function f
                      )
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    ForEachStruct<RandomAccessIterator, Function>* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    
    bloc = size / (1+count);
    if (bloc < 128) { count = size/128 -1; bloc = 128; }
    while (count >0)
    {
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(ForEachStruct<RandomAccessIterator, Function>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_f = f;
          ckaapi_assert( output_work->_iend - output_work->_ibeg >0);

          /* reply ok (1) to the request */
          kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
        }
        else {
          /* reply failed (=last 0 in parameter) to the request */
          kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        }
        --count; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  iend  = local_end;
  ckaapi_assert( iend - ibeg >0);
  return;
      
reply_failed:
    while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        --count; 
      }
      ++i;
    }
  }


  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, ForEachStruct<RandomAccessIterator, Function>* victim_work )
  {
    ForEachStruct<RandomAccessIterator, Function>* thief_work = 
      (ForEachStruct<RandomAccessIterator, Function>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::for_each(thief_work->_ibeg, thief_work->_iend, thief_work->_f);
#else
      ForEachStruct<RandomAccessIterator, Function> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_f 
            );
      work.doit();
#endif
    }
  }
};


/** Adaptive for_each
*/
template<class RandomAccessIterator, class Function>
void ForEachStruct<RandomAccessIterator,Function>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 1024;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _f);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     std::for_each(_ibeg, nano_iend, _f);
     _ibeg +=unit_size;

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Function>
void for_each(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, Function f)
{
  kaapi_steal_context_initpush( stealcontext );
  ForEachStruct<RandomAccessIterator, Function> work( stealcontext, begin, end, f);
  work.doit();

}
#endif
