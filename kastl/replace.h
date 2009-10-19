/*
 *  test_replace.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_REPLACE_H
#define _CKAAPI_REPLACE_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator, class T>
void replace ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
            const T& old_value, const T& new_value);

/** Stucture of a work for replace
*/
template<class RandomAccessIterator, class T>
class ReplaceStruct {
public:
  /* cstor */
  ReplaceStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    const T& old_value,
    const T& new_value
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _old_value(old_value), _new_value(new_value) 
  {}
  
  /* do replace */
  void doit();

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;

protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  T _old_value;
  T _new_value;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    ReplaceStruct<RandomAccessIterator, T>* w = (ReplaceStruct<RandomAccessIterator, 
              T>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, 
                        const T& old_value, const T& new_value
                      )
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    ReplaceStruct<RandomAccessIterator, T>* output_work =0;

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
                                              sizeof(ReplaceStruct<RandomAccessIterator, T>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_old_value = old_value;
          output_work->_new_value = new_value;
          xkaapi_assert( output_work->_iend - output_work->_ibeg >0);

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
  xkaapi_assert( iend - ibeg >0);
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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, ReplaceStruct<RandomAccessIterator, T>* victim_work )
  {
    ReplaceStruct<RandomAccessIterator, T>* thief_work = 
      (ReplaceStruct<RandomAccessIterator, T>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::replace(thief_work->_ibeg, thief_work->_iend, thief_work->_old_value, thief_work->_new_value);
#else
      ReplaceStruct<RandomAccessIterator, T> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_old_value,
              thief_work->_new_value 
            );
      work.doit();
#endif
    }
  }
};


/** Adaptive replace
*/
template<class RandomAccessIterator, class T>
void ReplaceStruct<RandomAccessIterator,T>::doit()
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
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _old_value, _new_value);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     std::replace(_ibeg, nano_iend, _old_value, _new_value);
     _ibeg +=unit_size;

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class T>
void replace(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
      const T& old_value, const T& new_value)
{
  kaapi_steal_context_initpush( stealcontext );
  ReplaceStruct<RandomAccessIterator, T> work( stealcontext, begin, end, old_value, new_value);
  work.doit();

}
#endif
