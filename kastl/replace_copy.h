/*
 *  test_replace_copy.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_REPLACE_COPY_H
#define _CKAAPI_REPLACE_COPY_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator1, class RandomAccessIterator2, class T>
void replace_copy ( kaapi_steal_context_t* stealcontext, RandomAccessIterator1 begin, RandomAccessIterator1 end, 
            RandomAccessIterator2 res, const T& old_value, const T& new_value);

/** Stucture of a work for replace_copy
*/
template<class RandomAccessIterator1, class RandomAccessIterator2, class T>
class Replace_Copy_Struct {
public:
  /* cstor */
  Replace_Copy_Struct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator1 ibeg,
    RandomAccessIterator1 iend,
    RandomAccessIterator2 obeg,
    const T& old_value,
    const T& new_value
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _obeg(obeg), _old_value(old_value), _new_value(new_value) 
  {}
  
  /* do replace_copy */
  void doit();

  typedef typename std::iterator_traits<RandomAccessIterator1>::value_type value_type;

protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator1  _ibeg;
  RandomAccessIterator1  _iend;
  RandomAccessIterator2 _obeg;
  T _old_value;
  T _new_value;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>* w = (Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator1 ibeg, RandomAccessIterator1& iend, 
                        RandomAccessIterator2 obeg, const T& old_value, const T& new_value
                      )
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator1 local_end = iend;

    Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>* output_work =0;

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
                                              sizeof(Replace_Copy_Struct<RandomAccessIterator1, 
                                              RandomAccessIterator2, T>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_obeg = obeg + (output_work->_ibeg - ibeg);
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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>* victim_work )
  {
    Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>* thief_work = 
      (Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::replace_copy(thief_work->_ibeg, thief_work->_iend, thief_work->_old_value, thief_work->_new_value);
#else
      Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend,
              thief_work->_obeg, 
              thief_work->_old_value,
              thief_work->_new_value 
            );
      work.doit();
#endif
    }
  }
};


/** Adaptive replace_copy
*/
template<class RandomAccessIterator1, class RandomAccessIterator2, class T>
void Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator1 nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 1024;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _obeg, _old_value, _new_value);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     std::replace_copy(_ibeg, nano_iend, _obeg, _old_value, _new_value);
     _ibeg +=unit_size;
     _obeg +=unit_size;

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator1, class RandomAccessIterator2, class T>
void replace_copy(kaapi_steal_context_t* stealcontext, RandomAccessIterator1 begin, RandomAccessIterator1 end,
         RandomAccessIterator2 res, const T& old_value, const T& new_value)
{
  kaapi_steal_context_initpush( stealcontext );
  Replace_Copy_Struct<RandomAccessIterator1, RandomAccessIterator2, T> work( stealcontext, begin, end, 
            res, old_value, new_value);
  work.doit();

}
#endif
