/*
 *  test_count.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_COUNT_H
#define _CKAAPI_COUNT_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator, class T>
 typename std::iterator_traits<RandomAccessIterator>::difference_type
   count ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
             const T& value);

/** Stucture of a work for count
*/
template<class RandomAccessIterator, class T>
class CountStruct {
public:

  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type Distance_type;
  /* cstor */
  CountStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    const T& value,
    Distance_type local_count
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _value(value), _local_count(local_count) 
  {}
  
  /* do count */
  void doit();

  /* get result */
  Distance_type get_count() {
     return _local_count;
  }
 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  T _value;
  Distance_type _local_count;  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    CountStruct<RandomAccessIterator, T>* w = (CountStruct<RandomAccessIterator, 
              T>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, const T& value)
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    CountStruct<RandomAccessIterator, T>* output_work =0;

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
                                              sizeof(CountStruct<RandomAccessIterator, T>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_value = value;
          output_work->_local_count = 0;
          kaapi_assert( output_work->_iend - output_work->_ibeg >0);

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
  kaapi_assert( iend - ibeg >0);
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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, 
                       CountStruct<RandomAccessIterator, T>* victim_data )
  {
    CountStruct<RandomAccessIterator, T>* thief_work = 
      (CountStruct<RandomAccessIterator, T>* )thief_data;

    CountStruct<RandomAccessIterator, T>* victim_work =
      (CountStruct<RandomAccessIterator, T>* )victim_data;

    victim_work->_local_count +=thief_work->_local_count; //merge of the two results


    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBCOUNT)
      victim_work->_local_count +=std::count(thief_work->_ibeg, thief_work->_iend, thief_work->_value);
#else
      CountStruct<RandomAccessIterator, T> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_value,
              victim_work->_local_count 
            );
      work.doit();

      victim_work->_local_count = work._local_count;
#endif
    }

  }
};


/** Adaptive count
*/
template<class RandomAccessIterator, class T>
void CountStruct<RandomAccessIterator,T>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  ptrdiff_t unit_size = 512;

  ptrdiff_t tmp_size = 0;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _value);

    tmp_size = _iend-_ibeg;
    if(tmp_size < unit_size ) {
       unit_size = tmp_size; nano_iend = _iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }
   
    /* sequential computation */
     _local_count +=std::count(_ibeg, nano_iend, _value);
     _ibeg +=unit_size;

    //if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class T>
typename std::iterator_traits<RandomAccessIterator>::difference_type
   count(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
          const T& value)
{
  kaapi_steal_context_initpush( stealcontext );
  CountStruct<RandomAccessIterator, T> work( stealcontext, begin, end, value, 0);
  work.doit();

 return work.get_count();
}
#endif
