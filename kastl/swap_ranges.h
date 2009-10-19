/*
 *  test_swap_ranges.cpp
 *  ckaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_SWAP_RANGES_H
#define _CKAAPI_SWAP_RANGES_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator1, class RandomAccessIterator2>
void swap_ranges ( kaapi_steal_context_t* stealcontext, RandomAccessIterator1 begin, RandomAccessIterator1 end, 
            RandomAccessIterator2 first2);

/** Stucture of a work for swap_ranges
*/
template<class RandomAccessIterator1, class RandomAccessIterator2>
class Swap_Ranges_Struct {
public:
  /* cstor */
  Swap_Ranges_Struct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator1 ibeg,
    RandomAccessIterator1 iend,
    RandomAccessIterator2 obeg
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _obeg(obeg) 
  {}
  
  /* do swap_ranges */
  void doit();

  typedef typename std::iterator_traits<RandomAccessIterator1>::value_type value_type;

protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator1  _ibeg;
  RandomAccessIterator1  _iend;
  RandomAccessIterator2 _obeg;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>* w = (Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator1 ibeg, RandomAccessIterator1& iend, 
                        RandomAccessIterator2 obeg
                      )
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator1 local_end = iend;

    Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>* output_work =0;

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
                                              sizeof(Swap_Ranges_Struct<RandomAccessIterator1, 
                                              RandomAccessIterator2>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_obeg = obeg + (output_work->_ibeg - ibeg);
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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>* victim_work )
  {
    Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>* thief_work = 
      (Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::swap_ranges(thief_work->_ibeg, thief_work->_iend, thief_work->_obeg);
#else
      Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend,
              thief_work->_obeg 
            );
      work.doit();
#endif
    }
  }
};


/** Adaptive swap_ranges
*/
template<class RandomAccessIterator1, class RandomAccessIterator2>
void Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2>::doit()
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
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _obeg);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     std::swap_ranges(_ibeg, nano_iend, _obeg);
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
template<class RandomAccessIterator1, class RandomAccessIterator2>
void swap_ranges(kaapi_steal_context_t* stealcontext, RandomAccessIterator1 begin, RandomAccessIterator1 end,
         RandomAccessIterator2 begin2)
{
  kaapi_steal_context_initpush( stealcontext );
  Swap_Ranges_Struct<RandomAccessIterator1, RandomAccessIterator2> work( stealcontext, begin, end, 
            begin2);
  work.doit();

}
#endif
