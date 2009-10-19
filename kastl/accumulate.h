/*
 *  test_accumulate.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_ACCUMULATE_H
#define _CKAAPI_ACCUMULATE_H
#include "kaapi_adapt.h"
#include <algorithm>
#include<numeric>
#include<functional>

template<class RandomAccessIterator, class T, class BinOp>
 T  accumulate ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
                 T init, BinOp op);

template<class RandomAccessIterator, class T>
 T  accumulate ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
                 T init);

/** Stucture of a work for accumulate
*/
template<class RandomAccessIterator, class T, class BinOp>
class AccumulateStruct {
public:
  /* cstor */
  AccumulateStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    T init,
    BinOp op,
    bool is_master = true) : _sc(sc), _ibeg(ibeg), _iend(iend), _local_accumulate(init),
                             _op(op), _is_master(is_master) 
  {}
  
  /* do accumulate */
  void doit();

  /* get result */
  T get_accumulate() {
     return _local_accumulate;
  }
 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  T _local_accumulate; 
  BinOp _op;
  bool _is_master; 
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    AccumulateStruct<RandomAccessIterator, T, BinOp>* w = (AccumulateStruct<RandomAccessIterator, 
              T, BinOp>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, BinOp op)
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    AccumulateStruct<RandomAccessIterator, T, BinOp>* output_work =0;

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
                                              sizeof(AccumulateStruct<RandomAccessIterator, T, BinOp>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_local_accumulate = 0;
          output_work->_op = op;
          output_work->_is_master = false;
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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, 
                       AccumulateStruct<RandomAccessIterator, T, BinOp>* victim_data )
  {
    AccumulateStruct<RandomAccessIterator, T, BinOp>* thief_work = 
      (AccumulateStruct<RandomAccessIterator, T, BinOp>* )thief_data;

    AccumulateStruct<RandomAccessIterator, T, BinOp>* victim_work =
      (AccumulateStruct<RandomAccessIterator, T, BinOp>* )victim_data;


   //merge of the two results
   if(thief_work->_local_accumulate!=T(0))    
     victim_work->_local_accumulate  = victim_work->_op(victim_work->_local_accumulate,
                                                        thief_work->_local_accumulate);

    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBCOUNT)
      victim_work->_local_accumulate = std::accumulate(thief_work->_ibeg, thief_work->_iend, 
                                        victim_work->__local_accumulate, victim_work->_op);
#else

      AccumulateStruct<RandomAccessIterator, T, BinOp> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              victim_work->_local_accumulate,
              thief_work->_op 
            );
      work.doit();

      victim_work->_local_accumulate = work._local_accumulate;
#endif
    }

  }
};


/** Adaptive accumulate
*/
template<class RandomAccessIterator, class T, class BinOp>
void AccumulateStruct<RandomAccessIterator, T, BinOp>::doit()
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
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _op);

    tmp_size = _iend-_ibeg;
    if(tmp_size < unit_size ) {
       unit_size = tmp_size; nano_iend = _iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }
    
    /* sequential computation */
    if(_is_master) _local_accumulate = std::accumulate(_ibeg, nano_iend, _local_accumulate, _op);
    else {
     RandomAccessIterator first = _ibeg;
     RandomAccessIterator last  = nano_iend;
     T init = *first++;
         while ( first!=last ) init = _op(init, *first++);
         _local_accumulate = (_local_accumulate==T(0))?init: _op(_local_accumulate, init);
    }
    _ibeg +=unit_size;

    //if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their inits which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class T, class BinOp>
T accumulate(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
           T init, BinOp op)
{
  AccumulateStruct<RandomAccessIterator, T, BinOp> work( stealcontext, begin, end, init, op);
  work.doit();

 return work.get_accumulate();
}

template<class RandomAccessIterator, class T>
T accumulate(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
           T init)
{
  kaapi_steal_context_initpush( stealcontext );
 return accumulate(stealcontext, begin, end, init, std::plus<T>());
}

#endif
