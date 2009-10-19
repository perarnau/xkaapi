/*
 *  test_count_if.cpp
 *  ckaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_COUNT_IF_H
#define _CKAAPI_COUNT_IF_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator, class Predicate>
 typename std::iterator_traits<RandomAccessIterator>::difference_type
   count_if ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
             Predicate pred);

/** Stucture of a work for count_if
*/
template<class RandomAccessIterator, class Predicate>
class CountIFStruct {
public:

  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type Distance_type;
  /* cstor */
  CountIFStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Predicate pred,
    Distance_type local_count_if
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _pred(pred), _local_count_if(local_count_if) 
  {}
  
  /* do count_if */
  void doit();

  /* get result */
  Distance_type get_count_if() {
     return _local_count_if;
  }
 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Predicate _pred;
  Distance_type _local_count_if;  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    CountIFStruct<RandomAccessIterator, Predicate>* w = (CountIFStruct<RandomAccessIterator, 
              Predicate>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count_if, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, Predicate pred)
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    CountIFStruct<RandomAccessIterator, Predicate>* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    
    bloc = size / (1+count_if);
    if (bloc < 128) { count_if = size/128 -1; bloc = 128; }
    while (count_if >0)
    {
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(CountIFStruct<RandomAccessIterator, Predicate>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_pred = pred;
          output_work->_local_count_if = 0;
          ckaapi_assert( output_work->_iend - output_work->_ibeg >0);

          /* reply ok (1) to the request */
          kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
        }
        else {
          /* reply failed (=last 0 in parameter) to the request */
          kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        }
        --count_if; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  iend  = local_end;
  ckaapi_assert( iend - ibeg >0);
  return;
      
reply_failed:
    while (count_if >0)
    {
      if (request[i] !=0)
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        --count_if; 
      }
      ++i;
    }
  }


  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, 
                       CountIFStruct<RandomAccessIterator, Predicate>* victim_data )
  {
    CountIFStruct<RandomAccessIterator, Predicate>* thief_work = 
      (CountIFStruct<RandomAccessIterator, Predicate>* )thief_data;

    CountIFStruct<RandomAccessIterator, Predicate>* victim_work =
      (CountIFStruct<RandomAccessIterator, Predicate>* )victim_data;

    victim_work->_local_count_if +=thief_work->_local_count_if; //merge of the two results


    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBCOUNT)
      victim_work->_local_count_if +=std::count_if(thief_work->_ibeg, thief_work->_iend, thief_work->_pred);
#else
      CountIFStruct<RandomAccessIterator, Predicate> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_pred,
              victim_work->_local_count_if 
            );
      work.doit();

      victim_work->_local_count_if = work._local_count_if;
#endif
    }

  }
};


/** Adaptive count_if
*/
template<class RandomAccessIterator, class Predicate>
void CountIFStruct<RandomAccessIterator, Predicate>::doit()
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
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _pred);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     _local_count_if +=std::count_if(_ibeg, nano_iend, _pred);
     _ibeg +=unit_size;

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their preds which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Predicate>
typename std::iterator_traits<RandomAccessIterator>::difference_type
   count_if(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
          Predicate pred)
{
  kaapi_steal_context_initpush( stealcontext );
  CountIFStruct<RandomAccessIterator, Predicate> work( stealcontext, begin, end, pred, 0);
  work.doit();

 return work.get_count_if();
}
#endif
