/*
 *  test_count_if.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Updated by FLM on december 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_COUNT_IF_H
#define _XKAAPI_COUNT_IF_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>


template<class RandomAccessIterator, class Predicate>
 typename std::iterator_traits<RandomAccessIterator>::difference_type
   count_if ( RandomAccessIterator begin, RandomAccessIterator end, Predicate pred);

/** Stucture of a work for count_if
*/
template<class RandomAccessIterator, class Predicate>
class CountIFStruct {
public:

  typedef CountIFStruct<RandomAccessIterator, Predicate> Self_t;
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type Distance_type;
  /* cstor */
  CountIFStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Predicate pred,
    Distance_type local_count_if
  ) : _ibeg(ibeg), _iend(iend), _pred(pred), _local_count_if(local_count_if) 
  {}
  
  /* do count_if */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* get result */
  Distance_type get_count_if() {
     return _local_count_if;
  }
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Predicate _pred;
  Distance_type _local_count_if;  

  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    size_t bloc;
    Predicate pred;

    request_handler(RandomAccessIterator& _local_end, size_t _bloc, const Predicate& _pred)
    : local_end(_local_end), bloc(_bloc), pred(_pred) {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_pred = pred;
      output_work->_local_count_if = 0;
      kaapi_assert( output_work->_iend - output_work->_ibeg >0);

      return true;
    }
  };

  typedef struct request_handler request_handler_t;

  /** splitter_work is called within the context of the steal point
  */
  int splitter
  (
   kaapi_stack_t* victim_stack,
   kaapi_task_t* task,
   int count,
   kaapi_request_t* request
  )
  {
    size_t size = (_iend - _ibeg);
    const int total_count = count;
    int replied_count = 0;
    size_t bloc;

    /* threshold should be defined (...) */
    if (size < 512)
      goto finish_splitter;
    
    bloc = size / (1 + count);

    if (bloc < 128) { count = size / 128 - 1; bloc = 128; }

    // iterate over requests
    {
      request_handler_t handler(_iend, bloc, _pred);

      replied_count =
	kaapi_utils::foreach_request
	(
	 victim_stack, task,
	 count, request,
	 handler, this
	 );

      // mute victim state after processing
      _iend = handler.local_end;

      kaapi_assert_debug(_iend - _ibeg > 0);
    }

  finish_splitter:
    {
      // fail the remaining requests

      const int remaining_count = total_count - replied_count;

      if (remaining_count)
	{
	  kaapi_utils::fail_requests
	    (
	     victim_stack,
	     task,
	     remaining_count,
	     request + replied_count
	     );
	}
    }

    // all requests have been replied to
    return total_count;
  }


#if 0 // TODO_REDUCER
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
#endif // TODO_REDUCER

};


/** Adaptive count_if
*/
template<class RandomAccessIterator, class Predicate>
void CountIFStruct<RandomAccessIterator, Predicate>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
    kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     _local_count_if +=std::count_if(_ibeg, nano_iend, _pred);
     _ibeg +=unit_size;

#warning "TODO_REDUCER"
#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint( _sc, 0 )) return ;
#endif
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
#warning "TODO_REDUCER"
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their preds which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Predicate>
typename std::iterator_traits<RandomAccessIterator>::difference_type
   count_if(RandomAccessIterator begin, RandomAccessIterator end, Predicate pred)
{
  CountIFStruct<RandomAccessIterator, Predicate> work( begin, end, pred, 0);
  kaapi_utils::start_adaptive_task(&work);
  return work.get_count_if();
}
#endif
