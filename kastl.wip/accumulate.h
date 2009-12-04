/*
 *  test_accumulate.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Created by FLM on december 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_ACCUMULATE_H
#define _XKAAPI_ACCUMULATE_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include <numeric>
#include <functional>

template<class RandomAccessIterator, class T>
 T  accumulate ( RandomAccessIterator begin, RandomAccessIterator end,
                 T init);

/** Stucture of a work for accumulate
*/
template<class RandomAccessIterator, class T, class BinOp>
class AccumulateStruct {
public:
  typedef AccumulateStruct<RandomAccessIterator, T, BinOp> Self_t;

  /* cstor */
  AccumulateStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    T init,
    BinOp op,
    bool is_master = true) : _ibeg(ibeg), _iend(iend), _local_accumulate(init),
                             _op(op), _is_master(is_master) 
  {}
  
  /* do accumulate */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* get result */
  T get_accumulate() {
     return _local_accumulate;
  }
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  T _local_accumulate; 
  BinOp _op;
  bool _is_master; 

  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    size_t bloc;
    BinOp op;

    request_handler(RandomAccessIterator& _local_end, size_t _bloc, BinOp _op)
    : local_end(_local_end), bloc(_bloc), op(_op) {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_local_accumulate = 0;
      output_work->_op = op;
      output_work->_is_master = false;

      kaapi_assert_debug( output_work->_iend - output_work->_ibeg > 0);

      return true;
    }
  };

  typedef struct request_handler request_handler_t;

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task,
		int count, kaapi_request_t* request )
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
      request_handler_t handler(_iend, bloc, _op);

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
#endif // TODO_REDUCER
};


/** Adaptive accumulate
*/
template<class RandomAccessIterator, class T, class BinOp>
void AccumulateStruct<RandomAccessIterator, T, BinOp>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
    kaapi_stealpoint( stack, task, &kaapi_utils::static_splitter<Self_t> );

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
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( stack, task );

#warning "TODO_REDUCER"

  /* Here the thiefs have finish the computation and returns their inits which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class T >
T accumulate(RandomAccessIterator begin, RandomAccessIterator end, T init)
{
  AccumulateStruct<RandomAccessIterator, T, std::plus<T> > work( begin, end, init, std::plus<T>() );
  kaapi_utils::start_adaptive_task(&work);
  return work.get_accumulate();
}

#endif
