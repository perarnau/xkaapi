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

  /* Called by the victim thread to collect work from one other thread
  */
  static int reducer
  (
   kaapi_stack_t* stack,
   kaapi_task_t* task,
   void* thief_data,
   void* victim_data
  )
  {
    const Self_t* const thief_work = static_cast<const Self_t*>(thief_data);
    Self_t* const victim_work = static_cast<Self_t*>(victim_data);

    kaapi_trace("reducer %lf, %lf",
		victim_work->_local_accumulate,
		thief_work->_local_accumulate);

    // merge of the two results
    victim_work->_local_accumulate =
      victim_work->_op
      (
       victim_work->_local_accumulate,
       thief_work->_local_accumulate
      );

    victim_work->_ibeg = thief_work->_ibeg;
    victim_work->_iend = thief_work->_iend;

    // always return 1 so that the
    // victim knows about preemption
    return 1;
  }
};


/** Adaptive accumulate
*/
template<class RandomAccessIterator, class T, class BinOp>
void AccumulateStruct<RandomAccessIterator, T, BinOp>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  RandomAccessIterator nano_iend;
  ptrdiff_t unit_size = 512;
  ptrdiff_t tmp_size;

  kaapi_trace("accumulate");

 complete_work:
  kaapi_trace("complete_work");

  while (_iend != _ibeg)
  {
    kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );

    tmp_size = _iend - _ibeg;

    if (tmp_size < unit_size)
      {
	unit_size = tmp_size;
	nano_iend = _iend;
      }
    else
      {
	nano_iend = _ibeg + unit_size;
      }

    // sequential computation
    _local_accumulate = std::accumulate(_ibeg, nano_iend, _local_accumulate, _op);

    _ibeg += unit_size;

    if (kaapi_preemptpoint(stack, task, NULL, this, sizeof(Self_t)))
      {
	// has been preempted

	kaapi_trace("preempted (%lf)", _local_accumulate);

	return ;
      }
  }

  // reduce thief results

 next_thief:
  kaapi_trace("nextthief");

  if (!kaapi_preempt_nextthief(stack, task, NULL, reducer, this))
  {
    kaapi_trace("!nextthief");

    // nothing preempted, we are done
    return ;
  }

  // a thief has been preempted. if it
  // finished the work, iterators are
  // the same and we must test for a
  // new thief prior to complete_work.

  if (_ibeg == _iend)
    goto next_thief;

  // a thief has been preempted and it
  // didnot finish its work. complete.

  goto complete_work;
}


/**
*/
template<class RandomAccessIterator, class T >
T accumulate(RandomAccessIterator begin, RandomAccessIterator end, T init)
{
  typedef AccumulateStruct<RandomAccessIterator, T, std::plus<T> > Self_t;
  Self_t work( begin, end, init, std::plus<T>() );
  kaapi_utils::start_adaptive_task<Self_t>(&work);
  return work.get_accumulate();
}

#endif
