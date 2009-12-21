/*
 *  transform.cpp
 *  xkaapi
 *
 *  Created by TG on 18/02/09.
 *  Updated by FLM on 12/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_TRANSFORM_H
#define _XKAAPI_TRANSFORM_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>


template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );

/** Stucture of a work for transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
class TransformStruct {
public:
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;
  
  /* cstor */
  TransformStruct(
    InputIterator ibeg,
    InputIterator iend,
    OutputIterator obeg,
    UnaryOperator  op
  ) : _ibeg(ibeg), _iend(iend), _obeg(obeg), _op(op) 
  {}
  
  /* do transform */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

  InputIterator  _ibeg;
  InputIterator  _iend;
  OutputIterator _obeg;
  UnaryOperator  _op;

  // request handler
  struct request_handler
  {
    InputIterator local_end;
    size_t bloc;

    request_handler(InputIterator& _local_end, size_t _bloc)
      : local_end(_local_end), bloc(_bloc)
    {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end - bloc;
      local_end         -= bloc;
      output_work->_obeg = self_work->_obeg + (output_work->_ibeg - self_work->_ibeg);
      output_work->_op   = self_work->_op;

      kaapi_assert_debug(output_work->_iend - output_work->_ibeg > 0);

      return true;
    }
  };

  typedef struct request_handler request_handler_t;

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    const size_t size = (_iend - _ibeg);
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
      request_handler_t handler(_iend, bloc);

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

    // there is no need to reduce
    // anything since the output
    // iterator is already filled

    victim_work->_ibeg = thief_work->_ibeg;
    victim_work->_iend = thief_work->_iend;

    // always return 1 so that the
    // victim knows about preemption
    return 1;
  }

};


/** Adaptive transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void TransformStruct<InputIterator,OutputIterator,UnaryOperator>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  InputIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 512;
  int tmp_size  = 0;

 complete_work:
  while (_iend != _ibeg)
  {
    kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );

    tmp_size = _iend - _ibeg;

    if (unit_size > tmp_size)
    {
      unit_size = tmp_size;
      nano_iend = _iend;
    }
    else
    {
      nano_iend = _ibeg + unit_size;
    }
    
#if 0
    kaapi_task_setaction( task, kaapi_utils::static_splitter<Self_t> );
#endif

    _obeg = std::transform(_ibeg, nano_iend, _obeg, _op);

#if 0
    kaapi_task_getaction( task );
#endif

    _ibeg += unit_size;

    if (kaapi_preemptpoint(stack, task, NULL, this, sizeof(Self_t)))
    {
      // has been preempted
      return ;
    }
  }

  // reduce thief results

 next_thief:
  if (!kaapi_preempt_nextthief(stack, task, NULL, reducer, this))
    return ;

  if (_ibeg == _iend)
    goto next_thief;

  goto complete_work;
}

/**
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op )
{
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;

  Self_t work(begin, end, to_fill, op);

  kaapi_utils::start_adaptive_task<Self_t>(&work);
}
#endif
