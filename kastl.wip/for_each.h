/*
 *  test_for_each.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Updated by FLM on december 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_FOR_EACH_STRUCT_H
#define _XKAAPI_FOR_EACH_STRUCT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>


template<class RandomAccessIterator, class Function>
void for_each (RandomAccessIterator begin, RandomAccessIterator end, Function f);

/** Stucture of a work for for_each
*/
template<class RandomAccessIterator, class Function>
class ForEachStruct {
public:

  typedef ForEachStruct<RandomAccessIterator, Function> Self_t;

  /* cstor */
  ForEachStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Function f
  ) : _ibeg(ibeg), _iend(iend), _f(f) 
  {}
  
  /* do for_each */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;

  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Function _f;
  
  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    size_t bloc;
    Function f;

    request_handler(RandomAccessIterator& _local_end, size_t _bloc, Function& _f)
    : local_end(_local_end), bloc(_bloc), f(_f) {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_f = f;
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
      request_handler_t handler(_iend, bloc, _f);

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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, ForEachStruct<RandomAccessIterator, Function>* victim_work )
  {
    ForEachStruct<RandomAccessIterator, Function>* thief_work = 
      (ForEachStruct<RandomAccessIterator, Function>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::for_each(thief_work->_ibeg, thief_work->_iend, thief_work->_f);
#else
      ForEachStruct<RandomAccessIterator, Function> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_f 
            );
      work.doit();
#endif
    }
  }
#endif // TODO_REDUCER

};


/** Adaptive for_each
*/
template<class RandomAccessIterator, class Function>
void ForEachStruct<RandomAccessIterator,Function>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
     std::for_each(_ibeg, nano_iend, _f);
     _ibeg +=unit_size;

#warning "TODO_REDUCER"
#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint( _sc, 0 )) return ;
#endif
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
#warning "TODO_REDUCER"
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Function>
void for_each(RandomAccessIterator begin, RandomAccessIterator end, Function f)
{
  ForEachStruct<RandomAccessIterator, Function> work( begin, end, f);
  kaapi_utils::start_adaptive_task(&work);
}
#endif
