/*
 *  test_copy.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Updated by FLM on december 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_COPY_H
#define _XKAAPI_COPY_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>


template<class RandomAccessIterator1, class RandomAccessIterator2>
void copy ( RandomAccessIterator1 begin, RandomAccessIterator1 end, 
            RandomAccessIterator2 res);

/** Stucture of a work for copy
*/
template<class RandomAccessIterator1, class RandomAccessIterator2>
class CopyStruct {
public:

  typedef CopyStruct<RandomAccessIterator1, RandomAccessIterator2> Self_t;

  /* cstor */
  CopyStruct(
    RandomAccessIterator1 ibeg,
    RandomAccessIterator1 iend,
    RandomAccessIterator2 obeg
  ) : _ibeg(ibeg), _iend(iend), _obeg(obeg) 
  {}
  
  /* do copy */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  typedef typename std::iterator_traits<RandomAccessIterator1>::value_type value_type;

  RandomAccessIterator1  _ibeg;
  RandomAccessIterator1  _iend;
  RandomAccessIterator2 _obeg;

  // request handler
  struct request_handler
  {
    RandomAccessIterator1 local_end;
    RandomAccessIterator1 ibeg;
    RandomAccessIterator2 obeg;
    size_t bloc;

    request_handler
    (
     RandomAccessIterator1& _local_end,
     RandomAccessIterator1 _ibeg,
     RandomAccessIterator2 _obeg,
     size_t _bloc
     ) : local_end(_local_end),
	 ibeg(_ibeg),
	 obeg(_obeg),
	 bloc(_bloc)
    {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_obeg = obeg + (output_work->_ibeg - ibeg);
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
      request_handler_t handler(_iend, _ibeg, _obeg, bloc);

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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, CopyStruct<RandomAccessIterator1, RandomAccessIterator2>* victim_work )
  {
    CopyStruct<RandomAccessIterator1, RandomAccessIterator2>* thief_work = 
      (CopyStruct<RandomAccessIterator1, RandomAccessIterator2>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::copy(thief_work->_ibeg, thief_work->_iend, thief_work->_obeg);
#else
      CopyStruct<RandomAccessIterator1, RandomAccessIterator2> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_obeg 
            );
      work.doit();
#endif
    }
  }
#endif // TODO_REDUCER

};


/** Adaptive copy
*/
template<class RandomAccessIterator1, class RandomAccessIterator2>
void CopyStruct<RandomAccessIterator1,RandomAccessIterator2>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
    kaapi_stealpoint( stack, task, &kaapi_utils::static_splitter<Self_t> );

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     std::copy(_ibeg, nano_iend, _obeg);
     _ibeg +=unit_size;
     _obeg +=unit_size;

#if 0 // TODO_REDUCER
     if (kaapi_preemptpoint( stack, task, NULL, NULL )) return ;
#else
# warning "TODO_REDUCER"
#endif // TODO_REDUCER

  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator1, class RandomAccessIterator2>
void copy (RandomAccessIterator1 begin, RandomAccessIterator1 end, RandomAccessIterator2 res)
{
  CopyStruct<RandomAccessIterator1, RandomAccessIterator2> work( begin, end, res);

  kaapi_utils::start_adaptive_task(&work);
}
#endif
