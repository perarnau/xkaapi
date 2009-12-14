/*
 *  test_count.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_COUNT_H
#define _XKAAPI_COUNT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>


template<class RandomAccessIterator, class T>
 typename std::iterator_traits<RandomAccessIterator>::difference_type
  count ( RandomAccessIterator begin, RandomAccessIterator end, const T& value);

/** Stucture of a work for count
*/
template<class RandomAccessIterator, class T>
class CountStruct {
public:

  typedef CountStruct<RandomAccessIterator, T> Self_t;
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type Distance_type;
  /* cstor */
  CountStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    const T& value,
    Distance_type local_count
  ) : _ibeg(ibeg), _iend(iend), _value(value), _local_count(local_count) 
  {}
  
  /* do count */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* get result */
  Distance_type get_count() {
     return _local_count;
  }
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  T _value;
  Distance_type _local_count;  

  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    size_t bloc;
    T value;

    request_handler(RandomAccessIterator& _local_end, size_t _bloc, const T& _value)
    : local_end(_local_end), bloc(_bloc), value(_value) {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_value = value;
      output_work->_local_count = 0;
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
      request_handler_t handler(_iend, bloc, _value);

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
#endif // TODO_REDUCER

};


/** Adaptive count
*/
template<class RandomAccessIterator, class T>
void CountStruct<RandomAccessIterator,T>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
    kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );

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
#warning "TODO_REDUCER"
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class T>
typename std::iterator_traits<RandomAccessIterator>::difference_type
   count(RandomAccessIterator begin, RandomAccessIterator end, const T& value)
{
  CountStruct<RandomAccessIterator, T> work( begin, end, value, 0);
  kaapi_utils::start_adaptive_task(&work);
  return work.get_count();
}
#endif
