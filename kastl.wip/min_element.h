/*
 *  test_min_element.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Updated by FLM on decembre 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_MIN_ELEMENT_H
#define _XKAAPI_MIN_ELEMENT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include <functional>

template<class RandomAccessIterator, class Compare>
  RandomAccessIterator 
  min_element ( RandomAccessIterator begin, RandomAccessIterator end, Compare comp);

template<class RandomAccessIterator>
  RandomAccessIterator
  min_element ( RandomAccessIterator begin, RandomAccessIterator end);

/** Stucture of a work for min_element
*/
template<class RandomAccessIterator, class Compare>
class Min_Element_Struct {
public:
  typedef Min_Element_Struct<RandomAccessIterator, Compare> Self_t;

  /* cstor */
  Min_Element_Struct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Compare comp,
    RandomAccessIterator min_element_pos
  ) : _ibeg(ibeg), _iend(iend), _comp(comp), _min_element_pos(min_element_pos) 
  {}
  
  /* do max_element */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* get result */
  RandomAccessIterator get_min_element() {
     return _min_element_pos;
  }
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Compare _comp;
  RandomAccessIterator _min_element_pos;  
  bool _is_master;

  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    size_t bloc;

    request_handler(RandomAccessIterator& _local_end, size_t _bloc)
      : local_end(_local_end), bloc(_bloc)
    {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_is_master = false;
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_comp = self_work->_comp;
      output_work->_min_element_pos = output_work->_ibeg;
      kaapi_assert( output_work->_iend - output_work->_ibeg >0);

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

  /* Called by the victim thread to collect work from one other thread
  */
  static bool reducer
    (
     kaapi_stack_t* stack, kaapi_task_t* self_task,
     Self_t* thief_work, Self_t* victim_work,
     RandomAccessIterator* ibeg,
     RandomAccessIterator* iend
    )
  {
    printf("-- reducer(%p, %p)\n", thief_work, victim_work);

    if (thief_work == NULL)
      return false;

    const bool is_less =
      victim_work->_comp
      (
       *thief_work->_min_element_pos,
       *victim_work->_min_element_pos
      );

    // merge of the two results
    if (is_less)
      victim_work->_min_element_pos =
	thief_work->_min_element_pos;

    if (thief_work->_ibeg == thief_work->_iend)
      return false;

    *ibeg = thief_work->_ibeg;
    *iend = thief_work->_iend;

    return true;
  }

  static void oda_reducer(kaapi_stack_t* , kaapi_task_t*, Self_t* victim_work , Self_t* thief_work)
  {
//     ::printf("oda_reducer(%p, %p)\n", victim_work, thief_work);

    if (victim_work == NULL)
      return ;

    if (thief_work == NULL)
      return ;

    ::printf("t(%lf), v(%lf)\n", *thief_work->_min_element_pos, *victim_work->_min_element_pos);

    const bool is_less =
      victim_work->_comp
      (
       *thief_work->_min_element_pos,
       *victim_work->_min_element_pos
      );

    if (is_less == false)
      return ;

    victim_work->_min_element_pos =
      thief_work->_min_element_pos;
  }

};


/** Adaptive min_element
*/
template<class RandomAccessIterator, class Compare>
void Min_Element_Struct<RandomAccessIterator, Compare>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  ptrdiff_t tmp_size = 0;
  ptrdiff_t unit_size = 512;
  // int count_loop = 1;

  if (_is_master == true)
    usleep(100000);

#if 0 // TODO_REDUCER
 redo_work:
#endif

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
    RandomAccessIterator  tmp = std::min_element(_ibeg, nano_iend, _comp);
    if(_comp(*tmp, *_min_element_pos))
      _min_element_pos = tmp;

    _ibeg +=unit_size;

#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint(stack, task, oda_reducer, this, this))
      return ;
#endif
  }

#if 0 // TODO_REDUCER
  if (kaapi_preempt_nextthief(stack, task, this, reducer, this, &_ibeg, &_iend))
    goto redo_work;
#endif
}

/**
*/
template<class RandomAccessIterator>
RandomAccessIterator
   min_element(RandomAccessIterator begin, RandomAccessIterator end)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;

  Min_Element_Struct<RandomAccessIterator, std::less<value_type> >
    work( begin, end, std::less<value_type>(), begin);

  work._is_master = true;

  kaapi_utils::start_adaptive_task(&work);

  RandomAccessIterator i = work.get_min_element();

  ::printf("[master] min_element == %lf\n", *i);

  return i;
}


#endif
