/*
 *  test_max_element.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Updated by FLM on decembre 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_MAX_ELEMENT_H
#define _XKAAPI_MAX_ELEMENT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include <functional>

template<class RandomAccessIterator, class Compare>
  RandomAccessIterator 
  max_element ( RandomAccessIterator begin, RandomAccessIterator end, Compare comp);

template<class RandomAccessIterator>
  RandomAccessIterator
  max_element ( RandomAccessIterator begin, RandomAccessIterator end);

/** Stucture of a work for max_element
*/
template<class RandomAccessIterator, class Compare>
class Max_Element_Struct {
public:
  typedef Max_Element_Struct<RandomAccessIterator, Compare> Self_t;

  /* cstor */
  Max_Element_Struct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Compare comp,
    RandomAccessIterator max_element_pos
  ) : _ibeg(ibeg), _iend(iend), _comp(comp), _max_element_pos(max_element_pos) 
  {}
  
  /* do max_element */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* get result */
  RandomAccessIterator get_max_element() {
     return _max_element_pos;
  }
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Compare _comp;
  RandomAccessIterator _max_element_pos;  

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
      output_work->_iend = local_end;
      output_work->_ibeg = local_end-bloc;
      local_end -= bloc;
      output_work->_comp = self_work->_comp;
      output_work->_max_element_pos = output_work->_ibeg;
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


#if 0 // TODO_REDUCER
  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, 
                       Max_Element_Struct<RandomAccessIterator, Compare>* victim_data )
  {
    Max_Element_Struct<RandomAccessIterator, Compare>* thief_work = 
      (Max_Element_Struct<RandomAccessIterator, Compare>* )thief_data;

    Max_Element_Struct<RandomAccessIterator, Compare>* victim_work =
      (Max_Element_Struct<RandomAccessIterator, Compare>* )victim_data;

    //merge of the two results
    if(victim_work->_comp(*victim_work->_max_element_pos, *thief_work->_max_element_pos))
      victim_work->_max_element_pos = thief_work->_max_element_pos;

    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBCOUNT)
      RandomAccessIterator  tmp=std::max_element(thief_work->_ibeg, thief_work->_iend, thief_work->_comp);
      if(_comp(*tmp, *_max_element_pos)) _max_element_pos = tmp;
#else
      Max_Element_Struct<RandomAccessIterator, Compare> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_comp,
              victim_work->_max_element_pos 
            );
      work.doit();

      victim_work->_max_element_pos = work._max_element_pos;
#endif
    }
  }
#endif // TODO_REDUCER

};


/** Adaptive max_element
*/
template<class RandomAccessIterator, class Compare>
void Max_Element_Struct<RandomAccessIterator, Compare>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
    RandomAccessIterator  tmp=std::max_element(_ibeg, nano_iend, _comp);
    if(_comp(*_max_element_pos, *tmp)) _max_element_pos = tmp;
     _ibeg +=unit_size;

#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint( _sc, 0 )) return ;
#endif // TODO_REDUCER
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  // TODO_REDUCER
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their comps which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator>
RandomAccessIterator
   max_element(RandomAccessIterator begin, RandomAccessIterator end)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;

  Max_Element_Struct<RandomAccessIterator, std::less<value_type> >
    work( begin, end, std::less<value_type>(), begin);
  kaapi_utils::start_adaptive_task(&work);
  return work.get_max_element();
}


#endif
