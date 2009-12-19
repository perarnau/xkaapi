/*
 *  test_stable_sort.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Updated by FLM on decembre 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_STABLE_SORT_H
#define _XKAAPI_STABLE_SORT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include<numeric>
#include<functional>

typedef enum { SORTING, MERGE_FINISHED, MERGE_NOT_FINISHED} SORT_STATE;

template<class RandomAccessIterator, class Compare>
void  stable_sort ( RandomAccessIterator begin,
		    RandomAccessIterator end,
		    Compare comp);

/** Stucture of a work for stable_sort
*/
template<class RandomAccessIterator, class Compare>
class StableSortStruct {
public:
  typedef StableSortStruct<RandomAccessIterator, Compare> Self_t;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  typedef val_t* ptr_type;
  /* cstor */
  StableSortStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    ptr_type buf,
    Compare comp) : _ibeg(ibeg), _init_beg(ibeg), _iend(iend),
		    _buf(buf), _comp(comp)
  {}
  
  /* do stable_sort */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* do sort_loop */
  void doit_sort_loop(RandomAccessIterator ibeg,
		      RandomAccessIterator iend,
		      ptr_type buf,
		      ptrdiff_t step_size,
		      Compare comp);

  RandomAccessIterator  _ibeg;
  RandomAccessIterator _init_beg;
  RandomAccessIterator  _iend;
  ptr_type _buf;
  Compare _comp;

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
      output_work->_init_beg = output_work->_ibeg;
      local_end -= bloc;
      output_work->_buf = self_work->_buf + (output_work->_ibeg - self_work->_ibeg);
      output_work->_comp = self_work->_comp;

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
                     StableSortStruct<RandomAccessIterator, Compare>* victim_data)
  {

    StableSortStruct<RandomAccessIterator, Compare>* thief_work =
      (StableSortStruct<RandomAccessIterator, Compare>* )thief_data;

    StableSortStruct<RandomAccessIterator, Compare>* victim_work =
      (StableSortStruct<RandomAccessIterator, Compare>* )victim_data;

    std::cout << "victim_work->_iend-victim_work->_init_beg=" << victim_work->_iend-victim_work->_init_beg << std::endl;
    std::cout << "thief_work->_iend-thief_work->_init_beg=" << thief_work->_iend-thief_work->_init_beg << std::endl;

    const ptrdiff_t len = (victim_work->_iend-victim_work->_init_beg)+(thief_work->_iend-thief_work->_init_beg);
    std::cout << "len=" << len << std::endl;

    RandomAccessIterator victim_middle = victim_work->_iend-(victim_work->_iend-victim_work->_init_beg)/2;
    RandomAccessIterator thief_middle  = thief_work->_iend-(thief_work->_iend-thief_work->_init_beg)/2;  
    ptr_type buf1 = victim_work->_buf;
    ptr_type buf1_last = buf1+(victim_middle-victim_work->_init_beg)+(thief_middle-thief_work->_init_beg);
    ptr_type buf2 = buf1_last;
    ptr_type buf2_last = buf1 + len;
 
    std::merge(victim_work->_init_beg, victim_middle, thief_work->_init_beg, thief_middle, buf1, 
	       victim_work->_comp);
    std::merge(victim_middle, victim_work->_iend,  thief_middle, thief_work->_iend, buf2,
	       victim_work->_comp);
    std::merge(buf1, buf1_last, buf2, buf2_last, victim_work->_init_beg, victim_work->_comp); 

    victim_work->_iend = thief_work->_iend;

  }
#endif // TODO_REDUCER

};

/** doit_sort_loop
*/
template<class RandomAccessIterator, class Compare>
void StableSortStruct<RandomAccessIterator, Compare>::doit_sort_loop
(
 RandomAccessIterator ibeg,
 RandomAccessIterator iend,
 ptr_type buf,
 ptrdiff_t step_size,
 Compare comp
)
{

}


/** Adaptive stable_sort
*/
template<class RandomAccessIterator, class Compare>
void StableSortStruct<RandomAccessIterator, Compare>::doit
(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend ;

  /* amount of work per iteration of the nano loop */
  ptrdiff_t unit_size  = 256;
  ptrdiff_t tmp_size;
  ptrdiff_t step_size = unit_size;
  while ( (_iend != _ibeg) )
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

    /** Local sorting */
    std::stable_sort(_ibeg, nano_iend, _comp);
    //std::__insertion_sort(_ibeg, nano_iend _comp);
   _ibeg += unit_size; 
  }

  const ptrdiff_t len = _iend - _init_beg;
  const ptr_type  buf_last = _buf + len;  

  while (step_size < len)
  {
     //doit_sort_loop(ibeg, _iend, _buf, step_size, _comp);
     std::__merge_sort_loop(_init_beg, _iend, _buf, step_size, _comp);
     step_size *= 2;
     //doit_sort_loop(_buf, buf_last, ibeg, step_size, _comp);
     std::__merge_sort_loop(_buf, buf_last, _init_beg, step_size, _comp);
     step_size *= 2;
  }
}

/**
*/
template<class RandomAccessIterator>
void stable_sort(RandomAccessIterator begin, RandomAccessIterator end)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  typedef typename std::iterator_traits<RandomAccessIterator>::pointer ptr_type;

  const ptrdiff_t size = end - begin;
  val_t* buf = new val_t[size];

  StableSortStruct<RandomAccessIterator, std::less<val_t> >
    work( begin, end, buf, std::less<val_t>());

  kaapi_utils::start_adaptive_task(&work);

  delete  [] buf;
}

#endif
