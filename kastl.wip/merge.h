/*
 *  test_merge.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Created by FLM on decembre 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_MERGE_H
#define _XKAAPI_MERGE_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include<numeric>
#include<functional>

template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
RandomAccessIterator3  merge ( RandomAccessIterator begin, 
                        RandomAccessIterator end, RandomAccessIterator2 begin2, RandomAccessIterator2 end2,
                        RandomAccessIterator3 res, Compare comp);

template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3>
RandomAccessIterator3  merge ( RandomAccessIterator begin, 
                  RandomAccessIterator end, RandomAccessIterator2 begin2, RandomAccessIterator2 end2, 
                  RandomAccessIterator3 res);

/** Stucture of a work for merge
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
class MergeStruct {
public:
  typedef MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>  Self_t;

  /* cstor */
  MergeStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    RandomAccessIterator2 ibeg2,
    RandomAccessIterator2 iend2,
    RandomAccessIterator3 res,
    Compare comp) : _ibeg(ibeg), _iend(iend), _ibeg2(ibeg2), _iend2(iend2), _res(res),  _comp(comp)
  {}
  
  /* do merge */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  RandomAccessIterator2 _ibeg2;
  RandomAccessIterator2 _iend2;
  RandomAccessIterator3 _res;
  Compare _comp;

  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    RandomAccessIterator2 local_end2;
    size_t bloc;
    ptrdiff_t size1;
    ptrdiff_t size2;
    ptrdiff_t pargrain;

    request_handler(RandomAccessIterator& _local_end,
		    RandomAccessIterator2& _local_end2,
		    size_t _bloc,
		    ptrdiff_t _size1,
		    ptrdiff_t _size2,
		    ptrdiff_t _pargrain)
      : local_end(_local_end),
	local_end2(_local_end2),
	bloc(_bloc),
	size1(_size1),
	size2(_size2),
	pargrain(_pargrain)
    {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      size1 = local_end - self_work->_ibeg;
      size2 = local_end2 - self_work->_ibeg2;

      if ((size1 < pargrain)&&(size2 < pargrain))
	{
	  // stop request processing
	  return false;
	}

      output_work->_iend = local_end;

      if (size1 > size2) {
        
	RandomAccessIterator mid = local_end - (size1/2);
	RandomAccessIterator2  split2 = std::lower_bound(self_work->_ibeg2, local_end2, *mid, self_work->_comp);
 
	output_work->_ibeg = mid;
	local_end = mid;
	output_work->_ibeg2 = split2;
	output_work->_iend2 = local_end2;
	local_end2 = split2;

      } else {

        RandomAccessIterator2 mid2 = local_end2 - (size2/2);
        RandomAccessIterator  split1 = std::upper_bound(self_work->_ibeg, local_end, *mid2, self_work->_comp);

        output_work->_ibeg = split1;
        local_end = split1;
        output_work->_ibeg2 = mid2;
        output_work->_iend2 = local_end2;
        local_end2 = mid2;
      }

      output_work->_res = self_work->_res + ((local_end - self_work->_ibeg) + (local_end2 - self_work->_ibeg2));
      output_work->_comp   = self_work->_comp;

      return true;
    }
  };

  typedef struct request_handler request_handler_t;

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    const int total_count = count;
    int replied_count = 0;
#warning "bloc used uninitialized"
    size_t bloc;  
    ptrdiff_t size1 = _iend - _ibeg;
    ptrdiff_t size2 = _iend2 - _ibeg2;
    ptrdiff_t pargrain = 32;

    /* threshold should be defined (...) */
    if ((size1 < pargrain)&&(size2 < pargrain))
      goto finish_splitter;

    // iterate over requests
    {
      request_handler_t handler(_iend, _iend2, bloc, size1, size2, pargrain);

      replied_count =
	kaapi_utils::foreach_request
	(
	 victim_stack, task,
	 count, request,
	 handler, this
	 );

      // mute victim state after processing
      _iend  = handler.local_end;
      _iend2  = handler.local_end2;

      kaapi_assert( _iend - _ibeg >=0);
      kaapi_assert( _iend2 - _ibeg2 >=0);
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
                    MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, 
                    Compare>* victim_data )
  {
    MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>* thief_work =
     (MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>*) thief_data;


   //std::cout << "thief_work->_iend-thief_work->_ibeg=" << thief_work->_iend-thief_work->_ibeg << std::endl;
   //std::cout << "thief_work->_iend2-thief_work->_ibeg2=" << thief_work->_iend2-thief_work->_ibeg2 << std::endl;
 
    if ((thief_work->_ibeg != thief_work->_iend) || (thief_work->_ibeg2 != thief_work->_iend2))
    {
    #if defined(SEQ_SUBMERGE)
      std::merge(thief_work->_ibeg, thief_work->_iend, thief_work->_ibeg2, thief_work->_iend2,
                 thief_work->_res, thief_work->_comp );
    #else
      MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>
        work( sc,
              thief_work->_ibeg,
              thief_work->_iend,
              thief_work->_ibeg2,
              thief_work->_iend2,
              thief_work->_res,
              thief_work->_comp);
      work.doit();
     #endif
    }
  }
#endif // TODO_REDUCER

};

/** Adaptive merge
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, 
         class Compare>
void MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend  =  _ibeg;
  RandomAccessIterator2 nano_iend2 =  _ibeg2; 


  /* amount of work per iteration of the nano loop */
  ptrdiff_t unit_size  = 512;
  ptrdiff_t unit_size2 = 512;
  ptrdiff_t tmp_size;

  while ( (_iend != _ibeg) || (_iend2 != _ibeg2))
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );

    /* nano computation of range1 is finished*/
    if((_iend != _ibeg) && ((_ibeg == nano_iend) || (_iend < nano_iend))) {
       tmp_size = _iend-_ibeg;
       if(tmp_size < unit_size ) {
         unit_size = tmp_size; nano_iend = _iend;
       } else {
       nano_iend = _ibeg + unit_size;
       }
    }

    /* nano computation of range2 is finished*/
    if((_iend2 != _ibeg2) && ((_ibeg2 == nano_iend2) || (_iend2 < nano_iend2))) {
       tmp_size = _iend2-_ibeg2;
       if(tmp_size < unit_size2 ) {
         unit_size2 = tmp_size; nano_iend2 = _iend2;
       } else {
       nano_iend2 = _ibeg2 + unit_size;
       }
    }

    /* sequential computation */ 
    if(_ibeg >= nano_iend) { 
       _res = std::copy(_ibeg2, nano_iend2, _res);
       _ibeg2 = nano_iend2;
    }
    else if(_ibeg2 >= nano_iend2) {
       _res = std::copy(_ibeg, nano_iend, _res);
        _ibeg = nano_iend;
    }
    else {
          while((_ibeg!=nano_iend) && (_ibeg2!=nano_iend2)) {
             if(_comp(*_ibeg2, *_ibeg))  *_res++ = *_ibeg2++;
             else *_res++ = *_ibeg++;
          }
    }
 
    //TG a voir apres if (kaapi_preemptpoint( _sc, 0 )) return ;
  }
}


/**
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3>
RandomAccessIterator3 merge(RandomAccessIterator begin, RandomAccessIterator end,
			    RandomAccessIterator2 begin2, RandomAccessIterator2 end2, 
			    RandomAccessIterator3 res)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;

  MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, std::less<val_t> >
    work(begin, end, begin2, end2, res, std::less<val_t>());
  kaapi_utils::start_adaptive_task(&work);
  return res + ((end-begin) + (end2-begin2));
}

#endif
