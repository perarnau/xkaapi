/*
 *  test_inner_product.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Created by FLM on decembre 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_INNER_PRODUCT_H
#define _XKAAPI_INNER_PRODUCT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include<numeric>
#include<functional>

template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
 T  inner_product ( RandomAccessIterator begin, RandomAccessIterator end, 
                    RandomAccessIterator2 begin2, T init, BinOp op, BinOp2 op2);

template<class RandomAccessIterator, class RandomAccessIterator2, class T>
 T  inner_product ( RandomAccessIterator begin, RandomAccessIterator end,
                 RandomAccessIterator2 begin2, T init);

/** Stucture of a work for inner_product
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
class InnerProductStruct {
public:

  typedef InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2> Self_t;

  /* cstor */
  InnerProductStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    RandomAccessIterator2 ibeg2,
    T init,
    BinOp op,
    BinOp2 op2,
    bool is_master = true) : _ibeg(ibeg), _iend(iend), _ibeg2(ibeg2), _local_inner_product(init),
                             _op(op), _op2(op2), _is_master(is_master) 
  {}
  
  /* do inner_product */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* get result */
  T get_inner_product() {
     return _local_inner_product;
  }
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  RandomAccessIterator2 _ibeg2;
  T _local_inner_product; 
  BinOp _op;
  BinOp2 _op2;
  bool _is_master; 


  // request handler
  struct request_handler
  {
    RandomAccessIterator local_end;
    size_t bloc;
    RandomAccessIterator ibeg;
    RandomAccessIterator2 ibeg2;

    request_handler(RandomAccessIterator& _local_end,
		    size_t _bloc,
		    RandomAccessIterator _ibeg,
		    RandomAccessIterator2 _ibeg2)
      : local_end(_local_end),
	bloc(_bloc),
	ibeg(_ibeg),
	ibeg2(_ibeg2)
    {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = local_end - bloc;
      output_work->_ibeg2 = ibeg2 + (output_work->_ibeg - ibeg);
      local_end -= bloc;
      output_work->_local_inner_product = 0;
      output_work->_op = self_work->_op;
      output_work->_op2 = self_work->_op2;
      output_work->_is_master = false;

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
      request_handler_t handler(_iend, bloc, _ibeg, _ibeg2);

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
                    InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* victim_data )
  {
    InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* thief_work = 
      (InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* )thief_data;

    InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* victim_work =
      (InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* )victim_data;


   //merge of the two results
   if(thief_work->_local_inner_product!=T(0))    
     victim_work->_local_inner_product  = victim_work->_op(victim_work->_local_inner_product,
                                                        thief_work->_local_inner_product);

    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBINNER_PRODUCT)
      victim_work->_local_inner_product = std::inner_product(thief_work->_ibeg, thief_work->_iend, 
                                        thief_work->_ibeg2, victim_work->_local_inner_product, victim_work->_op, 
                                        victim_work->_op2);
#else

      InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_ibeg2,
              victim_work->_local_inner_product,
              thief_work->_op,
              thief_work->_op2 
            );
      work.doit();

      victim_work->_local_inner_product = work._local_inner_product;
#endif
    }
  }
#endif // TODO_REDUCER

};


/** Adaptive inner_product
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
void InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
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
    if(_is_master) _local_inner_product = std::inner_product(_ibeg, nano_iend, _ibeg2, _local_inner_product, 
                            _op, _op2);
    else {
     RandomAccessIterator first = _ibeg;
     RandomAccessIterator last  = nano_iend;
     RandomAccessIterator first2 = _ibeg2;
     T init = *first++;
         while ( first!=last ) init = _op(init, _op2(*first++, *first2++));
         _local_inner_product = (_local_inner_product==T(0))?init: _op(_local_inner_product, init);
    }
    _ibeg +=unit_size;
    _ibeg2 +=unit_size;

#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint( _sc, 0 )) return ;
#endif
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  // TODO_REDUCER
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their inits which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class T>
T inner_product(RandomAccessIterator begin, RandomAccessIterator end,
		RandomAccessIterator2 begin2, T init)
{
  typedef std::plus<T> BinOp1_t;
  typedef std::multiplies<T> BinOp2_t;
  
  InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp1_t, BinOp2_t>
  work( begin, end, begin2, init, std::plus<T>(), std::multiplies<T>());

  kaapi_utils::start_adaptive_task(&work);
  return work.get_inner_product();
}

#endif
