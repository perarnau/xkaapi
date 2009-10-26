/*
 *  test_inner_product.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_INNER_PRODUCT_H
#define _CKAAPI_INNER_PRODUCT_H
#include "kaapi_adapt.h"
#include <algorithm>
#include<numeric>
#include<functional>

template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
 T  inner_product ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
                    RandomAccessIterator2 begin2, T init, BinOp op, BinOp2 op2);

template<class RandomAccessIterator, class RandomAccessIterator2, class T>
 T  inner_product ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
                 RandomAccessIterator2 begin2, T init);

/** Stucture of a work for inner_product
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
class InnerProductStruct {
public:
  /* cstor */
  InnerProductStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    RandomAccessIterator2 ibeg2,
    T init,
    BinOp op,
    BinOp2 op2,
    bool is_master = true) : _sc(sc), _ibeg(ibeg), _iend(iend), _ibeg2(ibeg2), _local_inner_product(init),
                             _op(op), _op2(op2), _is_master(is_master) 
  {}
  
  /* do inner_product */
  void doit();

  /* get result */
  T get_inner_product() {
     return _local_inner_product;
  }
 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  RandomAccessIterator2 _ibeg2;
  T _local_inner_product; 
  BinOp _op;
  BinOp2 _op2;
  bool _is_master; 
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* w = (InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, RandomAccessIterator2& ibeg2,
                       BinOp op, BinOp2 op2)
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    
    bloc = size / (1+count);
    if (bloc < 128) { count = size/128 -1; bloc = 128; }
    while (count >0)
    {
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(InnerProductStruct<RandomAccessIterator, 
                                              RandomAccessIterator2, T, BinOp, BinOp2>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          output_work->_ibeg2 = ibeg2 + (output_work->_ibeg - ibeg);
          local_end -= bloc;
          output_work->_local_inner_product = 0;
          output_work->_op = op;
          output_work->_op2 = op2;
          output_work->_is_master = false;
          kaapi_assert( output_work->_iend - output_work->_ibeg >0);

          /* reply ok (1) to the request */
          kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
        }
        else {
          /* reply failed (=last 0 in parameter) to the request */
          kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        }
        --count; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  iend  = local_end;
  kaapi_assert( iend - ibeg >0);
  return;
      
reply_failed:
    while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        --count; 
      }
      ++i;
    }
  }


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
};


/** Adaptive inner_product
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
void InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2>::doit()
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
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _ibeg2, _op, _op2);

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

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their inits which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class T, class BinOp, class BinOp2>
T inner_product(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
           RandomAccessIterator2 begin2, T init, BinOp op, BinOp2 op2)
{
  InnerProductStruct<RandomAccessIterator, RandomAccessIterator2, T, BinOp, BinOp2> work( stealcontext, begin, end, 
        begin2, init, op, op2);
  work.doit();

 return work.get_inner_product();
}

template<class RandomAccessIterator, class RandomAccessIterator2, class T>
T inner_product(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
           RandomAccessIterator2 begin2, T init)
{
  kaapi_steal_context_initpush( stealcontext );
 return inner_product(stealcontext, begin, end, begin2, init, std::plus<T>(), std::multiplies<T>());
}

#endif
