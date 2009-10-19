/*
 *  test_copy.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_COPY_H
#define _CKAAPI_COPY_H
#include "kaapi_adapt.h"
#include <algorithm>


unsigned long int fibo(kaapi_steal_context_t*, int n);

/** Stucture of a work for copy
*/
template<class RandomAccessIterator1, class RandomAccessIterator2>
class FiboStruct {
public:
  /* cstor */
  FiboStruct(
    kaapi_steal_context_t* sc,
    int n) : _sc(sc), _n(n)
  {}
  
  /* do copy */
  void doit();

protected:  
  kaapi_steal_context_t* _sc;
  int _n;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    FiboStruct* w = (FiboStruct)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, )
  {
    int i = 0;
 
    /* threshold should be defined (...) */
    if (is_empty_deque) goto reply_failed;
    
    while (count >0)
    {
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(FiboStruct) 
                                            ) ==0)
        {
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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, FiboStruct<RandomAccessIterator1, RandomAccessIterator2>* victim_work )
  {
    FiboStruct<RandomAccessIterator1, RandomAccessIterator2>* thief_work = 
      (FiboStruct<RandomAccessIterator1, RandomAccessIterator2>* )thief_data;
  };


/** Adaptive fibo
*/
void FiboStruct::doit()
{ 
  /* amount of work per iteration of the nano loop */
  int unit_size = 2;
  int k = *(--ptlocaldeque) ;
  if (k<= 2) localres = 1;
  else  {
    *(ptlocaldeque++) = k-1 ;
    *(ptlocaldeque++) = k-2 ;
    while (ptlocaldeque != localdeque)
    {
      /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
      */
      kaapi_stealpoint( _sc, splitter, localdeque);
 
      if(ptlocaldeque != localdeque) {
        int i = *(--ptlocaldeque) ;
        if (i <= 2) localres += 1;
        else {
          *(ptlocaldeque++) = i-1 ; // push F(i-1)
          *(ptlocaldeque++) = i-2 ; // push F(i-2)
        }
     }
    }
   }
  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
unsigned long int fibo(kaapi_steal_context_t* stealcontext, int n)
{
  kaapi_steal_context_initpush( stealcontext );
  FiboStruct work(n);
  work.doit();
 return work.get_result();
}
#endif
