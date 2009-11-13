/*
 *  transform.cpp
 *  xkaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_TRANSFORM_H
#define _CKAAPI_TRANSFORM_H
#include "kaapi.h"
#include <algorithm>


template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( kaapi_steal_context_t* stealcontext, InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );

/** Stucture of a work for transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
class TransformStruct {
public:
  /* cstor */
  TransformStruct(
    InputIterator ibeg,
    InputIterator iend,
    OutputIterator obeg,
    UnaryOperator  op
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _obeg(obeg), _op(op) 
  {}
  
  /* do transform */
  void doit(kaapi_stack_t* stack);

  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

protected:  
  kaapi_stack_t* _sc;
  InputIterator  _ibeg;
  InputIterator  _iend;
  OutputIterator _obeg;
  UnaryOperator  _op;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_stack_t* stack, void* data)
  {
    TransformStruct<InputIterator, OutputIterator, UnaryOperator>* w = (TransformStruct<InputIterator, OutputIterator, UnaryOperator>*)data;
    w->doit(stack);
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_task_t* task, int count, kaapi_steal_request_t** request, 
                        InputIterator ibeg, InputIterator& iend, OutputIterator obeg, UnaryOperator op
                      )
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    InputIterator local_end = iend;

    TransformStruct<InputIterator, OutputIterator, UnaryOperator>* output_work =0;

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
                                              sizeof(TransformStruct<InputIterator, OutputIterator, UnaryOperator>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_obeg = obeg + (output_work->_ibeg - ibeg);
          kaapi_assert( output_work->_iend - output_work->_ibeg >0);
          output_work->_op   = op;

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
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, TransformStruct<InputIterator, OutputIterator, UnaryOperator>* victim_work )
  {
    TransformStruct<InputIterator, OutputIterator, UnaryOperator>* thief_work = 
      (TransformStruct<InputIterator, OutputIterator, UnaryOperator>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::transform( thief_work->_ibeg, thief_work->_iend, thief_work->_obeg, thief_work->_op );
#else
      TransformStruct<InputIterator, OutputIterator, UnaryOperator> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_obeg, 
              victim_work->_op);
      work.doit();
#endif
    }
  }
};


/** Adaptive transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void TransformStruct<InputIterator,OutputIterator,UnaryOperator>::doit(kaapi_stack_t* stack)
{
  if (stack ==0) stack = kaapi_self_stack();
  
  /* will receive & process steal request */
  kaapi_task_t task;
  kaapi_task_init( &task, KAAPI_TASK_F_ADAPTIVE, KAAPI_EVENT_STEAL );

  /* local iterator for the nano loop */
  InputIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 512;
  int tmp_size = 0;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
#if 0
    kaapi_stealpoint( &stack, &task, &splitter, _ibeg, _iend, _obeg, _op );
#else
    if (kaapi_stealpoint_isactive( &stack, &task ))
    { 
      splitter( task, _ibeg, _iend, _obeg, _op );
    }
#elif 0
    if (kaapi_point_isactive( &stack, &task, KAAPI_EVENT_STEAL ))
    { 
      splitter( task, _ibeg, _iend, _obeg, _op );
    }
#endif

    tmp_size =  _iend-_ibeg;
    if (unit_size > tmp_size) { unit_size = tmp_size; nano_iend = _iend; }
    else nano_iend = _ibeg + unit_size;
    
    /* sequential computation: push task action in order to allows steal at this point while I'm doing seq computation */
    kaapi_task_push(stack, task, &splitter);
    _obeg = std::transform( _ibeg, nano_iend, _obeg, _op );

    /* return from sequential computation: pop task action in order to disable any steal at this point while I'm doing seq computation */
    kaapi_task_pop(stack);

    _ibeg += unit_size;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( task, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op )
{
  TransformStruct<InputIterator, OutputIterator, UnaryOperator> work( begin, end, to_fill, op);
  work.doit();

}
#endif
