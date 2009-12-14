/*
 *  transform.cpp
 *  xkaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_TRANSFORM_H
#define _XKAAPI_TRANSFORM_H
#include "kaapi.h"
#include <algorithm>


template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );

/** Stucture of a work for transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
class TransformStruct {
public:
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;
  
  static InputIterator  beg0;
  static OutputIterator obeg0;

  /* cstor */
  TransformStruct(
    InputIterator ibeg,
    InputIterator iend,
    OutputIterator obeg,
    UnaryOperator  op
  ) : _ibeg(ibeg), _iend(iend), _obeg(obeg), _op(op) 
  {}
  
  /* do transform */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

protected:  
  InputIterator  _ibeg;
  InputIterator  _iend;
  OutputIterator _obeg;
  UnaryOperator  _op;
  
  /* Entry in case of thief execution */
  static void static_thiefentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
// std::cout << "Thief [" << self_work->_ibeg << "," << self_work->_iend << ")= " << self_work->_iend-self_work->_ibeg << std::endl;
    self_work->doit(task, stack);
  }

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    int i = 0;
    int reply_count = 0;
    size_t bloc, size = (_iend - _ibeg);
    InputIterator local_end = _iend;

    Self_t* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    bloc = size / (1+count);
//    std::cout << "Split [" << _ibeg << "," << _iend << ") in " << 1+count << ", bloc=" << bloc << std::endl;
    if (bloc < 128) { count = size/128 -1; bloc = 128; }
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_stack_t* thief_stack = request[i].stack;
        kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
        kaapi_task_init( thief_stack, thief_task, KAAPI_TASK_ADAPTIVE);
        kaapi_task_setbody( thief_task, &static_thiefentrypoint );
        kaapi_task_setargs(thief_task, kaapi_stack_pushdata(thief_stack, sizeof(Self_t)));
        output_work = kaapi_task_getargst(thief_task, Self_t);

        output_work->_iend = local_end;
        output_work->_ibeg = local_end-bloc;
        local_end         -= bloc;
        output_work->_obeg = _obeg + (output_work->_ibeg - _ibeg);
        kaapi_assert_debug( output_work->_iend - output_work->_ibeg >0);
        output_work->_op   = _op;

        kaapi_stack_pushtask( thief_stack );

        /* reply ok (1) to the request */
        kaapi_request_reply( victim_stack, task, &request[i], thief_stack, 1 );
        --count; 
        ++reply_count;
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  _iend  = local_end;
  kaapi_assert_debug( _iend - _ibeg >0);
  return reply_count;
      
reply_failed:
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( victim_stack, task, &request[i], 0, 0 );
        --count; 
        ++reply_count;
      }
      ++i;
    }
    return reply_count;
  }


  static int static_splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
    return self_work->splitter( victim_stack, task, count, request );
  }
};


/** Adaptive transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void TransformStruct<InputIterator,OutputIterator,UnaryOperator>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  InputIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 512;
  int tmp_size  = 0;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal
    */
    kaapi_stealpoint( stack, task, &static_splitter );

    tmp_size =  _iend-_ibeg;
    if (unit_size > tmp_size) { unit_size = tmp_size; nano_iend = _iend; }
    else nano_iend = _ibeg + unit_size;
    
    /* sequential computation: push task action in order to allows steal at this point while I'm doing seq computation */
//    kaapi_task_setaction( task, &static_splitter );
    _obeg = std::transform( _ibeg, nano_iend, _obeg, _op );

    /* return from sequential computation: remove concurrent task action 
       in order to disable any steal at this point while I'm doing seq computation */
//    kaapi_task_getaction( task );

    _ibeg += unit_size;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( stack, task );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function */  
}


template<class InputIterator, class OutputIterator, class UnaryOperator>
InputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::beg0;
template<class InputIterator, class OutputIterator, class UnaryOperator>
OutputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::obeg0;

/**
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op )
{
  TransformStruct<InputIterator, OutputIterator, UnaryOperator>::beg0 = begin;
  TransformStruct<InputIterator, OutputIterator, UnaryOperator>::obeg0 = to_fill;
  
  TransformStruct<InputIterator, OutputIterator, UnaryOperator> work( begin, end, to_fill, op);
  kaapi_stack_t* stack = kaapi_self_stack();
  kaapi_frame_t frame;
  kaapi_stack_save_frame(stack, &frame);

  /* will receive & process steal request */
  kaapi_task_t* task = kaapi_stack_toptask(stack);
  kaapi_task_initadaptive(stack, task, KAAPI_TASK_ADAPT_DEFAULT);
  kaapi_task_setargs(task, &work);
  kaapi_stack_pushtask(stack);

  /* directly execute the task with forking it: correct because no data dependencies */
  work.doit(task, stack);
  
  kaapi_stack_restore_frame(stack, &frame);
}
#endif
