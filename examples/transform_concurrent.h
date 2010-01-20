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
#include "kaapi++.h"
#include <algorithm>


template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );


int unit_size = 1;

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

  /* Entry in case of thief execution */
  static void static_mainentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
 std::cout << "Main [" << self_work->_ibeg-beg0 << "," << self_work->_iend-beg0 << ")= " << self_work->_iend-self_work->_ibeg << std::endl;
    self_work->doit(task, stack);

    /* definition of the finalization point where all stolen work a interrupt and collected */
    kaapi_finalize_steal( stack, task, 0, 0 );

    /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function */  
  }

protected:  
  InputIterator  _ibeg;
  InputIterator  _iend;
  OutputIterator _obeg;
  UnaryOperator  _op;
  

  /* Entry in case of thief execution */
  static void static_thiefentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
 std::cout << "Thief [" << self_work->_ibeg-beg0 << "," << self_work->_iend-beg0 << ")= " << self_work->_iend-self_work->_ibeg << std::endl;
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

    /* if distance is less than unit_size, the victim will bloc... */
    if (size < (unsigned)unit_size) goto reply_failed;
    size -= unit_size;
    bloc = size / (1+count);

//    std::cout << "Split [" << _ibeg << "," << _iend << ") in " << 1+count << ", bloc=" << bloc << std::endl;
    if (bloc < (unsigned)unit_size) { count = size/unit_size -1; bloc = unit_size; }
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_stack_t* thief_stack = request[i].stack;
        kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
        kaapi_task_init( thief_stack, thief_task, &static_thiefentrypoint, kaapi_stack_pushdata(thief_stack, sizeof(Self_t)), KAAPI_TASK_ADAPTIVE);
        output_work = kaapi_task_getargst(thief_task, Self_t);

        output_work->_iend = local_end;
        output_work->_ibeg = local_end-bloc;
        local_end         -= bloc;
        output_work->_obeg = _obeg + (output_work->_ibeg - _ibeg);
        kaapi_assert_debug( output_work->_iend - output_work->_ibeg >0);
        kaapi_assert_debug( output_work->_ibeg - _ibeg > unit_size);
        output_work->_op   = _op;

        kaapi_stack_pushtask( thief_stack );

        /* reply ok (1) to the request */
        kaapi_request_reply( victim_stack, task, &request[i], thief_stack, 0, 1 );
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
  InputIterator nano_beg;
  OutputIterator nano_obeg;
  
  /* amount of work per iteration of the nano loop */
  int tmp_size  = 0;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       on return _iend may have been updated by the splitter function
    */
    kaapi_stealpoint( stack, task, &static_splitter );

    tmp_size =  _iend-_ibeg;
    if (unit_size > tmp_size) { unit_size = tmp_size; nano_iend = _iend; }
    else nano_iend = _ibeg + unit_size;
    nano_beg = _ibeg; 
    nano_obeg= _obeg;
    _ibeg += unit_size;
    _obeg += unit_size;

    /* sequential computation: push task action in order to allows steal at this point while I'm doing seq computation */
    kaapi_stealbegin( stack, task, (kaapi_task_splitter_t)&static_splitter, 0 );
    nano_obeg = std::transform( nano_beg, nano_iend, nano_obeg, _op );

    /* return from sequential computation: remove concurrent task action 
       in order to disable any steal at this point while I'm doing seq computation */
    kaapi_stealend( stack, task );
  }
}


template<class InputIterator, class OutputIterator, class UnaryOperator>
InputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::beg0;
template<class InputIterator, class OutputIterator, class UnaryOperator>
OutputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::obeg0;

template<class InputIterator, class OutputIterator, class UnaryOperator>
struct TaskTransform1 : public atha::Task<3>::Signature<atha::Shared_r<InputIterator>, atha::Shared_w<InputIterator>, UnaryOperator> {};

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
  kaapi_task_initadaptive(stack, task, 
      &TransformStruct<InputIterator, OutputIterator, UnaryOperator>::static_mainentrypoint,
      &work, KAAPI_TASK_ADAPT_DEFAULT);
  kaapi_stack_pushtask(stack);
  
  kaapi_sched_sync(stack);
  kaapi_stack_restore_frame(stack, &frame);
}
#endif
