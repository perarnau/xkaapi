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
#include <algorithm>
#include "kaapi.h"

namespace kastl2
{
struct __global_lock
{
  static volatile long _lock;

  static void acquire()
  {
    while (!__sync_bool_compare_and_swap(&_lock, 0, 1))
      ;
  }

  static void release()
  {
    _lock = 0;
  }
};

volatile long __global_lock::_lock = 0L;

template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );

/** Stucture of a work for transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
struct TransformStruct {
  /* MySelf */
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;

  /* cstor */
  TransformStruct(
    InputIterator ibeg,
    InputIterator iend,
    OutputIterator obeg,
    UnaryOperator  op
  ) : _ibeg(ibeg), _iend(iend), _obeg(obeg), _op(op) 
  {
    _msc = NULL;
  }
    
  template<class InputIterator, class OutputIterator, class UnaryOperator>
  void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );
  
  typedef typename std::iterator_traits<InputIterator>::value_type value_type;
    
  void doit( kaapi_stealcontext_t* sc_transform, kaapi_thread_t* thread )
  {
    __global_lock::acquire();
    //       kaapi_steal_begincritical( sc_transform );
    nano_ibeg = _ibeg;
    nano_obeg = _obeg;
    tmp_size  = _iend - nano_ibeg;
    if (unit_size > tmp_size)
      unit_size = tmp_size;
    nano_iend = nano_ibeg  + unit_size;
    _ibeg = nano_iend;
    _obeg += unit_size;
    //       kaapi_steal_endcritical( sc_transform );
    __global_lock::release();

    if (unit_size == 0)
      break;

    std::transform(nano_ibeg, nano_iend, nano_obeg, _op);
  }

  static int static_splitter
  (kaapi_stealcontext_t* sc_transform,
   int count, kaapi_request_t* request,
   void* argsplitter)
  {
    Self_t* self_work = (Self_t*)argsplitter;
    __global_lock::acquire();
    int res = self_work->splitter( sc_transform, count, request );
    __global_lock::release();
    return res;
  }

  /* Entry in case of thief execution */
  static void static_task_entrypoint( void* arg, kaapi_thread_t* thread )
  {
    Self_t* self_work = (Self_t*)arg;

    kaapi_stealcontext_t* sc_transform = kaapi_thread_pushstealcontext( 
      thread,
      KAAPI_STEALCONTEXT_LINKED,
      Self_t::static_splitter,
      self_work,
      self_work->_msc
    );
    self_work->doit( sc_transform, thread );

    kaapi_steal_finalize( sc_transform );
  }

  /** splitter_work is called within the context of the steal point
  */
  int splitter(kaapi_stealcontext_t* sc_transform, int count, kaapi_request_t* request)
  {
    int i = 0;
    int reply_count = 0;
    InputIterator local_end = _iend;
    size_t bloc, size = (_iend - _ibeg);

    Self_t* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512)
      return 0;

    bloc = size / (1 + count);

    if (bloc < 128)
    {
      count = size / 128 - 1;
      bloc = 128;
    }

    while (count > 0)
    {
      if (kaapi_request_ok(&request[i]))
      {
	kaapi_thread_t* thief_thread = kaapi_request_getthread(&request[i]);
	kaapi_task_t*  thief_task  = kaapi_thread_toptask(thief_thread);
	output_work = (Self_t*)kaapi_thread_pushdata_align(thief_thread, sizeof(Self_t), 8);
          
	output_work->_iend = local_end;
	output_work->_ibeg = local_end - bloc;
	local_end         -= bloc;
	output_work->_obeg = _obeg + (output_work->_ibeg - _ibeg);
	output_work->_op   = _op;
	output_work->_msc  = _msc;
          
	kaapi_task_init(thief_task, &static_task_entrypoint, output_work);
	kaapi_thread_pushtask(thief_thread);
	kaapi_request_reply_head(sc_transform, &request[i], NULL);
          
	--count; 
	++reply_count;
      }
      ++i;
    }
      /* mute the end of input work of the victim */
      _iend  = local_end;
      return reply_count;      
    }

  InputIterator  volatile _ibeg __attribute__((aligned(8)));
  InputIterator  volatile _iend __attribute__((aligned(8)));
  OutputIterator volatile _obeg;
  UnaryOperator  _op;
  kaapi_stealcontext_t* _msc;
} __attribute__((aligned(64)));


/**
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform
(InputIterator begin, InputIterator end,
 OutputIterator to_fill, UnaryOperator op)
{
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;
  Self_t work( begin, end, to_fill, op);

  kaapi_thread_t* thread;
  kaapi_task_t* task;
  kaapi_frame_t frame;

  thread = kaapi_self_thread();
  kaapi_thread_save_frame(thread, &frame);
  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, Self_t::static_task_entrypoint, (void*)&work);
  kaapi_thread_pushtask(thread);
  kaapi_sched_sync();
  kaapi_thread_restore_frame(thread, &frame);
}
} // kastl2::

#endif
