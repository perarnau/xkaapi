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
#include "kaapi++"
#include <algorithm>


template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op );

/** Stucture of a work for transform
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
class TransformStruct {
public:
  /* MySelf */
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;

  /* cstor */
  TransformStruct(
    InputIterator ibeg,
    InputIterator iend,
    OutputIterator obeg,
    UnaryOperator  op
  ) : _ibeg(ibeg), _iend(iend), _obeg(obeg), _op(op) 
  {}
  
  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

  void doit( kaapi_stealcontext_t* sc_transform, kaapi_thread_t* thread )
  {
    /* local iterator for the nano loop */
    InputIterator nano_ibeg;
    InputIterator nano_iend;
    InputIterator nano_obeg;

    /* amount of work per iteration of the nano loop */
    int unit_size = 512;
    int tmp_size  = 0;

    /* Using THE: critical section could be avoided */
    while (1)
    {
      kaapi_steal_begincritical( sc_transform );
      nano_ibeg = _ibeg;
      nano_obeg = _obeg;
      tmp_size  = _iend -nano_ibeg;
      if (unit_size > tmp_size) { unit_size = tmp_size; nano_iend = _iend; }
      else nano_iend = _ibeg + unit_size;
      _ibeg = nano_iend;
      _obeg += nano_iend-nano_ibeg;
      kaapi_assert_debug( _iend-_ibeg <= 1000 );
      kaapi_assert_debug( nano_iend-nano_iend <= 1000 );
      kaapi_steal_endcritical( sc_transform );
      if (nano_iend == nano_ibeg) break;
      
      /* sequential computation: push task action in order to allows steal at this point while I'm doing seq computation */
      std::transform( nano_ibeg, nano_iend, nano_obeg, _op );
    }
  }

  static int static_splitter( kaapi_stealcontext_t* sc_transform, int count, kaapi_request_t* request, void* argsplitter )
  {
    Self_t* self_work = (Self_t*)argsplitter;
    return self_work->splitter( sc_transform, count, request );
  }

protected:
  /* Entry in case of thief execution */
  static void static_thiefentrypoint( void* arg, kaapi_thread_t* thread )
  {
    Self_t* self_work = (Self_t*)arg;
    kaapi_stealcontext_t* sc_transform = kaapi_thread_pushstealcontext( 
      thread,
      KAAPI_STEALCONTEXT_DEFAULT,
      Self_t::static_splitter,
      self_work,
      0
    );
    self_work->doit( sc_transform, thread );
    kaapi_steal_finalize( sc_transform );
  }


  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stealcontext_t* sc_transform, int count, kaapi_request_t* request )
  {
    int i = 0;
    int reply_count = 0;
    
    kaapi_assert_debug( _ibeg <= _iend );
    kaapi_assert_debug( _iend-_ibeg <= 1000 );
    InputIterator local_end = _iend;
    size_t bloc, size = (_iend - _ibeg);

    Self_t* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) return 0;
    bloc = size / (1+count);

    if (bloc < 128) { count = size/128 -1; bloc = 128; }
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_thread_t* thief_thread = request[i].thread;
        kaapi_task_t*  thief_task  = kaapi_thread_toptask(thief_thread);
        kaapi_task_init( thief_task, &static_thiefentrypoint, kaapi_thread_pushdata(thief_thread, sizeof(Self_t)) );
        output_work = kaapi_task_getargst(thief_task, Self_t);

        output_work->_iend = local_end;
        output_work->_ibeg = local_end-bloc;
        local_end         -= bloc;
        output_work->_obeg = _obeg + (output_work->_ibeg - _ibeg);
        kaapi_assert_debug( output_work->_iend > output_work->_ibeg);
        kaapi_assert_debug( output_work->_iend - output_work->_ibeg <= 1000 );
        output_work->_op   = _op;

        kaapi_thread_pushtask( thief_thread );

        /* reply ok (1) to the request */
        kaapi_request_reply_head( sc_transform, &request[i], 0 );
        --count; 
        ++reply_count;
      }
      ++i;
    }
    /* mute the end of input work of the victim */
    _iend  = local_end;
    kaapi_assert_debug( _iend - _ibeg >0);
    return reply_count;      
  }

protected:  
  InputIterator  volatile _ibeg __attribute__((aligned(8)));
  InputIterator  volatile _iend __attribute__((aligned(8)));
  OutputIterator volatile _obeg;
  UnaryOperator  _op;
} __attribute__((aligned(64)));


/**
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op )
{
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;
  
  Self_t work( begin, end, to_fill, op);

  kaapi_thread_t* thread =  kaapi_self_thread();
  kaapi_stealcontext_t* sc_transform = kaapi_thread_pushstealcontext( 
    thread,
    KAAPI_STEALCONTEXT_DEFAULT,
    Self_t::static_splitter,
    &work,
    0
  );
  
  work.doit( sc_transform, thread );
  
  kaapi_steal_finalize( sc_transform );
  ka::Sync();
}
#endif
