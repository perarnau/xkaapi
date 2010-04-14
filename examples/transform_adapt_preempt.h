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
  static InputIterator beg0;
  static InputIterator end0;
  static OutputIterator obeg;
  
  /* MySelf */
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;

  /* cstor */
  TransformStruct(
    InputIterator ibeg,
    InputIterator iend,
    OutputIterator obeg,
    UnaryOperator  op
  ) : _ibeg(ibeg), _iend(iend), _obeg(obeg), _op(op), _thief_result(0)
  {}
  
  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

  void doit( kaapi_stealcontext_t* sc_transform, kaapi_thread_t* thread )
  {
    /* local iterator for the nano loop */
    InputIterator nano_ibeg;
    InputIterator nano_iend;
    InputIterator nano_obeg;

    /* amount of work per iteration of the nano loop */
    int unit_size = 128;
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
      kaapi_steal_endcritical( sc_transform );
      if (nano_iend == nano_ibeg) break;
      
      /* sequential computation: push task action in order to allows steal at this point while I'm doing seq computation */
      std::transform( nano_ibeg, nano_iend, nano_obeg, _op );
      
      /* here pass sc_transform to link thieves of sc_transform to its master */
      if (_thief_result) 
      {
        int (*reducer)( kaapi_taskadaptive_result_t*, 
                         void* victim_arg, 
                         Self_t* me
                  )
            = static_reducer;
        int retval = kaapi_preemptpoint( 
            _thief_result,         /* to test preemption */
            sc_transform,          /* to merge my thieves into list of the victim */
            reducer,               /* function to call if preemption signal */
            _thief_result,         /* extra data to pass to victim -> it will get it into its reducer */ 
            this, sizeof(Self_t),  /* data to pass to victim = at most the size given in kaapi_allocate_thief_result */
            this                   /* extra argument(s) for my reducer */
        );
        if (retval) {
          std::cout << "Thief: " << _thief_result << " returns with remaining ["
                    << _ibeg - beg0 << "," << _iend -beg0 << "), out=" << _obeg - obeg
                    << ", #items=" << _iend-_ibeg << std::endl;
          return;
        }
        usleep(5000);
      }
      usleep(1000);
    }
  }
  
  
  static int static_reducer( kaapi_taskadaptive_result_t* sc_transform, void* victim_arg, Self_t* me )
  {
    std::cout << "Thief:" << me->_thief_result << " I'm preempted !!!" << std::endl;
    return 1;
  }

  static int static_splitter( kaapi_stealcontext_t* sc_transform, int count, kaapi_request_t* request, void* argsplitter )
  {
    Self_t* self_work = (Self_t*)argsplitter;
    return self_work->splitter( sc_transform, count, request );
  }

  static int static_mainreducer( 
      kaapi_stealcontext_t* sc_transform, 
      void*                 thief_arg, 
      void*                 thiefdata, 
      size_t                thiefsize,
      Self_t*               myself
  )
  {
    Self_t *thief = (Self_t*)thiefdata;
    std::cout << "Master: Thief :" << thief_arg << " ?= " << thief_arg << " has been preempted [" 
              << thief->_ibeg - beg0 << "," << thief->_iend - beg0 << ") out=" << thief->_obeg - obeg
              << ", #items=" << thief->_iend - thief->_ibeg
              << std::endl;
    /* make a jump to the remaining part of the thief remainding work */
    myself->_ibeg = thief->_ibeg;
    myself->_iend = thief->_iend;
    myself->_obeg = thief->_obeg;
    kaapi_assert_debug( myself->_ibeg <= myself->_iend );
    return (myself->_iend != myself->_ibeg);
  }

protected:
  /* Entry in case of thief execution */
  static void static_thiefentrypoint( void* arg, kaapi_thread_t* thread )
  {
    Self_t* self_work = (Self_t*)arg;
    kaapi_stealcontext_t* sc_transform = kaapi_thread_pushstealcontext( 
      thread,
      KAAPI_STEALCONTEXT_DEFAULT,
      0,  // to avoid steal of thief Self_t::static_splitter,
      0,  // to avoid steal of thief self_work
      0
    );
    self_work->doit( sc_transform, thread );
    
    std::cout << "End of Thief: " << self_work->_thief_result << std::endl;
    kaapi_steal_finalize( sc_transform );
  }


  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stealcontext_t* sc_transform, int count, kaapi_request_t* request )
  {
    int i = 0;
    int reply_count = 0;
    int savecount = count;

    kaapi_assert_debug( _ibeg <= _iend );
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
        kaapi_assert_debug( output_work->_iend - output_work->_ibeg >0);
        output_work->_op   = _op;
        
        /* I reclaim to allocate Self_t data, in fact only 2 InputIterator + 1 OutputIterator is enough */
        output_work->_thief_result = kaapi_allocate_thief_result( sc_transform, sizeof(Self_t), 0 );
        output_work->_thief_result->data = output_work;
        std::cout << "New thief" << reply_count << "/" << savecount 
                  << ": " << output_work->_thief_result 
                  << ", with [" << output_work->_ibeg-beg0 << "," <<output_work->_iend-beg0
                  << ") out=" << output_work->_obeg - obeg 
                  << ", #= " << output_work->_iend-output_work->_ibeg << std::endl;

        kaapi_thread_pushtask( thief_thread );

        /* reply ok (1) to the request */
        kaapi_request_reply_head( sc_transform, &request[i], output_work->_thief_result );
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
  InputIterator                _ibeg;
  InputIterator                _iend;
  OutputIterator               _obeg;
  UnaryOperator                _op;
  kaapi_taskadaptive_result_t* _thief_result;
};


template<class InputIterator, class OutputIterator, class UnaryOperator>
InputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::beg0;
template<class InputIterator, class OutputIterator, class UnaryOperator>
InputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::end0;
template<class InputIterator, class OutputIterator, class UnaryOperator>
OutputIterator TransformStruct<InputIterator, OutputIterator, UnaryOperator>::obeg;

/**
*/
template<class InputIterator, class OutputIterator, class UnaryOperator>
void transform ( InputIterator begin, InputIterator end, OutputIterator to_fill, UnaryOperator op )
{
  typedef TransformStruct<InputIterator, OutputIterator, UnaryOperator> Self_t;
  Self_t::beg0 = begin;
  Self_t::end0 = end;
  Self_t::obeg = to_fill;
  
  Self_t work( begin, end, to_fill, op);

  kaapi_thread_t* thread =  kaapi_self_thread();
  kaapi_stealcontext_t* sc_transform = kaapi_thread_pushstealcontext( 
    thread,
    KAAPI_STEALCONTEXT_DEFAULT,
    Self_t::static_splitter,
    &work,
    0
  );
  
  do {
    work.doit( sc_transform, thread );

    kaapi_taskadaptive_result_t* thief = kaapi_preempt_getnextthief_head( sc_transform );
    if (thief !=0)
    {
      std::cout << "Master preempt Thief:" << thief << std::endl;
      if (kaapi_preempt_thief ( 
          sc_transform, 
          thief,                       /* thief to preempt */
          0,                           /* arg for the thief */
          Self_t::static_mainreducer,  /* my reducer */
          &work                        /* extra arg for the reducer */
      )) 
      {
        std::cout << "Continue with the remainding work of the thief:" << thief << std::endl;
        continue;
      }
      std::cout << "No work from thief:" << thief << std::endl;
    }
    else break;
  } while (1);
  
  
  kaapi_steal_finalize( sc_transform );
}
#endif
