/*
 *  transform.cpp
 *  xkaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_COUNT_H
#define _XKAAPI_COUNT_H
#include "kaapi++.h"
#include <algorithm>


template<class InputIterator, class OutputIterator, class T>
typename std::iterator_traits<InputIterator>::difference_type
 count ( InputIterator begin, InputIterator end,  const T& value );


int unit_size = 1024;
static double t0;

/** Stucture of a work for transform
*/
template<class InputIterator, class Predicate>
class CountStruct {
public:
  typedef typename std::iterator_traits<InputIterator>::difference_type ReturnType;
  typedef CountStruct<InputIterator, Predicate> Self_t;
  
  static InputIterator  beg0;

  /* cstor */
  CountStruct(
    InputIterator ibeg,
    InputIterator iend,
    const Predicate& pred,
    ReturnType* retval = 0
  ) : _ibeg(ibeg), _iend(iend), _pred(pred), _result(retval)
  {}
  
  /* do transform */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

  /* Entry in case of thief execution */
  static void static_mainentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[MAIN] begin size:" << self_work->_iend - self_work->_ibeg << std::endl;
    self_work->doit(task, stack);
    /* merge... */
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[MAIN] begin preempt" << std::endl;
    while ( kaapi_preempt_nextthief(stack, task, NULL, &static_reducer, self_work->_result) );
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[MAIN] end preempt" << std::endl;
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[MAIN] end" << std::endl;
    /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function */  
  }

protected:  
  InputIterator    _ibeg;
  InputIterator    _iend;
  const Predicate& _pred;
  ReturnType*      _result;
  

  /* Entry in case of thief execution */
  static void static_thiefentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[THIEF] begin size:" << self_work->_iend - self_work->_ibeg << std::endl;
    ReturnType retval (0);
    self_work->_result = &retval;
    self_work->doit(task, stack);
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[THIEF] end compute" << std::endl;

    /* definition of the finalization point where all stolen work a interrupt and collected */
    kaapi_finalize_steal( stack, task, &retval, sizeof(ReturnType) );
//    std::cout << kaapi_get_elapsedtime()-t0 << "::[THIEF] end" << std::endl;
  }

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    int i = 0;
    int reply_count = 0;
    size_t bloc, size = (_iend - _ibeg);
    InputIterator local_end = _iend;
    Self_t* athief_work;

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
        kaapi_task_init( thief_stack, thief_task, &static_thiefentrypoint, kaapi_thread_pushdata(thief_stack, sizeof(Self_t)), KAAPI_TASK_ADAPTIVE);
        athief_work = new (kaapi_task_getargs(thief_task)) Self_t( local_end-bloc, local_end, _pred );
        local_end         -= bloc;
        kaapi_assert_debug( athief_work->_iend - athief_work->_ibeg >0);
        kaapi_assert_debug( athief_work->_ibeg - _ibeg > unit_size);

        kaapi_stack_pushtask( thief_stack );

        /* reply ok (1) to the request */
        kaapi_request_reply( victim_stack, task, &request[i], thief_stack, sizeof(ReturnType), 1 );
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


  int reducer( kaapi_stack_t* stack, kaapi_task_t* task, ReturnType* thief_result, ReturnType* victim_data )
  {
    *victim_data += *thief_result;
    return 1;
  }


  static int static_splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
    return self_work->splitter( victim_stack, task, count, request );
  }

  static int static_reducer( kaapi_stack_t* stack, kaapi_task_t* task, void* thief_data, void* victim_data )
  {
    Self_t* self_work = kaapi_task_getargst(task, Self_t);
    return self_work->reducer( stack, task, static_cast<ReturnType*>(thief_data), static_cast<ReturnType*>(victim_data) );
  }
};



/** Adaptive transform
*/
template<class InputIterator, class Predicate>
void CountStruct<InputIterator, Predicate>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  InputIterator nano_iend;
  InputIterator nano_beg;
  
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
    _ibeg += unit_size;

    /* sequential computation: push task action in order to allows steal at this point while I'm doing seq computation */
//    kaapi_stealbegin( stack, task, (kaapi_task_splitter_t)&static_splitter, 0 );
    *_result += std::count_if( nano_beg, nano_iend, _pred );

    /* return from sequential computation: remove concurrent task action 
       in order to disable any steal at this point while I'm doing seq computation */
//    kaapi_stealend( stack, task );
  }
  
}


/**
*/
template<class InputIterator, class Predicate>
typename std::iterator_traits<InputIterator>::difference_type count_if ( InputIterator begin, InputIterator end, const Predicate& pred )
{
  t0 = kaapi_get_elapsedtime();

  typename std::iterator_traits<InputIterator>::difference_type result(0);
  CountStruct<InputIterator, Predicate> work( begin, end, pred, &result);
  kaapi_stack_t* stack = kaapi_self_frame();
  kaapi_frame_t frame;
  kaapi_thread_save_frame(stack, &frame);

  /* will receive & process steal request */
  kaapi_task_t* task = kaapi_stack_toptask(stack);
  kaapi_task_initadaptive(stack, task, 
      &CountStruct<InputIterator, Predicate>::static_mainentrypoint,
      &work, KAAPI_TASK_ADAPT_DEFAULT);
  kaapi_stack_pushtask(stack);
  
  kaapi_sched_sync(stack);
  kaapi_thread_restore_frame(stack, &frame);
  return result;
}

template<class ValueType>
struct EqualToPredicate {
  EqualToPredicate( const ValueType& value ) : _value(value) {}
  bool operator()(const ValueType& v ) const
  { return (_value == v); }
  
  const ValueType& _value;
};

template<class InputIterator, class ValueType>
typename std::iterator_traits<InputIterator>::difference_type count ( InputIterator begin, InputIterator end, const ValueType& value )
{
  EqualToPredicate<ValueType> pred(value);
  return count_if( begin, end, pred );
}


#endif
