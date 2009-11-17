/*
 *  test_find_if.cpp
 *  xkaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _KAAPI_FINDIF_H_
#define _KAAPI_FINDIF_H_
#include "kaapi.h"
#include <algorithm>

template<class InputIterator, class Predicate>
InputIterator find_if ( InputIterator begin, InputIterator end, Predicate pred );

/** Stucture of a work for transform
*/
template<class InputIterator, class Predicate>
class FindIfStruct {
public:
  typedef FindIfStruct<InputIterator, Predicate> Self_t;

  /* cstor */
  FindIfStruct(
    InputIterator ibeg,
    InputIterator iend,
    Predicate     pred
  ) : _ibeg(ibeg), _iend(iend), _ifound(iend), _pred(pred)
  {}
  
  /* do transform */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

public:  
  InputIterator          _ibeg;
  InputIterator          _iend;
  InputIterator          _ifound;
  InputIterator          _macro_iend;
  Predicate              _pred;
  
  /* Entry in case of thief execution */
  static void static_thiefentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    Self_t* self_work = kaapi_task_argst(task, Self_t);
    self_work->doit(task, stack);
  }

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    FindIfStruct<InputIterator, Predicate>* output_work =0;
    int reply_count = 0;

    size_t size = (_macro_iend - _ibeg);
    if (size < 128) return;
    
    size_t bloc = size / (1+count);
    if (bloc < 128) { count = (size+127)/128 -1; bloc = 128; }

    int i = 0;
    InputIterator local_end = _macro_iend;

    while ((count >0) && (i<KAAPI_MAXSTACK_STEAL))
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_stack_t* thief_stack = request[i].stack;
        kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
        kaapi_task_init( thief_stack, thief_task, KAAPI_TASK_ADAPTIVE);
        kaapi_task_setbody( thief_task, &static_thiefentrypoint );
        kaapi_task_setargs(thief_task, kaapi_stack_pushdata(thief_stack, sizeof(Self_t)));
        output_work = kaapi_task_argst(thief_task, Self_t);

        output_work->_iend = local_end;
        output_work->_ibeg = output_work->_iend-bloc;
        output_work->_pred = pred;
        ckaapi_assert( output_work->_iend - output_work->_ibeg >0);
        kaapi_stack_pushtask( thief_stack );

#if defined(LOG)
        std::cout << request[i]->_index 
                  << "::Extract from " << stealcontext->_stack->_index 
                  << " work [" << *output_work->_ibeg 
                  << ":" << *output_work->_iend << "]" << std::endl;
#endif

        /* update end of the local work !! */
        local_end  = output_work->_ibeg;

        /* reply ok (1) to the request */
        kaapi_request_reply( stack, task, &request[i], thief_stack, 1 );
        ++reply_count;
        --count; 
      }
      ++i;
    }
    /* mute the input work !! */
    _macro_iend = local_end;
    kaapi_assert_debug( _macro_iend - _ibeg >0);
    return reply_count;
  }

  static int static_splitter( kaapi_stack_t* stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    Self_t* self_work = kaapi_task_argst(task, Self_t);
    return self_work->splitter( stack, task, count, request );
  }


  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, void* victim_data, void* iend_data )
  {
    FindIfStruct<InputIterator, Predicate>* thief_work = 
      (FindIfStruct<InputIterator, Predicate>* )thief_data;
    FindIfStruct<InputIterator, Predicate>* victim_work = 
      (FindIfStruct<InputIterator, Predicate>* )victim_data;
    InputIterator iend = (InputIterator)iend_data;

    if (victim_work->_ifound != iend)
    {
      return;
    }

    if (thief_work->_ifound != thief_work->_iend)
    {
      victim_work->_ifound = thief_work->_ifound;
      return;
    }
    /* do find_if in remainder work */
    victim_work->_ifound = std::find_if( thief_work->_ibeg, thief_work->_iend, victim_work->_pred );
    if (victim_work->_ifound == thief_work->_iend) victim_work->_ifound = iend;
  }

  static int preempt( void* victim_data, FindIfStruct<InputIterator, Predicate>* thief_work, InputIterator iend )
  {
    FindIfStruct<InputIterator, Predicate>* victim_work = 
      (FindIfStruct<InputIterator, Predicate>* )victim_data;
    thief_work->_ifound = iend;
    return 1;
  }
};



/** Adaptive transform
*/
template<class InputIterator, class Predicate>
void FindIfStruct<InputIterator, Predicate>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  InputIterator nano_iend;
  
  /* amount of work for macro loop */
  int unit_macro_step = 128;

  while (_ibeg != _iend)
  {
    /* macro loop */
    _macro_iend = _ibeg + unit_macro_step;
    if ((_iend - _macro_iend) <0 ) _macro_iend = _iend;
    _ifound = _macro_iend; 
    
    /* amount of work before testing request */
    int unit_size = 512;

    while (_ibeg != _macro_iend)
    {
      if (kaapi_preemptpoint( task, &preempt, this, _iend )) return;  

      /* definition of the steal point where steal_work may be called in case of steal request 
         -here size is pass as parameter and updated in case of steal.
      */
      kaapi_stealpoint( stack, task, &splitter );

      if (unit_size > macro_iend-_ibeg) unit_size = _macro_iend-_ibeg;
      nano_iend = _ibeg + unit_size;
      
      /* sequential computation */
      _ifound = std::find_if( _ibeg, nano_iend, _pred);
      if (_ifound != nano_iend)
      {
        /* preempt my thiefs */
        kaapi_pre( stack, task)

        /* finalize my thiefs */
        kaapi_finalize_steal( stack, task)
        return;
      }
      _ibeg = nano_iend;

    }
    kaapi_finalize_steal( _sc, this, &reducer, this, macro_iend );
    if (_ifound != macro_iend) return; 
    unit_macro_step = 3*unit_macro_step / 2;
  }
  
  /* finalize all steal request */
  kaapi_finalize_steal( _sc, 0, reducer, this, _iend );
}



template<class InputIterator, class Predicate>
InputIterator find_if ( kaapi_steal_context_t* stealcontext, InputIterator begin, InputIterator end, Predicate pred )
{
  kaapi_steal_context_initpush( stealcontext );
  FindIfStruct<InputIterator, Predicate> work( stealcontext, begin, end, pred);
  work.doit();

  return work._ifound;
}


#endif
