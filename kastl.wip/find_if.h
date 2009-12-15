/*
 *  test_find_if.cpp
 *  xkaapi
 *
 *  Created by TG on 18/02/09.
 *  Updated by FLM on 12/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_FINDIF_H_
#define _XKAAPI_FINDIF_H_
#include "kaapi.h"
#include "kaapi_utils.h"
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
    Predicate     pred,
    bool is_master=true) : _ibeg(ibeg), _iend(iend), _ifound(iend), _pred(pred), _is_master(is_master)
  {}
  
  /* do find */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* do find with macro-loop*/
  void main_doit(kaapi_task_t* task, kaapi_stack_t* stack);

  /* do find without macro-loop*/
  void local_doit(kaapi_task_t* task, kaapi_stack_t* stack);

public:  
  InputIterator          _ibeg;
  InputIterator          _iend;
  InputIterator          _ifound;
  Predicate              _pred;
  bool _is_master;
  InputIterator _iend_local;  

  // request handler
  struct request_handler
  {
    InputIterator local_end;
    size_t bloc;
    Predicate pred;

    request_handler(InputIterator& _local_end, size_t _bloc, const Predicate& _pred)
    : local_end(_local_end), bloc(_bloc), pred(_pred) {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      output_work->_iend = local_end;
      output_work->_ibeg = output_work->_iend-bloc;
      output_work->_pred = pred;
      output_work->_is_master = false;
      kaapi_assert( output_work->_iend - output_work->_ibeg >0);
      local_end  = output_work->_ibeg;

      return true;
    }
  };

  typedef struct request_handler request_handler_t;

  /** splitter_work is called within the context of the steal point
  */
  int splitter
  (
   kaapi_stack_t* victim_stack,
   kaapi_task_t* task,
   int count,
   kaapi_request_t* request
  )
  {
    size_t size = (_iend - _ibeg);
    const int total_count = count;
    int replied_count = 0;
    size_t bloc;

    /* threshold should be defined (...) */
    if (size < 512)
      goto finish_splitter;
    
    bloc = size / (1 + count);

    if (bloc < 128) { count = size / 128 - 1; bloc = 128; }

    // iterate over requests
    {
      request_handler_t handler(_iend, bloc, _pred);

      replied_count =
	kaapi_utils::foreach_request
	(
	 victim_stack, task,
	 count, request,
	 handler, this
	 );

      // mute victim state after processing
      _iend = handler.local_end;

      kaapi_assert_debug(_iend - _ibeg > 0);
    }

  finish_splitter:
    {
      // fail the remaining requests

      const int remaining_count = total_count - replied_count;

      if (remaining_count)
	{
	  kaapi_utils::fail_requests
	    (
	     victim_stack,
	     task,
	     remaining_count,
	     request + replied_count
	     );
	}
    }

    // all requests have been replied to
    return total_count;
  }


#if 0 // TODO_REDUCER

  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, void* victim_data /*, void* iend_data*/)
  {
    FindIfStruct<InputIterator, Predicate>* thief_work = 
      (FindIfStruct<InputIterator, Predicate>* )thief_data;
    FindIfStruct<InputIterator, Predicate>* victim_work = 
      (FindIfStruct<InputIterator, Predicate>* )victim_data;
    //InputIterator iend = (InputIterator)iend_data;
    InputIterator iend =  victim_work->_iend_local;
#if defined(LOG)
    std::cout << "In reducer : ifound_thief " << thief_work->_sc->_stack->_index << " ="  << *thief_work->_ifound << std::endl;
    std::cout << "In reducer : ifound_victim " << victim_work->_sc->_stack->_index << " =" << *victim_work->_ifound << std::endl;
#endif

    if (victim_work->_ifound != iend)
    {
//#if defined(LOG)
      std::cout << "Victim has found the work !!!" << std::endl;
//#endif
      return;
    }

    if (thief_work->_ifound != thief_work->_iend)
    {
//#if defined(LOG)
      std::cout << "Thief " << thief_work->_sc->_stack->_index << " has found the work !!!" << std::endl;
//#endif
      victim_work->_ifound = thief_work->_ifound;
      //victim_work->_ibeg = victim_work->_iend = thief_work->_iend;
      return;
    }
    
    /* do find_if in remainder work */
    victim_work->_ifound = std::find_if( thief_work->_ibeg, thief_work->_iend, victim_work->_pred );
    if (victim_work->_ifound == thief_work->_iend) victim_work->_ifound = iend;
    //victim_work->_ibeg = victim_work->_iend = thief_work->_iend;
  }

  static int preempt( void* victim_data, FindIfStruct<InputIterator, Predicate>* thief_work, InputIterator iend )
  {
    FindIfStruct<InputIterator, Predicate>* victim_work = 
      (FindIfStruct<InputIterator, Predicate>* )victim_data;
//#if defined(LOG)
    std::cout << thief_work->_sc->_stack->_index 
              << "::Preempted by " << victim_work->_sc->_stack->_index 
              << std::endl; 
//#endif
    thief_work->_ifound = iend;
    return 1;
  }

#endif // TODO_REDUCER

};


/** Adaptive find_if with macro-loop
*/
template<class InputIterator, class Predicate>
void FindIfStruct<InputIterator, Predicate>::main_doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the macro loop */
  InputIterator macro_iend;

  /* local iterator for the nano loop */
  InputIterator nano_iend;
  
  /* amount of work for macro loop */
  int unit_macro_step = 128;

  while (_ibeg != _iend)
  {
    /* macro loop */
    macro_iend = _ibeg + unit_macro_step;
    if ((_iend - macro_iend) <0 ) macro_iend = _iend;
    _ifound = macro_iend; 
    
    /* amount of work before testing request */
    int unit_size = 512;

    while (_ibeg != macro_iend)
    {

#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint( _sc, &preempt, this, _iend )) return;
#else
# warning "TODO_REDUCER"
#endif // TODO_REDUCER

      /* definition of the steal point where steal_work may be called in case of steal request 
         -here size is pass as parameter and updated in case of steal.
      */
      kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );

      if (unit_size > macro_iend-_ibeg) unit_size = macro_iend-_ibeg;
      nano_iend = _ibeg + unit_size;
      
      /* sequential computation */
      _ifound = std::find_if( _ibeg, nano_iend, _pred);
      if (_ifound != nano_iend)
      {
        /* finalize my thiefs */
        _iend_local = nano_iend; // add by D. Traore
#warning "TODO_REDUCER"
        kaapi_finalize_steal(stack, task);
       return;
      }
      _ibeg = nano_iend;

    }
    _iend_local = macro_iend; // add by D. Traore
#warning "TODO_REDUCER"
    kaapi_finalize_steal(stack, task);
    if (_ifound != macro_iend) return; 
    unit_macro_step = 3*unit_macro_step / 2;
  }
 
  _iend_local = _iend; // add by D. Traore
  /* finalize all steal request */
#warning "TODO_REDUCER"
  kaapi_finalize_steal(stack, task);
}

/** Adaptive find_if with macro-loop 
*/
template<class InputIterator, class Predicate>
void FindIfStruct<InputIterator, Predicate>::local_doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* local iterator for the nano loop */
  InputIterator nano_iend;

 _ifound = _iend;
  while (_ibeg != _iend)
  {

    /* amount of work before testing request */
    int unit_size = 512;

#if 0 // TODO_REDUCER
    if (kaapi_preemptpoint( _sc, &preempt, this, _iend )) return;
#else
# warning "TODO_REDUCER"
#endif // TODO_REDUCER

      /* definition of the steal point where steal_work may be called in case of steal request
         -here size is pass as parameter and updated in case of steal.
      */
      kaapi_stealpoint( stack, task, &kaapi_utils::static_splitter<Self_t> );

      if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
      nano_iend = _ibeg + unit_size;

      /* sequential computation */
      _ifound = std::find_if( _ibeg, nano_iend, _pred);
      if (_ifound != nano_iend)
      {
        /* finalize my thiefs */
        _iend_local = nano_iend; // add by D. Traore
#warning "TODO_REDUCER"
        kaapi_finalize_steal(stack, task);
        return;
      }
      _ibeg = nano_iend;

    _iend_local = _iend; // add by D. Traore
#warning "TODO_REDUCER"
    kaapi_finalize_steal(stack, task);
    if (_ifound != _iend) return;
  }

  _iend_local = _iend; // add by D. Traore
  /* finalize all steal request */
#warning "TODO_REDUCER"
  kaapi_finalize_steal(stack, task);
}

/** Adaptive find_if
*/
template<class InputIterator, class Predicate>
void FindIfStruct<InputIterator, Predicate>::doit(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /*if(_is_master)*/ main_doit(task, stack);
  // else local_doit(task, stack);
}


template<class InputIterator, class Predicate>
InputIterator find_if ( InputIterator begin, InputIterator end, Predicate pred )
{
  FindIfStruct<InputIterator, Predicate> work( begin, end, pred);
  kaapi_utils::start_adaptive_task(&work);
  return work._ifound;
}


#endif
