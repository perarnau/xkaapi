/*
 *  marc_problem.cpp
 *  xkaapi
 *  Exemple on how to iterate through a vector with a sliding window of constant size.
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include <algorithm>

#define WINDOW_SIZE 128

typedef double* InputIterator;

/**
*/
struct BasicOperation 
{
  void operator()( double value )
  {
  }
};

/** Stucture of a work for Marc's problem
*/
class SlidingWindowWork {
public:
  /* cstor */
  SlidingWindowWork(
    InputIterator ibeg,
    InputIterator iend
  ) : _ibeg(ibeg), _iend(iend)
  {
  }
  
  /* do computation */
  void doit(kaapi_task_t* task, kaapi_stack_t* data);

protected:  
  InputIterator  _ibeg;
  InputIterator  _iend;
  
  /* Entry in case of main execution */
  static void static_entrypoint(kaapi_task_t* task, kaapi_stack_t* data)
  {
    SlidingWindowWork* w = kaapi_task_getargst(task, SlidingWindowWork);
    w->doit(task, data);
  }

  /** splitter_work is called within the context of the steal point
  */
  static int splitter( kaapi_stack_t* stack, kaapi_task_t* self_task, 
                       int count, kaapi_request_t* request, 
                       double* ibeg, double** local_iend
                      )
  {
    int i = 0;

    size_t blocsize, size = (*local_iend - ibeg);
    InputIterator thief_end = *local_iend;

    SlidingWindowWork* output_work =0; 

    /* threshold should be defined (...) */
    if (size < 2) return 0;

    /* Sliding window: do not give to thiefs more than WINDOW_SIZE size of work 
       Cannot occurs on the thefts because they have a initial work less thant WINDOW_SIZE.
    */
    if (size > WINDOW_SIZE) 
    {
      size = WINDOW_SIZE;
      thief_end = ibeg + WINDOW_SIZE;
    }
    
    /* keep a bloc for myself */
    blocsize = size / (1+count); 
    
    /* adjust the number of bloc in order to do not have bloc of size less than 1 */
    if (blocsize < 1) { count = size-1; blocsize = 1; }
    
    int reply_count = 0;
    /* reply to all thiefs */
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_stack_t* thief_stack = request[i].stack;
        kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
        kaapi_task_init( thief_stack, thief_task, KAAPI_TASK_ADAPTIVE);
        kaapi_task_setbody( thief_task, &static_entrypoint );
        kaapi_task_setargs(thief_task, kaapi_stack_pushdata(thief_stack, sizeof(SlidingWindowWork)));
        output_work = kaapi_task_getargst(thief_task, SlidingWindowWork);

        output_work->_iend = thief_end;
        output_work->_ibeg = thief_end-blocsize;
        thief_end -= blocsize;
        kaapi_assert( output_work->_iend - output_work->_ibeg >0);

        /* reply ok (1) to the request */
        kaapi_request_reply( stack, self_task, &request[i], thief_stack, 1);
        --count; ++reply_count;
      }
      ++i;
    }
    /* mute the end of input work of the victim */
    *local_iend  = thief_end;
    kaapi_assert( *local_iend - ibeg >0);
    return reply_count;      
  }


  /* Called by the main thread to collect work from one other thief thread
  */
  static bool reducer( kaapi_task_t* self_task,
                       SlidingWindowWork* victim_work,
                       double** ibeg, double** local_iend,
                       SlidingWindowWork* thief_work )
  {
    if (thief_work->_ibeg == thief_work->_iend)
      /* no more work on the thief */
      return false;

    /* master get unfinished work of the thief */
    *ibeg = thief_work->_ibeg;
    *local_iend = thief_work->_iend;
    
    return true;
  }
};


/** Main entry point
*/
void SlidingWindowWork::doit( kaapi_task_t* task, kaapi_stack_t* stack )
{

  /* local iterator for the nano loop */
  double* nano_iend;
  
  /* amount of work per iteration of the nano loop */
  const size_t unit_size = 8;  /* should be automatically computed */
  size_t tmp_size = 0;

  /* maximum iend to be done in sequential */
  double* local_iend = _iend;

redo_work:  
  while (local_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       note that here local_iend is passed as parameter and updated in case of steal.
       The splitter cannot give more than WINDOW_SIZE size work to all other threads.
       Thus each thief thread cannot have more than WINDOW_SIZE size work.
    */
    kaapi_stealpoint_macro( stack, task, splitter, _ibeg, &local_iend );

    tmp_size = local_iend-_ibeg;
    if (tmp_size < unit_size ) {
       nano_iend = local_iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }

    /* sequential computation of the marc computation*/
    std::for_each( _ibeg, nano_iend, BasicOperation() );

    _ibeg = nano_iend;

    /* Return true iff the thread has been preempted.
       In order to avoid this comparaison for the sequential thread, a different entrypoint could be passed
       when a thread has been theft.
       here no function is called on the preemption point and the data 'this' is passed 
       to the thread that initiates the preemption.
    */
    if (kaapi_preemptpoint_macro( stack, task,   /* context */
                                  0,             /* function to call in case of preemption */
                                  this           /* arg to pass to the thread that do preemption */
                                                 /* here possible extra arguments to pass to the function call */
                        )) return;      
  }
  
  /* Try to preempt each theft thread in the reverse order defined by steal reponse (see Daouda order).
     Return true iff some work have been preempted and should be processed locally.
     0 is the value passed to the victim thread (no value). this, ibeg and local_iend are passed to 
     the call to reducer function.
     If work has been preempted, then ibeg != local_iend and should be done locally. See reducer function.
     If no more work has been preempted, it means that all the computation is finished.
  */
  if (kaapi_preempt_nextthief_macro( stack, task, 
                                     0,                         /* arg to pass to the thief */
                                     &reducer,                  /* function to call if a preempt exist */
                                     this, &_ibeg, &local_iend  /* arg to pass to the function call reducer */
                            )) 
    goto redo_work;

  /* Definition of the finalization point where main thread waits all the works.
     After this point, we enforce memory synchronisation: all data that has been writen before terminaison of the thiefs,
     could be read (...)
  */
  kaapi_finalize_steal( stack, task );
}


/**
*/
void marc_problem ( double* begin, double* end )
{
  /* push new adaptative task with following flags:
    - master finalization: the main thread will waits the end of the computation.
    - master preemption: the main thread will be able to preempt all thiefs.
  */
  kaapi_stack_t* stack = kaapi_self_stack();


  SlidingWindowWork work( begin, end);

  /* create the task on the top of the stack */
  kaapi_task_t*  task  = kaapi_stack_toptask(stack);
  kaapi_task_init( stack, task, KAAPI_TASK_ADAPTIVE);
  kaapi_task_setargs(task, &work );
  
  /* push_it task on the top of the stack */
  kaapi_stack_pushtask(stack);

  work.doit( task, stack );
}



/**
*/
int main( int argc, char* argv )
{
  /* */
  double* buffer = new double[8192];
  marc_problem( buffer, buffer + 8192 );
  return 0;
}
