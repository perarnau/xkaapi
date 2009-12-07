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
#include <athapascan-1>
#include <iostream>
#include <math.h>

#define WINDOW_SIZE 128

typedef double* InputIterator;

InputIterator  ibeg0;
/**
*/
struct BasicOperation 
{
  void operator()( double& value )
  {
    double d = value;
    int max = (int)d;
    for (int i=0; i<max; ++i) d = sin(d);
//    std::cout << ">" << max; 
    value = d;
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
    atha::logfile() << "I'm a thief: BEGIN WORK [" << w->_ibeg - ibeg0 << "," << w->_iend - ibeg0 << ')' << std::endl;
    w->doit(task, data);
    atha::logfile() << "I'm a thief: END WORK [" << w->_ibeg - ibeg0 << "," << w->_iend - ibeg0 << ')' << std::endl;
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
    if (size < 32) return 0;

    atha::logfile() << "In Split work [" << ibeg - ibeg0 << "," << *local_iend - ibeg0 << "), #=" << count+1 << std::endl << std::flush;

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
        kaapi_task_initadaptive( thief_stack, thief_task, KAAPI_TASK_ADAPT_DEFAULT);
        kaapi_task_setbody( thief_task, &static_entrypoint );
        kaapi_task_setargs( thief_task, kaapi_stack_pushdata(thief_stack, sizeof(SlidingWindowWork)) );
        output_work = kaapi_task_getargst(thief_task, SlidingWindowWork);

        output_work->_iend = thief_end;
        output_work->_ibeg = thief_end-blocsize;
        thief_end -= blocsize;
        kaapi_assert( output_work->_iend - output_work->_ibeg >0);
        
        kaapi_stack_pushtask(thief_stack);
        
//        atha::logfile() << "I'm split work to a the thief" << std::endl;

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
  static bool reducer( kaapi_stack_t* stack, kaapi_task_t* self_task,
                       SlidingWindowWork* thief_work,
                       SlidingWindowWork* victim_work,
                       double** ibeg, double** local_iend
                      )
  {
    if ((thief_work ==0) || (thief_work->_ibeg == thief_work->_iend))
    {
      if (victim_work->_ibeg == victim_work->_iend)
      {
        atha::logfile() << "(0) Reduced work, mywork=[" << victim_work->_ibeg - ibeg0 << "," << victim_work->_iend -ibeg0 << ")"
                  << std::endl;
        return false;
      }
      *local_iend = victim_work->_iend;
      atha::logfile() << "(1) Reduced work, mywork=[" << victim_work->_ibeg - ibeg0 << "," << victim_work->_iend -ibeg0 << ")"
                << std::endl;
      return true;
    }

    /* master get unfinished work of the thief */
    *ibeg = thief_work->_ibeg;
    *local_iend = thief_work->_iend + (WINDOW_SIZE - (thief_work->_iend-thief_work->_ibeg));
    if (*local_iend > victim_work->_iend) 
      *local_iend = victim_work->_iend;
    atha::logfile() << "(2) Reduced work, mywork=[" << victim_work->_ibeg - ibeg0 << "," << victim_work->_iend -ibeg0 << ")"
          << std::endl;

    return true;
  }
  
  static void display_reducer(kaapi_stack_t* stack, kaapi_task_t* self_task, void* arg_from_victim, SlidingWindowWork* mywork )
  {
    SlidingWindowWork* victim_work = (SlidingWindowWork*)arg_from_victim;
    atha::logfile() << "I'm preempted by the victim [" << victim_work->_ibeg-ibeg0 << "," << victim_work->_iend -ibeg0 << "),"
              << " my work [" << mywork->_ibeg-ibeg0 << "," << mywork->_iend-ibeg0 << ")"
              << std::endl;
    
  }
};


/** Main entry point
*/
void SlidingWindowWork::doit( kaapi_task_t* task, kaapi_stack_t* stack )
{

  /* local iterator for the nano loop */
  double* nano_iend;
  
  /* amount of work per iteration of the nano loop */
  const size_t unit_size = 32;  /* should be automatically computed */
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
    kaapi_stealpoint( stack, task, &splitter, _ibeg, &local_iend );

    tmp_size = local_iend-_ibeg;
    if (tmp_size < unit_size ) {
       nano_iend = local_iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }

    /* sequential computation of the marc computation*/
    atha::logfile() << "Do work, mywork=[" << _ibeg - ibeg0 << "," << nano_iend - ibeg0 << ")"
          << std::endl;
    std::for_each( _ibeg, nano_iend, BasicOperation() );

    _ibeg = nano_iend;

    /* Return true iff the thread has been preempted.
       In order to avoid this comparaison for the sequential thread, a different entrypoint could be passed
       when a thread has been theft.
       here no function is called on the preemption point and the data 'this' is passed 
       to the thread that initiates the preemption.
    */
    if (kaapi_preemptpoint( stack, task,     /* context */
                            display_reducer, /* function to call in case of preemption */
                            this,            /* arg to pass to the thread that do preemption */
                            this             /* here possible extra arguments to pass to the function call */
                        )) return;      
  }
  
  /* Try to preempt each theft thread in the reverse order defined by steal reponse (see Daouda order).
     Return true iff some work have been preempted and should be processed locally.
     0 is the value passed to the victim thread (no value). this, ibeg and local_iend are passed to 
     the call to reducer function.
     If work has been preempted, then ibeg != local_iend and should be done locally. See reducer function.
     If no more work has been preempted, it means that all the computation is finished.
  */
  if (kaapi_preempt_nextthief( stack, task, 
                                     this,                      /* arg to pass to the thief */
                                     &reducer,                  /* function to call if a preempt exist */
                                     this, &_ibeg, &local_iend  /* arg to pass to the function call reducer */
                            ))
  {
    goto redo_work;
  }

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
  kaapi_task_t* task = kaapi_stack_toptask(stack);
  kaapi_task_initadaptive( stack, task, KAAPI_TASK_ADAPT_DEFAULT);
  kaapi_task_setargs(task, &work );
  
  /* push_it task on the top of the stack */
  kaapi_stack_pushtask(stack);

  work.doit( task, stack );
}



/**
*/
int main( int argc, char** argv )
{
  /* */
  double* buffer = new double[1024];
  ibeg0 = buffer;
  for (int i=0; i<1024; ++i) buffer[i] = i;
  marc_problem( buffer, buffer + 1024 );
  return 0;
}
