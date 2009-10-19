/*
 *  marc_problem.cpp
 *  xkaapi
 *  Exemple on how to iterate through a vector with a sliding window of constant size.
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */

#define WINDOW_SIZE 128

/** Stucture of a work for Marc's problem
*/
class SlidingWindowWork : public kaapi_work_t {
public:
  /* cstor */
  SlidingWindowsWork(
    InputIterator ibeg,
    InputIterator iend
  ) : _ibeg(ibeg), _iend(iend)
  {
  }
  
  /* do computation */
  void doit();

protected:  
  InputIterator  _ibeg;
  InputIterator  _iend;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, kaapi_work_t* data)
  {
    SlidingWindowWork* w = (SlidingWindowWork*)data;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_task_t* self_task, int count, kaapi_steal_request_t** request, 
                        double* ibeg, double** local_iend
                      )
  {
    int i = 0;

    size_t blocsize, size = (*local_iend - ibeg);
    InputIterator thief_end = *local_iend;

    SlidingWindowWork* output_work =0; 

    /* threshold should be defined (...) */
    if (size < 2) goto reply_failed;

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
    
    /* reply to all thiefs */
    while (count >0)
    {
      if (request[i] !=0)
      {
        output_work->_iend = thief_end;
        output_work->_ibeg = thief_end-bloc;
        thief_end -= blocsize;
        kaapi_assert( output_work->_iend - output_work->_ibeg >0);

        /* reply ok (1) to the request */
        kaapi_request_reply_ok( self_task, request[i], 
                &thief_entrypoint, 
                output_work, sizeof(output_work), 
                KAAPI_MASTER_PREEMPT_FLAG|KAAPI_MASTER_FINALIZE_FLAG
        );
        --count; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  *local_iend  = thief_end;
  kaapi_assert( iend - ibeg >0);
  return;
      
reply_failed: /* to all other request */
    while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed to the request */
        kaapi_request_reply_fail( request[i] );
        --count; 
      }
      ++i;
    }
  }


  /* Called by the main thread to collect work from one other thief thread
  */
  static bool reducer( kaapi_task_t* self_task,
                       SlidingWindowWork* victim_work,
                       double** ibeg, double** local_iend
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
void MarcProblem::doit()
{
  kaapi_task_t* self_task = kaapi_self_task();


  /* local iterator for the nano loop */
  double* nano_iend;
  
  /* amount of work per iteration of the nano loop */
  const ptrdiff_t unit_size = 8;  /* should be automatically computed */
  ptrdiff_t tmp_size = 0;

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
    kaapi_stealpoint( self_task, splitter, _ibeg, &local_iend ))

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
    if (kaapi_preempoint( self_task, 
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
  if (kaapi_preempt_theft( self_task, 
                           &reducer,                  /* function to call if a preempt exist */
                           0,                         /* arg to pass to the thief */
                           this, &ibeg, &local_iend   /* arg to pass to the function call reducer */
                          )) 
    goto redo_work;

  /* Definition of the finalization point where main thread waits all the works.
     After this point, we enforce memory synchronisation: all data that has been writen before terminaison of the thiefs,
     could be read (...)
  */
  kaapi_finalize_steal( _sc );
}


/**
*/
void marc_problem ( double* begin, double* end )
{
  /* push new adaptative task with following flags:
    - master finalization: the main thread will waits the end of the computation.
    - master preemption: the main thread will be able to preempt all thiefs.
  */
  kaapi_task_t task;
  kaapi_self_push_task( &task, KAAPI_MASTER_PREEMPT_FLAG | KAAPI_MASTER_FINALIZE_FLAG );

  SlidingWindowWork work( begin, end);
  work.doit();

  kaapi_self_pop_task();
}
#endif
