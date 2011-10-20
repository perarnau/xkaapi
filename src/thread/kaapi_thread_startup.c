/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include "kaapi_impl.h"
#include <stdio.h>


/** Task to bootstrap the execution onto a kprocessor after a successful steal
*/
void kaapi_taskstartup_body( 
  void*           taskarg,
  kaapi_frame_t*  fp,
  kaapi_task_t*   task
)
{
  kaapi_processor_t* kproc = (kaapi_processor_t*)taskarg;

#if defined(KAAPI_DEBUG)
  kaapi_frame_t           save_frame = *fp;
#endif

  kaapi_assert( kaapi_frame_isempty(fp) );
  kaapi_assert( fp == kproc->thread->stack.stackframe + 1 );
  kaapi_assert( task == kproc->thread->stack.stackframe->pc );
  

  /* last step: acquire the task for execution:
     - because this task will be embedded into kaapi_taskstartup_body,
     its state does not move to EXEC (as kaapi_thread_execframe does).
     Thus dot it manually here to avoid concurrency with preemption from the victim.
  */
  if (kaapi_task_markexec(kproc->thief_task))
  {
#if defined(HUGEDEBUG)
    printf("%i::[StarUp Thread] BEGIN to execute thieftask:%p, original task:%p\n", 
        kproc->kid,
        (void*)kproc->thief_task,
        (void*)((kaapi_tasksteal_arg_t*)kproc->thief_task->sp)->origin_task
    );
    fflush(stdout);
#endif
    ((kaapi_task_body_internal_t)kproc->thief_task->body)( kproc->thief_task->sp, fp, kproc->thief_task );
#if defined(HUGEDEBUG)
    printf("%i::[StarUp Thread] END to execute thieftask:%p\n", 
      kproc->kid,
      (void*)kproc->thief_task
    );
    fflush(stdout);
#endif
  }
#if defined(HUGEDEBUG)
  else {
    printf("%i::[StarUp Thread] ABORT: thieftask: %p, origin task:%p was preempted by victim\n", 
        kproc->kid,
        (void*)kproc->thief_task,
        (void*)((kaapi_tasksteal_arg_t*)kproc->thief_task->sp)->origin_task
    ); 
    fflush(stdout);
  }
#endif
}
