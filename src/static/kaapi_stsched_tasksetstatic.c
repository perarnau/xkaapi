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
#include <inttypes.h>


/* 
*/
void kaapi_staticschedtask_body( void* sp, kaapi_thread_t* uthread )
{
  int err;
  int save_state;
  kaapi_frame_t* fp;
  kaapi_tasklist_t* tasklist;
  double t0=0.0;
  double t1=0.0;
  
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();

  kaapi_assert( thread->sfp == (kaapi_frame_t*)uthread );
  
  /* Push a new frame */
  fp = (kaapi_frame_t*)thread->sfp;
  /* push the frame for the next task to execute */
  thread->sfp[1] = *fp;
  ++thread->sfp;
  
  /* unset steal capability and wait no more thief 
     lock the kproc: it ensure that no more thief has reference on it 
  */
  kaapi_sched_lock(&thread->proc->lock);
  save_state = thread->unstealable;
  thread->unstealable = 1;
  kaapi_sched_unlock(&thread->proc->lock);

  /* allocate the tasklist for this task
  */
  tasklist = (kaapi_tasklist_t*)malloc(sizeof(kaapi_tasklist_t));
  kaapi_tasklist_init( tasklist );

  if (arg->npart == -1)
  {    
    /* the embedded task cannot be steal because it was not visible to thieves */
    arg->sub_body( arg->sub_sp, uthread );
  
    /* currently: that all, do not compute other things */
    t0 = kaapi_get_elapsedtime();
    kaapi_thread_computereadylist(thread, tasklist);
    t1 = kaapi_get_elapsedtime();
  }
  else 
  {
    /* here we assume that all subtasks will be forked using SetPartition attribute:
       - after each forked task, the attribut will call kaapi_thread_online_computedep
       that directly call dependencies analysis
    */
    arg->sub_body( arg->sub_sp, uthread );
  }
  printf("[tasklist] T1:%" PRIu64 "\n", tasklist->cnt_tasks);
  printf("[tasklist] Tinf:%" PRIu64 "\n", tasklist->t_infinity);
  printf("[tasklist] analysis dependency time %e (s)\n",t1-t0);
  thread->unstealable = save_state;
  kaapi_writemem_barrier();
  thread->sfp->tasklist = tasklist;
  
#if defined(KAAPI_DEBUG)
  if (getenv("KAAPI_DUMP_GRAPH") !=0)
  {
    static uint32_t counter = 0;
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"/tmp/graph.%s.%i.dot", getenv("USER"), counter++ );
    else
      sprintf(filename,"/tmp/graph.%i.dot",counter++);
    FILE* filedot = fopen(filename, "w");
    kaapi_frame_print_dot( filedot, thread->sfp, 0 );
    fclose(filedot);
  }
#endif
  
  /* exec the spawned subtasks */
  err = kaapi_thread_execframe_tasklist( thread );
  kaapi_assert( (err == 0) || (err == ECHILD) );
  
  /* Pop & restore the frame */
  kaapi_sched_lock(&thread->proc->lock);
  thread->sfp->tasklist = 0;
  --thread->sfp;
  kaapi_sched_unlock(&thread->proc->lock);

  KAAPI_ATOMIC_ADD( &tasklist->count_exec, tasklist->cnt_exectasks );
  /* here we wait that all tasks of the tasklist have been executed 
     - is may be better to try to steal extra tasks for other threads
     by forking a non ready task that will be updated when condition
     becomes true.
     - this tasks will have an aftersteal method to delete the tasklist
  */
  while (KAAPI_ATOMIC_READ(&tasklist->count_exec) != (signed)tasklist->cnt_tasks)
   kaapi_slowdown_cpu();
  kaapi_assert( KAAPI_ATOMIC_READ(&tasklist->count_exec) == (signed)tasklist->cnt_tasks );

#if 0
  printf("%i::[tasklist] exec tasks: %llu\n", 
    kaapi_get_self_kid(),
    tasklist->cnt_exectasks 
  );
  fflush(stdout);
#endif
  
  kaapi_tasklist_destroy( tasklist );
  free(tasklist);

}


