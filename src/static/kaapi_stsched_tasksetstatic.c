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

#if defined(KAAPI_USE_CUDA)
# include "../machine/cuda/kaapi_cuda_execframe.h"
# include "../machine/cuda/kaapi_cuda_threadgroup_execframe.h"
#endif

/* 
*/
void kaapi_staticschedtask_body( void* sp, kaapi_thread_t* uthread )
{
  int save_state;
  kaapi_frame_t* fp;
  kaapi_frame_t save_fp;
  kaapi_tasklist_t* tasklist;
#if defined(KAAPI_USE_PERFCOUNTER)
  double t0=0.0;
  double t1=0.0;
  double t0_exec=0.0;
  double t1_exec=0.0;
#endif
  
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();

  kaapi_assert( thread->sfp == (kaapi_frame_t*)uthread );

  /* here... begin execute frame tasklist*/
  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_STATIC_BEG );
  
  /* Push a new frame */
  fp = (kaapi_frame_t*)thread->sfp;
  /* push the frame for the next task to execute */
  save_fp = *fp;
  thread->sfp[1] = *fp;
  ++thread->sfp;
  
  /* unset steal capability and wait no more thief 
     lock the kproc: it ensure that no more thief has reference on it 
  */
  save_state = thread->unstealable;
  kaapi_thread_set_unstealable(1);

  /* some information to pass to the task : TODO */
  arg->schedinfo.nkproc[KAAPI_PROC_TYPE_MPSOC-1] = 0;

  /* */
  if (arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU-1] == (uint32_t)-1)
    arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU-1] = kaapi_count_kprocessors;

  /* */
  if (arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU-1] == (uint32_t)-1)
    arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU-1] = 0;

  /* test if at least one ressource */
  if ( (arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU-1] == 0) &&
       (arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU-1] == 0) )
    arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU-1] = kaapi_count_kprocessors;

  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_STATIC_TASK_BEG );

  /* the embedded task cannot be steal because it was not visible to thieves */
  arg->sub_body( arg->sub_sp, uthread, &arg->schedinfo );

  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_STATIC_TASK_END );

  /* allocate the tasklist for this task
  */
  tasklist = (kaapi_tasklist_t*)malloc(sizeof(kaapi_tasklist_t));
  kaapi_tasklist_init( tasklist, thread );

  /* currently: that all, do not compute other things */
#if defined(KAAPI_USE_PERFCOUNTER)
  t0 = kaapi_get_elapsedtime();
#endif
  kaapi_thread_computereadylist(thread, tasklist);
#if defined(KAAPI_USE_PERFCOUNTER)
  t1 = kaapi_get_elapsedtime();
#endif
  KAAPI_ATOMIC_WRITE(&tasklist->count_thief, 0);
  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_STATIC_END );

  /* populate tasklist with initial ready tasks */
  kaapi_thread_tasklistready_push_init( &tasklist->rtl, &tasklist->readylist );
  kaapi_thread_tasklist_commit_ready( tasklist );

  /* keep the first task to execute outside the workqueue */
  tasklist->context.chkpt = 2;

#if defined(KAAPI_USE_PERFCOUNTER)
  /* here sfp is initialized, dump graph if required */
  if (getenv("KAAPI_DUMP_GRAPH") !=0)
  {
    static uint32_t counter = 0;
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"/tmp/graph.%s.%i.dot", getenv("USER"), counter++ );
    else
      sprintf(filename,"/tmp/graph.%i.dot",counter++);
    FILE* filedot = fopen(filename, "w");
    kaapi_thread_tasklist_print_dot( filedot, tasklist, 0 );
    fclose(filedot);
  }
#endif
  
  /* restore state */
  kaapi_thread_set_unstealable(save_state);
  thread->sfp->tasklist = tasklist;
  
//  kaapi_print_state_tasklist( tasklist );
//  printf("----------->>> before execution\n\n");

  /* exec the spawned subtasks */
#if defined(KAAPI_USE_PERFCOUNTER)
  t0_exec = kaapi_get_elapsedtime();
#endif
  kaapi_sched_sync_(thread);
#if defined(KAAPI_USE_PERFCOUNTER)
  t1_exec = kaapi_get_elapsedtime();
#endif
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&tasklist->count_thief) == 0);

#if defined(KAAPI_USE_PERFCOUNTER)
  printf("[tasklist] T1                      : %" PRIu64 "\n", tasklist->cnt_tasks);
  printf("[tasklist] Tinf                    : %" PRIu64 "\n", tasklist->t_infinity);
  printf("[tasklist] dependency analysis time: %e (s)\n",t1-t0);
  printf("[tasklist] exec time               : %e (s)\n",t1_exec-t0_exec);
#endif
#if 0
  printf("%i::[tasklist] exec tasks: %llu\n", 
    kaapi_get_self_kid(),
    tasklist->cnt_exectasks 
  );
  fflush(stdout);
#endif

  /* Pop & restore the frame */
  kaapi_sched_lock(&thread->proc->lock);
  thread->sfp->tasklist = 0;
  --thread->sfp;
  *thread->sfp = save_fp;
  kaapi_sched_unlock(&thread->proc->lock);

#if 1 /* TODO: do not allocate if multiple uses of tasklist */
  kaapi_tasklist_destroy( tasklist );
  free(tasklist);
//HERE: hack to do loop over SetStaticSched because memory state
// is leaved in inconsistant state.
  kaapi_memory_destroy();
  kaapi_memory_init();
#endif /* TODO */
}


