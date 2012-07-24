/*
** xkaapi
** 
**
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
void kaapi_staticschedtask_body( void* sp, kaapi_thread_t* uthread, kaapi_task_t* pc )
{
  int save_state;
  kaapi_frame_t* fp;
  kaapi_frame_t save_fp;
  kaapi_frame_tasklist_t* frame_tasklist;
  int16_t ngpu = 0;
  int16_t ncpu = 0;
  
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();

  kaapi_assert( thread->stack.sfp == (kaapi_frame_t*)uthread );

  /* here... begin execute frame tasklist*/
  KAAPI_EVENT_PUSH0(thread->stack.proc, thread, KAAPI_EVT_STATIC_BEG );
  
  /* Push a new frame */
  fp = (kaapi_frame_t*)thread->stack.sfp;
  /* push the frame for the next task to execute */
  save_fp = *fp;
  thread->stack.sfp[1] = *fp;
  kaapi_writemem_barrier();
  ++thread->stack.sfp;
  
  /* unset steal capability and wait no more thief 
     lock the kproc: it ensure that no more thief has reference on it 
  */
  save_state = thread->unstealable;
  kaapi_thread_set_unstealable(1);

  /* kaapi_sched_lock(&thread->stack.proc->lock); */
  /* kaapi_sched_unlock(&thread->stack.proc->lock); */

  /* some information to pass to the task : TODO */
  arg->schedinfo.nkproc[KAAPI_PROC_TYPE_MPSOC] = 0;

  /* Here, either:
     - nkproc[0] is !=-1 and represents annonymous ressources
       then nkproc[CPU] and nkproc[GPU] == -1.
     - nkproc[CPU] == nkproc[GPU] == nkproc[0] = -1, means auto detect
     - nkproc[CPU] or nkproc[GPU] == -2, means all ressources of ginve type
     - nkproc[CPU] or nkproc[GPU] is set to a user requested number
  */

  if (arg->schedinfo.nkproc[0] != -1)
  {
    kaapi_assert_debug(arg->schedinfo.nkproc[0] >0);

    /* first take all GPUs ressources, then complet with CPU ressources */
#if defined(KAAPI_USE_CUDA)
    arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU] = ngpu = kaapi_cuda_get_proc_count();
#endif
    if (ngpu < arg->schedinfo.nkproc[0])
    {
      ncpu = arg->schedinfo.nkproc[0] - ngpu;
      if (ncpu > (int16_t)kaapi_count_kprocessors)
        ncpu = kaapi_count_kprocessors;
    }
    arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU] = ncpu;
    arg->schedinfo.nkproc[0] = ngpu + ncpu;
  }
  else 
  {
    if (arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU] <0)
    { /* do not separate the case -1 and -2 (autodetec and all ressources) because
         the runtime is unable to return the available idle ressources
      */
      arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU] = kaapi_count_kprocessors;
    }
      
#if defined(KAAPI_USE_CUDA)
    if (arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU] <0)
    { /* do not separate the case -1 and -2 (autodetec and all ressources) because
         the runtime is unable to return the available idle ressources
      */
      arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU] = kaapi_cuda_get_proc_count();
    }
#else
    arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU] = 0;
#endif

    arg->schedinfo.nkproc[0] = arg->schedinfo.nkproc[KAAPI_PROC_TYPE_CPU] 
        + arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU];
  }

  /* the embedded task cannot be steal because it was not visible to thieves */
  arg->sub_body( arg->sub_sp, uthread, pc );

  /* allocate the tasklist for this task
  */
  frame_tasklist = (kaapi_frame_tasklist_t*)malloc(sizeof(kaapi_frame_tasklist_t));
  kaapi_frame_tasklist_init( frame_tasklist, thread );

  /* currently: that all, do not compute other things */
  kaapi_thread_computereadylist(thread, frame_tasklist);

  /* populate tasklist with initial ready tasks */
  kaapi_thread_tasklistready_push_init( &frame_tasklist->tasklist, &frame_tasklist->readylist );

  KAAPI_EVENT_PUSH0(thread->stack.proc, thread, KAAPI_EVT_STATIC_END );

#if defined(KAAPI_USE_PERFCOUNTER)
  /* here sfp is initialized, dump graph if required */
  if (getenv("KAAPI_DUMP_GRAPH") !=0)
  {
    static uint32_t counter = 0;
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"/tmp/graph.%s.%i.dot", getenv("USER"), counter++ );
    else
      sprintf(filename,"/tmp/graph.%i.dot", counter++);
    FILE* filedot = fopen(filename, "w");
    kaapi_frame_tasklist_print_dot( filedot, frame_tasklist, 0 );
    fclose(filedot);
  }
  if (getenv("KAAPI_DUMP_TASKLIST") !=0)
  {
    static uint32_t counter = 0;
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"/tmp/tasklist.%s.%i.log", getenv("USER"), counter++ );
    else
      sprintf(filename,"/tmp/tasklist.%i.log",counter++);
    FILE* filetask = fopen(filename, "w");
    kaapi_frame_tasklist_print( filetask, frame_tasklist );
    fclose(filetask);
  }
#endif
  
  /* restore state */
  kaapi_thread_set_unstealable(save_state);
  thread->stack.sfp->tasklist = &frame_tasklist->tasklist;

  /* exec the spawned subtasks */
  kaapi_sched_sync_(thread);

#if 0
  fprintf(stdout, "[%s] kid=%i tasklist tasks: %llu total: %llu\n", 
    __FUNCTION__,
    kaapi_get_self_kid(),
    KAAPI_ATOMIC_READ(&tasklist->cnt_exec),
    tasklist->total_tasks
  );
  fflush(stdout);
#endif

#if 0//defined(KAAPI_USE_PERFCOUNTER)
  printf("[tasklist] T1                      : %" PRIu64 "\n", tasklist->cnt_tasks);
  printf("[tasklist] Tinf                    : %" PRIu64 "\n", tasklist->t_infinity);
  printf("[tasklist] dependency analysis time: %e (s)\n",t1-t0);
  printf("[tasklist] exec time               : %e (s)\n",t1_exec-t0_exec);
#endif

  /* Pop & restore the frame */
  kaapi_sched_lock(&thread->stack.proc->lock);
  thread->stack.sfp->tasklist = 0;
  --thread->stack.sfp;
  *thread->stack.sfp = save_fp;
  kaapi_sched_unlock(&thread->stack.proc->lock);

#if 1 /* TODO: do not allocate if multiple uses of tasklist */
  kaapi_frame_tasklist_destroy( frame_tasklist );
  free(frame_tasklist);

//HERE: hack to do loop over SetStaticSched because memory state
// is leaved in inconsistant state.
//  kaapi_memory_destroy();
//  kaapi_memory_init();
#endif /* TODO */
}


