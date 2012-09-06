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
void kaapi_staticschedtask_body_gen( 
    void* sp, 
    kaapi_thread_t* uthread, 
    kaapi_task_t* pc,
    int selector
)
{
  int save_state;
  kaapi_frame_tasklist_t* frame_tasklist;
  int16_t ngpu = 0;
  int16_t ncpu = 0;
  
  kaapi_assert_debug( (selector==1) || (selector==2) )
  
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();

  kaapi_assert( thread->stack.sfp == (kaapi_frame_t*)uthread );

  /* here... begin execute frame tasklist*/
  KAAPI_EVENT_PUSH0(thread->stack.proc, thread, KAAPI_EVT_STATIC_BEG );
  
  /* Push a new frame */
  kaapi_thread_push_frame_(thread);
  
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
  switch (selector) 
  {
    case 1:
      arg->sub_body( arg->sub_sp, uthread, pc );
      break;

    case 2:
    {
      /* wh handle -> get it from embded task */
      const kaapi_format_t* fmt = kaapi_format_resolvebybody( (kaapi_task_body_t)arg->sub_body);
      kaapi_task_vararg_body_t body_wh = (kaapi_task_vararg_body_t)kaapi_format_get_task_bodywh_by_arch(
          fmt, 
          kaapi_processor_get_type(thread->stack.proc)
      );
      body_wh( arg->sub_sp, uthread, pc );
    } break;

    default:
      kaapi_assert(0);
      break;
  }

  if (!kaapi_frame_isempty(thread->stack.sfp)) 
  {

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

  /* Pop & restore the frame */
  thread->stack.sfp->tasklist = 0;
  kaapi_thread_pop_frame_(thread);

#if 1 /* TODO: do not allocate if multiple uses of tasklist */
  kaapi_frame_tasklist_destroy( frame_tasklist );
  free(frame_tasklist);

//HERE: hack to do loop over SetStaticSched because memory state
// is leaved in inconsistant state.
//  kaapi_memory_destroy();
//  kaapi_memory_init();
#endif /* TODO */

  } // end if tasks created

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

}


void kaapi_staticschedtask_body( void* sp, kaapi_thread_t* uthread, kaapi_task_t* pc )
{
  kaapi_staticschedtask_body_gen( sp, uthread, pc, 1); /* 1 == without wh */
}

void kaapi_staticschedtask_body_wh( void* sp, kaapi_thread_t* uthread, kaapi_task_t* pc )
{
  kaapi_staticschedtask_body_gen( sp, uthread, pc, 2); /* 2 == with wh */
}

void kaapi_staticschedtask_body_gpu( void* sp, kaapi_gpustream_t stream )
{
  kaapi_staticschedtask_body_gen( sp, kaapi_self_thread(), 0, 1); /* 1 == without wh */
}

void kaapi_staticschedtask_body_gpu_wh( void* sp, kaapi_gpustream_t stream )
{
  kaapi_staticschedtask_body_gen( sp, kaapi_self_thread(), 0, 2); /* 2 == with wh */
}


/* --------- format for task SetStatic --------- 
   - same format as the embedded task
*/
static size_t kaapi_taskformat_get_count_params(
 const struct kaapi_format_t* f,
 const void* sp
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt ==0) return 0;
  return kaapi_format_get_count_params(task_fmt, arg->sub_sp);
}

static kaapi_access_mode_t kaapi_taskformat_get_mode_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt ==0) 
    return KAAPI_ACCESS_MODE_VOID;
  return kaapi_format_get_mode_param(task_fmt, i, arg->sub_sp);
}

static void* kaapi_taskformat_get_off_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt ==0) return 0;
  return kaapi_taskformat_get_off_param(task_fmt, i, arg->sub_sp);
}

static kaapi_access_t kaapi_taskformat_get_access_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt !=0) 
    return kaapi_format_get_access_param(task_fmt, i, arg->sub_sp);
  {
    kaapi_access_t dummy;
    return dummy;
  }
}

static void kaapi_taskformat_set_access_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* sp,
 const kaapi_access_t* a
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt ==0) return;
  kaapi_format_set_access_param(task_fmt, i, arg->sub_sp, a);
}

static const struct kaapi_format_t* kaapi_taskformat_get_fmt_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt ==0) return 0;
  return kaapi_format_get_fmt_param(task_fmt, i, arg->sub_sp);
}

static kaapi_memory_view_t kaapi_taskformat_get_view_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt !=0) 
    return kaapi_format_get_view_param(task_fmt, i, arg->sub_sp);
  {
    kaapi_memory_view_t view;
    return view;
  }
}

static void kaapi_taskformat_set_view_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* sp,
 const kaapi_memory_view_t* v
)
{
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  const kaapi_format_t*	task_fmt= kaapi_format_resolvebybody((kaapi_task_body_t)arg->sub_body);
  if (task_fmt !=0)
    kaapi_format_set_view_param(task_fmt, i, arg->sub_sp, v);
}

__attribute__((unused)) 
static void kaapi_taskformat_reducor
(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* p,
 const void* q
)
{
  kaapi_abort();
}

__attribute__((unused)) 
static void kaapi_taskformat_redinit
(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp,
 void* p
)
{
  kaapi_abort();
}

__attribute__((unused))
static void kaapi_taskformat_get_task_binding(
 const struct kaapi_format_t* f,
 const kaapi_task_t* t,
 kaapi_task_binding_t* b
)
{
  b->type = KAAPI_BINDING_ANY;
}


void kaapi_register_staticschedtask_format(void)
{
  struct kaapi_format_t* format = kaapi_format_allocate();
  kaapi_format_taskregister_func
  (
    format,
    (kaapi_task_body_t)kaapi_staticschedtask_body, 
    (kaapi_task_body_t)kaapi_staticschedtask_body_wh,
    "kaapi_staticschedtask_body",
    sizeof(kaapi_staticschedtask_arg_t),
    kaapi_taskformat_get_count_params,
    kaapi_taskformat_get_mode_param,
    kaapi_taskformat_get_off_param,
    kaapi_taskformat_get_access_param,
    kaapi_taskformat_set_access_param,
    kaapi_taskformat_get_fmt_param,
    kaapi_taskformat_get_view_param,
    kaapi_taskformat_set_view_param,
    0, /* reducor */
    0, /* redinit */
    0, /* task binding */
    0  /* get_splitter */
  );
  kaapi_format_taskregister_body
  (
    format,
    (kaapi_task_body_t)kaapi_staticschedtask_body_gpu, 
    (kaapi_task_body_t)kaapi_staticschedtask_body_gpu_wh,
    KAAPI_PROC_TYPE_CUDA
  );
}

