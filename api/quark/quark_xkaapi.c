/*
** xkaapi
** 
** Created on 10/01/2011
** Copyright 2011 INRIA.
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
#include <stdarg.h>
#include "quark_unpack_args.h"


#define TRACE 1
#define STATIC 1

typedef struct quark_one_param_t {
  kaapi_access_t      addr;  /* .data used to store value */
  size_t              size;
  kaapi_access_mode_t mode;
} quark_one_param_t __attribute__((aligned(8)));


#define MAX_PARAMQUARK 14

/* XKaapi interface for Quark task with variable number of parameters */
typedef struct quark_task_s {
  void                 (*function) (Quark *);
  uintptr_t             nparam;    /* number of parameters */
  quark_one_param_t     param[MAX_PARAMQUARK];
} kaapi_quark_task_t;


/* format for the kaapi' quark task */
static kaapi_format_t* kaapi_quark_task_format = 0;


/* init format */
static void kaapi_quark_task_format_constructor(void);


/* Type for task sequences */
typedef struct Quark_sequence_s {
  int                     save_state;  /* */
  kaapi_frame_t           save_fp;     /* saved fp of the current stack before sequence creation */
  kaapi_tasklist_t        tasklist;    /* */
} kaapi_quark_sequence_t;


/* opaque structure to user's quark program
*/
typedef struct quark_s {
  kaapi_thread_context_t* thread;
  kaapi_quark_task_t*     task;     /* the quark task = arg of Kaapi task */
  kaapi_quark_sequence_t* sequence;
} XKaapi_Quark;

static XKaapi_Quark default_Quark[KAAPI_MAX_PROCESSOR];


/* trampoline to call Quark function */
void kaapi_wrapper_quark_function( void* a, kaapi_thread_t* thread, kaapi_task_t* task )
{
//printf("%s\n", __PRETTY_FUNCTION__);  
  kaapi_quark_task_t* arg  = (kaapi_quark_task_t*)a;
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  XKaapi_Quark* myquark    = &default_Quark[kproc->kid];
  myquark->thread          = kproc->thread; 
  myquark->task            = arg; 
  arg->function( myquark );
}


/**
*/
Quark *QUARK_Setup(int num_threads)
{
  static kaapi_atomic_t isinit = {0};
  
  if (KAAPI_ATOMIC_INCR(&isinit) > 1) 
  {
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    return &default_Quark[kproc->kid];
  }
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  memset( &default_Quark, 0, sizeof(default_Quark) );
  kaapi_init(0, 0, 0);
  kaapi_quark_task_format_constructor();
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  default_Quark[kproc->kid].thread   = kproc->thread;
  default_Quark[kproc->kid].sequence = 0;
  kaapi_begin_parallel(KAAPI_SCHEDFLAG_DEFAULT);
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  return &default_Quark[kproc->kid];
}


/* Setup scheduler data structures, spawn worker threads, start the workers working  */
Quark *QUARK_New(int num_threads)
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  default_Quark[kproc->kid].thread = kproc->thread;
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  return &default_Quark[kproc->kid];
}


/* Add a task, called by the master process (thread_rank 0)  */
unsigned long long QUARK_Insert_Task(
  XKaapi_Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...
)
{
//printf("%s\n", __PRETTY_FUNCTION__);  
  va_list varg_list;
  int arg_size;
  kaapi_thread_t* thread = kaapi_self_thread();

  quark_one_param_t onearg[128];
  int nparam = 0;

//printf("Begin task\n");

  /* For each argument */
  va_start(varg_list, task_flags);
  while( (arg_size = va_arg(varg_list, int)) != 0) 
  {
      void *arg_ptr  = va_arg(varg_list, void *);
      int arg_flags  = va_arg(varg_list, int);
      onearg[nparam].size         = arg_size;
      onearg[nparam].addr.version = 0;
      quark_direction_t arg_direction = (quark_direction_t) (arg_flags & QUARK_DIRECTION_BITMASK);
      switch ( arg_direction ) 
      {
        case VALUE:
//printf("VALUE, @:%p, argsize:%i\n", arg_ptr, arg_size);
          onearg[nparam].addr.data   = *(void**)arg_ptr;
          onearg[nparam].mode      = KAAPI_ACCESS_MODE_V;
        break;

        case NODEP: /* but keep pointer */
          onearg[nparam].addr.data = arg_ptr;
          onearg[nparam].mode      = KAAPI_ACCESS_MODE_V;
//printf("NODEP, @:%p, argsize:%i\n", arg_ptr, arg_size);
        break;

        case INPUT:
//printf("INPUT, @:%p, argsize:%i\n", arg_ptr, arg_size);
          onearg[nparam].addr.data = arg_ptr;
          onearg[nparam].mode      = KAAPI_ACCESS_MODE_R;
        break;
        case OUTPUT:

          onearg[nparam].addr.data = arg_ptr;
          if (arg_flags & ACCUMULATOR)
          {
            printf("OUTPUT, @:%p, argsize:%i\n", arg_ptr, arg_size);
            onearg[nparam].mode    = KAAPI_ACCESS_MODE_CW;
          }
          else
            onearg[nparam].mode    = KAAPI_ACCESS_MODE_W;
//printf("OUTPUT, @:%p, argsize:%i\n", arg_ptr, arg_size);
        break;

        case SCRATCH:
          printf("SCRATCH MODE....\n"); kaapi_assert(0);
        break;

        case INOUT:
//printf("INOUT, @:%p, argsize:%i\n", arg_ptr, arg_size);
          onearg[nparam].addr.data = arg_ptr;
          onearg[nparam].mode      = KAAPI_ACCESS_MODE_RW;
        break;

        default:
          printf("Unknown access mode: %i\n", (int)arg_direction );
      }
      ++nparam;
  }
  va_end(varg_list);
  kaapi_assert( nparam <= MAX_PARAMQUARK );
  kaapi_quark_task_t* arg = kaapi_alloca( 
      thread, 
      sizeof(kaapi_quark_task_t) /* + nparam*sizeof(quark_one_param_t)*/ 
  );
  arg->nparam   = nparam;
  arg->function = function;
  memcpy( arg->param, onearg, nparam * sizeof(quark_one_param_t) );

printf("#param: %i\n", nparam); fflush(stdout);

  kaapi_task_t* task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, (kaapi_task_body_t)kaapi_wrapper_quark_function, (void*)arg);

  /* next parameters must follows in the stack */
  if (task_flags->task_sequence != 0)
  {
    /* push task with online computation of dependencies ? currently made at the end of the sequence  */
    kaapi_thread_pushtask(thread);
  }
  else
  {
    kaapi_thread_pushtask(thread);
  }
#if defined(TRACE)
  printf("%s:: Push task: thread:%p, sfp:%p\n",
    __PRETTY_FUNCTION__,
    quark->thread, thread
  );
  fflush(stdout);
#endif

//printf("End task\n");
  return (uintptr_t)task;
}

/* Main work loop, called externally by everyone but the master
 * (master manages this internally to the insert_task and waitall
 * routines). Each worker thread can call work_main_loop( quark,
 * thread_rank), where thread rank is 1...NUMTHREADS ) */
void QUARK_Worker_Loop(Quark *quark, int thread_rank)
{
  kaapi_sched_idle( kaapi_get_current_processor() );
}

/* Finish work and return.  Workers do not exit */
void QUARK_Barrier(Quark * quark)
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
}

/* Just wait for current tasks to complete, the scheduler and
 * strutures remain as is... should allow for repeated use of the
 * scheduler.  The workers return from their loops.*/
void QUARK_Waitall(Quark * quark)
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  if (quark->sequence !=0)
    QUARK_Sequence_Wait(quark, quark->sequence);
  else
    kaapi_sched_sync();
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
}

/* Delete scheduler, shutdown threads, finish everything, free structures */
void QUARK_Delete(Quark * quark)
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  kaapi_end_parallel(0);
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  kaapi_finalize();
}

/* Free scheduling data structures */
void QUARK_Free(Quark * quark)
{}

/* Cancel a specific task */
int QUARK_Cancel_Task(Quark *quark, unsigned long long taskid)
{
  kaapi_task_t* task = (kaapi_task_t*)taskid;
  uintptr_t state = kaapi_task_getstate(task);
  if (state == KAAPI_TASK_STATE_INIT)
    kaapi_task_casstate(task, KAAPI_TASK_STATE_INIT, KAAPI_TASK_STATE_TERM);
  return 0;
}

/* Returns a pointer to the list of arguments, used when unpacking the
   arguments; Returna a pointer to icl_list_t, so icl_list.h will need
   bo included if you use this function */
void *QUARK_Args_List(Quark *quark)
{
  /* return the kaapi_quark_task_t* */
  return quark->task;
}

/* Returns the rank of a thread in a parallel task */
int QUARK_Get_RankInTask(Quark *quark)
{ return kaapi_get_current_processor()->kid; }

/* Return a pointer to an argument.  The variable last_arg should be
   NULL on the first call, then each subsequent call will used
   last_arg to get the the next argument. */
void *QUARK_Args_Pop( void *args_list, void **last_arg)
{
  kaapi_quark_task_t* true_args = (kaapi_quark_task_t*)args_list;
  void* retval;
  if (*last_arg ==0) 
  {
    if (true_args->param->mode != KAAPI_ACCESS_MODE_V)
      retval = &true_args->param->addr.data;
    else
      retval = true_args->param->addr.data;
    *last_arg = true_args->param+1;
  }
  else {
    quark_one_param_t* true_last_arg = (quark_one_param_t*)*last_arg;
    if (true_last_arg->mode != KAAPI_ACCESS_MODE_V)
      retval = &true_last_arg->addr.data;
    else
      retval = true_last_arg->addr.data;
    *last_arg = true_last_arg+1;
  }
  return retval;
}

/* to debug quark */
extern void *kaapi_memcpy(void *dest, const void *src, size_t n)
{
  return memcpy( dest, src, n );
}

/* Utility function returning rank of the current thread */
int QUARK_Thread_Rank(Quark *quark)
{ return kaapi_get_current_processor()->kid; }

/* Packed task interface */
/* Create a task data structure to hold arguments */
Quark_Task *QUARK_Task_Init(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags )
{
  kaapi_assert(0);
  return 0;
}

/* Add (or pack) the arguments into a task data structure (make sure of the correct order) */
void QUARK_Task_Pack_Arg( Quark *quark, Quark_Task *task, int arg_size, void *arg_ptr, int arg_flags )
{
  kaapi_assert(0);
  return;
}

/* Insert the packed task data strucure into the scheduler for execution */
unsigned long long QUARK_Insert_Task_Packed(Quark * quark, Quark_Task *task )
{
  kaapi_assert(0);
  return 0;
}

/* Unsupported function for debugging purposes; execute task AT ONCE */
unsigned long long QUARK_Execute_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...)
{
  kaapi_assert(0);
  return 0;
}

/* Get the label (if any) associated with the current task; used for printing and debugging  */
char *QUARK_Get_Task_Label(Quark *quark)
{
  kaapi_assert(0);
  return 0;
}


/* Method for setting task flags */
Quark_Task_Flags *QUARK_Task_Flag_Set( Quark_Task_Flags *task_flags, int flag, intptr_t val )
{
  switch (flag)
  {
    case TASK_PRIORITY:
        task_flags->task_priority = (int)val;
        break;
    case TASK_LOCK_TO_THREAD:
        task_flags->task_lock_to_thread = (int)val;
        break;
    case TASK_LABEL:
        task_flags->task_label = (char *)val;
        break;
    case TASK_COLOR:
        task_flags->task_color = (char *)val;
        break;
    case TASK_SEQUENCE:
        task_flags->task_sequence = (Quark_Sequence *)val;
        break;
    case TASK_THREAD_COUNT:
        task_flags->task_thread_count = (int)val;
        break;
    case THREAD_SET_TO_MANUAL_SCHEDULING:
        task_flags->thread_set_to_manual_scheduling = (int)val;
        break;
  }
  return task_flags;
}


/* Create a seqeuence structure, to hold sequences of tasks */
Quark_Sequence *QUARK_Sequence_Create( Quark *quark )
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
  kaapi_assert( quark->thread == kaapi_self_thread_context() );

  /* activate static schedule */
  kaapi_quark_sequence_t* qs = (kaapi_quark_sequence_t*)malloc( sizeof(kaapi_quark_sequence_t) );
  kaapi_tasklist_init(&qs->tasklist, quark->thread);

  /* set state of thread to unstealable */
  qs->save_state = quark->thread->unstealable;
  kaapi_thread_set_unstealable(1);

  /* push new frame for task */
  qs->save_fp = *(kaapi_frame_t*)quark->thread->stack.sfp;
  quark->thread->stack.sfp[1] = qs->save_fp;
  kaapi_writemem_barrier();
  ++quark->thread->stack.sfp;

#if 1//defined(TRACE)
  printf("Push Frame, in static sched: thread:%p, sfp:%p = {pc:%p, sp:%p, spdata:%p}\n", 
      quark->thread, 
      quark->thread->stack.sfp,
      quark->thread->stack.sfp->pc,
      quark->thread->stack.sfp->sp,
      quark->thread->stack.sfp->sp_data
  );
  fflush(stdout);
#endif
  quark->sequence = qs;
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
  return qs;
}

/* Called by worker, cancel any pending tasks, and mark sequence so that it does not accept any more tasks */
int QUARK_Sequence_Cancel( Quark *quark, Quark_Sequence *sequence )
{
//printf("%s\n", __PRETTY_FUNCTION__);  
  kaapi_assert(0);
  return 0;
}

/* Destroy a sequence structure, cancelling any pending tasks */
Quark_Sequence *QUARK_Sequence_Destroy( Quark *quark, Quark_Sequence *sequence )
{
//printf("%s\n", __PRETTY_FUNCTION__);  
//  QUARK_Sequence_Wait( quark, sequence );
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
  kaapi_tasklist_destroy( &sequence->tasklist );
  free(sequence);
  quark->sequence = 0;
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
  return 0;
}

/* Wait for a sequence of tasks to complete */
int QUARK_Sequence_Wait( Quark *quark, Quark_Sequence *sequence )
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
  kaapi_assert_debug( quark->thread == kaapi_self_thread_context() );

#if defined(STATIC)
  kaapi_thread_computereadylist(quark->thread, &sequence->tasklist);

  /* see kaapi_stsched_tasksetstatic.c */
  /* populate tasklist with initial ready tasks */
  kaapi_thread_tasklistready_push_init( &sequence->tasklist.rtl, &sequence->tasklist.readylist );
  kaapi_thread_tasklist_commit_ready( &sequence->tasklist );
  sequence->tasklist.context.chkpt = 0;

  quark->thread->stack.sfp->tasklist = &sequence->tasklist;
  kaapi_writemem_barrier();
#endif
  kaapi_thread_set_unstealable(sequence->save_state);

  /* synchronize and execute tasks */
  kaapi_sched_sync_(quark->thread);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sequence->tasklist.count_thief) == 0);
printf("%s sync!!\n", __PRETTY_FUNCTION__);  fflush(stdout);

  /* Pop & restore the frame: should use popframe */
  kaapi_sched_lock(&quark->thread->stack.lock);
  quark->thread->stack.sfp->tasklist = 0;
  --quark->thread->stack.sfp;
  *quark->thread->stack.sfp = sequence->save_fp;
  kaapi_sched_unlock(&quark->thread->stack.lock);
  
#if 1//defined(TRACE)
  printf("Pop Frame, in static sched: thread:%p, sfp:%p = {pc:%p, sp:%p, spdata:%p}\n", 
      quark->thread, 
      quark->thread->stack.sfp,
      quark->thread->stack.sfp->pc,
      quark->thread->stack.sfp->sp,
      quark->thread->stack.sfp->sp_data
  );
  fflush(stdout);
#endif
#if defined(TRACE)
printf("OUT %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
  
  return 1;
}

/* Get the sequence information associated the current task/worker, this was provided when the tasks was created */
Quark_Sequence *QUARK_Get_Sequence(Quark *quark)
{
printf("%s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
  return quark->sequence;
}

/* Get the priority associated the current task/worker */
int QUARK_Get_Priority(Quark *quark)
{
  return 0;
}

/* Get information associated the current task and worker thread;
 * Callable from within a task, since it works on the currently
 * executing task */
intptr_t QUARK_Task_Flag_Get( Quark *quark, int flag )
{
  return 0;
}

/* Enable and disable DAG generation via API.  Only makes sense after
 * a sync, such as QUARK_Barrier. */
void QUARK_DOT_DAG_Enable( Quark *quark, int boolean_value )
{
  return;
}



static
size_t kaapi_quark_task_format_get_count_params(const struct kaapi_format_t* fmt, const void* sp)
{ 
  kaapi_quark_task_t* arg = (kaapi_quark_task_t*)sp;
  return arg->nparam;
}

static
kaapi_access_mode_t kaapi_quark_task_format_get_mode_param(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp
)
{ 
  kaapi_quark_task_t* arg = (kaapi_quark_task_t*)sp;
  return arg->param[i].mode;
}

static
void* kaapi_quark_task_format_get_off_param(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp
)
{
//DEPRECATED
  return 0;
}

static
kaapi_access_t kaapi_quark_task_format_get_access_param(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp
)
{
  kaapi_quark_task_t* arg = (kaapi_quark_task_t*)sp;
  kaapi_access_t retval = {0,0};
  if (arg->param[i].mode != KAAPI_ACCESS_MODE_V)
    retval = arg->param[i].addr;
  return retval;
}

static
void kaapi_quark_task_format_set_access_param(
  const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_access_t* a
)
{
  kaapi_quark_task_t* arg = (kaapi_quark_task_t*)sp;
  if (arg->param[i].mode != KAAPI_ACCESS_MODE_V)
    arg->param[i].addr = *a;
}

static
const struct kaapi_format_t* kaapi_quark_task_format_get_fmt_param(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp
)
{
  kaapi_quark_task_t* arg __attribute__((unused)) = (kaapi_quark_task_t*)sp;
  return kaapi_char_format;
}

static
kaapi_memory_view_t kaapi_quark_task_format_get_view_param(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp
)
{
  kaapi_quark_task_t* arg = (kaapi_quark_task_t*)sp;
  return kaapi_memory_view_make1d( arg->param[i].size, 1);
}

static
void kaapi_quark_task_format_set_view_param(
  const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_memory_view_t* view
)
{
  kaapi_quark_task_t* arg __attribute__((unused))= (kaapi_quark_task_t*)sp;
}

static
void kaapi_quark_task_format_reducor(
 const struct kaapi_format_t* fmt, unsigned int i, void* sp, const void* v
)
{
  kaapi_quark_task_t* arg __attribute__((unused))= (kaapi_quark_task_t*)sp;
}

static
void kaapi_quark_task_format_redinit(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp, void* v
)
{
  kaapi_quark_task_t* arg __attribute__((unused))= (kaapi_quark_task_t*)sp;
}

static
void kaapi_quark_task_format_get_task_binding(
  const struct kaapi_format_t* fmt, const void* t, kaapi_task_binding_t* tb
)
{ 
  return; 
}

/* constructor method */
static void kaapi_quark_task_format_constructor(void)
{
  if (kaapi_quark_task_format !=0) return;
  kaapi_quark_task_format = kaapi_format_allocate();
  kaapi_format_taskregister_func(
    kaapi_quark_task_format,
    (kaapi_task_body_t)kaapi_wrapper_quark_function,
    0,
    "kaapi_quark_task_format",
    sizeof(kaapi_quark_task_t),
    kaapi_quark_task_format_get_count_params,
    kaapi_quark_task_format_get_mode_param,
    kaapi_quark_task_format_get_off_param,
    kaapi_quark_task_format_get_access_param,
    kaapi_quark_task_format_set_access_param,
    kaapi_quark_task_format_get_fmt_param,
    kaapi_quark_task_format_get_view_param,
    kaapi_quark_task_format_set_view_param,
    kaapi_quark_task_format_reducor,
    kaapi_quark_task_format_redinit,
    kaapi_quark_task_format_get_task_binding
  );
}