/*
** xkaapi
** 
**
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

double* base_addr = 0;

//#define TRACE 1
#define STATIC 1
//#define LOG_ACCESS 1

typedef struct quark_one_param_t {
  kaapi_access_t      addr;  /* .data used to store value */
  size_t              size;
  kaapi_access_mode_t mode;
} quark_one_param_t __attribute__((aligned(8)));


#define MAX_PARAMQUARK 32

/* XKaapi interface for Quark task with variable number of parameters */
typedef struct quark_task_s {
  void                 (*function) (Quark *);
  uintptr_t             callitwith_handle;  
  uint32_t              scratchbit;     
  uint32_t              nparam;    /* number of parameters */
  quark_one_param_t     param[MAX_PARAMQUARK];
} kaapi_quark_task_t;


/* format for the kaapi' quark task */
static kaapi_format_t* kaapi_quark_task_format = 0;

/* dump dot representation */
static int quark_dump_dot = 0;


/* init format */
static void kaapi_quark_task_format_constructor(void);

/* State fo Sequence */
#define QUARK_SEQUENCE_INIT          0x1
#define QUARK_SEQUENCE_FREE          0x2
#define QUARK_SEQUENCE_TASKLIST_INIT 0x4

/* Type for task sequences */
typedef struct Quark_sequence_s {
  int                     state_init;  /* see flag above */
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


static void kaapi_quark_helper_delete_scratch( kaapi_quark_task_t* arg )
{
  kaapi_bitmap_value32_t bits = { arg->scratchbit };
  while (!kaapi_bitmap_value_empty_32( &bits ))
  {
    int ith = kaapi_bitmap_first1_and_zero_32( &bits );
    kaapi_assert_debug(ith != 0);
    quark_one_param_t* param = &arg->param[ith-1];
    if (param->addr.data !=0) free(param->addr.data);
  }
}

/* trampoline to call Quark function */
static void kaapi_wrapper_quark_function( void* a, kaapi_thread_t* thread, kaapi_task_t* task )
{
//printf("%s\n", __PRETTY_FUNCTION__);  
  kaapi_quark_task_t* arg  = (kaapi_quark_task_t*)a;
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  XKaapi_Quark* myquark    = &default_Quark[kproc->kid];
  myquark->thread          = kproc->thread; 
  myquark->task            = arg; 
  arg->callitwith_handle   = 0;
  arg->function( myquark );
  if (arg->scratchbit)
    kaapi_quark_helper_delete_scratch( arg );
}

/* trampoline to call Quark function */
static void kaapi_wrapper_wh_quark_function( void* a, kaapi_thread_t* thread, kaapi_task_t* task )
{
//printf("%s\n", __PRETTY_FUNCTION__);  
  kaapi_quark_task_t* arg  = (kaapi_quark_task_t*)a;
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  XKaapi_Quark* myquark    = &default_Quark[kproc->kid];
  myquark->thread          = kproc->thread; 
  myquark->task            = arg; 
  arg->callitwith_handle   = 1;
  arg->function( myquark );
  if (arg->scratchbit)
    kaapi_quark_helper_delete_scratch( arg );
}


/**
*/
Quark *QUARK_Setup(int num_threads)
{
  static kaapi_atomic_t isinit = {0};
  
  if (KAAPI_ATOMIC_INCR(&isinit) > 1) 
  {
    /* already initialized */
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    return &default_Quark[kproc->kid];
  }
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  char* e = getenv("QUARK_DOT_DAG_ENABLE");
  if (e !=0)
  {
    if (strcmp(e, "0") ==0)
      quark_dump_dot = 0;
    else 
      quark_dump_dot = 1;
  }
  
#if 0
  /* It seems that PLASMA creates threads in place of quark (...).
    A BETTER UNDERSTANDING OF THREAD MANAGEMENT WITH QUARK IS REQUIRED !

    At first glance, each parallel section in PLASMA must correspond to 
      - thread creation (lazy)
      - execution
      - termination
    This does not correspond to the natural execution model of
    Kaapi where only the main thread is view from the programmer....

    In order to run XKaapi with quark -> 
      --nthreads=1 within KAAPI_CPUCOUNT or KAAPI_CPUSET defined.
  */
  if ((num_threads <1) || (num_threads > KAAPI_MAX_PROCESSOR)) 
    return 0;
  char tmp[32];
  snprintf(tmp, 32,"%i", num_threads);
printf("Setup environment KAAPI_CPUCOUNT:%s\n", tmp); fflush(stdout);
  setenv("KAAPI_CPUCOUNT",tmp, 1);
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

//printf("Begin task\n");
#if defined(LOG_ACCESS)
if (task_flags->task_priority) {
  printf("Priority info on task: %i\n", task_flags->task_priority );
}
#endif

  kaapi_quark_task_t* arg = kaapi_alloca( 
      thread, 
      sizeof(kaapi_quark_task_t) /* + nparam*sizeof(quark_one_param_t)*/ 
  );
  arg->scratchbit = 0;
  arg->nparam     = 0;
  arg->function   = function;

  /* For each argument */
  int nparam = 0;

#if defined(LOG_ACCESS)
printf("--->> taskarg: %p\n", (void*)arg);
#endif
  va_start(varg_list, task_flags);
  while( (arg_size = va_arg(varg_list, int)) != 0) 
  {
      void *arg_ptr  = va_arg(varg_list, void *);
      int arg_flags  = va_arg(varg_list, int);
      arg->param[nparam].size         = arg_size;
      arg->param[nparam].addr.version = 0;
      quark_direction_t arg_direction = (quark_direction_t) (arg_flags & QUARK_DIRECTION_BITMASK);
      switch ( arg_direction ) 
      {
        case VALUE:
//printf("VALUE [%i] @:%p, argsize:%i\n", nparam, arg_ptr, arg_size);
          arg->param[nparam].mode      = KAAPI_ACCESS_MODE_V;
          if (arg_size <= sizeof(void*)) /* copy the value in the pointer */
            arg->param[nparam].addr.data = *(void**)arg_ptr;
          else
          {
            arg->param[nparam].addr.data = kaapi_alloca(thread, arg_size);
            memcpy(arg->param[nparam].addr.data, arg_ptr, arg_size);
          }
          
#if defined(LOG_ACCESS)
printf("V[%i] @:%pp, sp@:%p", nparam, arg_ptr, arg->param[nparam].addr.data);
#endif
        break;

        case NODEP: /* but keep pointer */
          arg->param[nparam].addr.data = arg_ptr;
          arg->param[nparam].mode      = KAAPI_ACCESS_MODE_V;
#if defined(LOG_ACCESS)
printf("NO DEP[%i] @:%p", nparam, arg_ptr);
#endif
        break;

        case INPUT:
//printf("INPUT [%i] @:%p, argsize:%i\n", nparam, arg_ptr, arg_size);
          arg->param[nparam].addr.data = arg_ptr;
          arg->param[nparam].mode      = KAAPI_ACCESS_MODE_R;
#if defined(LOG_ACCESS)
printf("R[%i] @:%p", nparam, arg_ptr);
#endif
        break;
        
        case OUTPUT:

          arg->param[nparam].addr.data = arg_ptr;
//printf("OUTPUT [%i] @:%p, argsize:%i\n", nparam, arg_ptr, arg_size);
          if (arg_flags & ACCUMULATOR)
          {
            arg->param[nparam].mode    = KAAPI_ACCESS_MODE_CW;
            printf("ACCUMULATOR: Not yet implemented\n");
            abort();
          }
          else
            arg->param[nparam].mode    = KAAPI_ACCESS_MODE_W;
//printf("OUTPUT, @:%p, argsize:%i\n", arg_ptr, arg_size);
#if defined(LOG_ACCESS)
printf("W[%i] @:%p", nparam, arg_ptr);
#endif
        break;

        case SCRATCH:
          arg->param[nparam].addr.data = arg_ptr;
          arg->param[nparam].mode      = KAAPI_ACCESS_MODE_SCRATCH; /* new extension for Kaapi */
          arg->scratchbit |= (1 << nparam);
#if defined(LOG_ACCESS)
printf("SCRATCH[%i]\n", nparam);
#endif
        break;

        case INOUT:
//printf("INOUT [%i] @:%p, argsize:%i\n", nparam, arg_ptr, arg_size);
          arg->param[nparam].addr.data = arg_ptr;
          arg->param[nparam].mode      = KAAPI_ACCESS_MODE_RW;
#if defined(LOG_ACCESS)
printf("X[%i] @:%p",nparam, arg_ptr);
#endif
        break;

        default:
          printf("Unknown access mode: %i\n", (int)arg_direction );
          abort();
      }
#if defined(LOG_ACCESS)
{
  printf(", size:%i ", arg_size);
  int p = 0;
  if (arg_flags & LOCALITY) 
  {
    if (p) printf(" | ");
    printf("LOCALITY");
    p = 1;
  }
  if (arg_flags & ACCUMULATOR) 
  {
    if (p) printf(" | ");
    printf("ACCUM");
    p = 1;
  }
  if (arg_flags & GATHERV) 
  {
    if (p) printf(" | ");
    printf("GATHERV");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_0) 
  {
    if (p) printf(" | ");
    printf("REGION_0");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_1) 
  {
    if (p) printf(" | ");
    printf("REGION_1");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_2) 
  {
    if (p) printf(" | ");
    printf("REGION_2");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_3) 
  {
    if (p) printf(" | ");
    printf("REGION_3");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_4) 
  {
    if (p) printf(" | ");
    printf("REGION_4");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_5) 
  {
    if (p) printf(" | ");
    printf("REGION_5");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_6) 
  {
    if (p) printf(" | ");
    printf("REGION_6");
    p = 1;
  }
  if (arg_flags & QUARK_REGION_7) 
  {
    if (p) printf(" | ");
    printf("REGION_7");
    p = 1;
  }
  printf("\n");
}
#endif
      ++nparam;
      kaapi_assert( nparam <= MAX_PARAMQUARK );
  }
  va_end(varg_list);
#if defined(LOG_ACCESS)
printf("<<\n");
fflush(stdout);
#endif

//printf("#param: %i\n", nparam); fflush(stdout);
  arg->nparam        = nparam;
  kaapi_task_t* task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, (kaapi_task_body_t)kaapi_wrapper_quark_function, (void*)arg);

  if (task_flags->task_priority)
    kaapi_task_set_priority(task, KAAPI_TASK_MAX_PRIORITY);

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

//printf("End task: %p\n", (void*)task);
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
//  kaapi_assert_debug( quark->thread == kaapi_self_thread_context() );
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
  if (quark->sequence !=0)
    QUARK_Sequence_Wait(quark, quark->sequence);
  else {
    kaapi_sched_sync();
//    kaapi_assert_debug( quark->thread == kaapi_self_thread_context() );
  }
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
{
#if defined(TRACE)
printf("IN/OUT %s\n", __PRETTY_FUNCTION__);  
  fflush(stdout);
#endif
}

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
#if defined(LOG_ACCESS)
static int cntparam = 0;
#endif
void *QUARK_Args_List(Quark *quark)
{
  /* return the kaapi_quark_task_t* */
#if defined(LOG_ACCESS)
printf("POP ARG--->> taskarg: %p\n", (void*)quark->task);
cntparam = 0;
#endif
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
  kaapi_quark_task_t* taskarg = (kaapi_quark_task_t*)args_list;
  quark_one_param_t* arg2pop  = (quark_one_param_t*)*last_arg;
  void* retval;

  if (arg2pop ==0) 
    arg2pop = taskarg->param;

  if (arg2pop->mode & KAAPI_ACCESS_MODE_S)
  {
    /* allocate a scratch zone: freeed after task execution */
    if (arg2pop->addr.data ==0)
    {
      arg2pop->addr.data = malloc(arg2pop->size);
    }
    retval = &arg2pop->addr.data;
  }
  else
  {
    if (arg2pop->mode != KAAPI_ACCESS_MODE_V) 
    {
      if (taskarg->callitwith_handle)
      {
        kaapi_data_t* gd = (kaapi_data_t*)arg2pop->addr.data;
        retval = &gd->ptr.ptr;
      }
      else
        retval = &arg2pop->addr.data;
    }
    else 
    {
      if (arg2pop->size <= sizeof(void*))
        retval = &arg2pop->addr.data;
      else
        retval = arg2pop->addr.data;
    }
  }
  *last_arg = arg2pop+1;
#if defined(LOG_ACCESS)
void* addr = retval;
char mode;
switch (arg2pop->mode) {
  case KAAPI_ACCESS_MODE_V:
    mode = 'V';
    break;
  case KAAPI_ACCESS_MODE_R:
    mode = 'R';
    break;
  case KAAPI_ACCESS_MODE_W:
    mode = 'W';
    break;
  case KAAPI_ACCESS_MODE_RW:
    mode = 'X';
    break;
  case KAAPI_ACCESS_MODE_SCRATCH:
    mode = 'S';
    break;
  default:
    break;
}
printf("%c[%i], @: %p, size:%i\n", mode, cntparam, addr, arg2pop->size);
++cntparam;
#endif

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

static Quark_Sequence* last_qs_free = 0;

/* Create a seqeuence structure, to hold sequences of tasks */
Quark_Sequence *QUARK_Sequence_Create( Quark *quark )
{
#if defined(TRACE)
printf("IN %s\n", __PRETTY_FUNCTION__);  fflush(stdout);
#endif
//  kaapi_assert( quark->thread == kaapi_self_thread_context() );
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  
  /* activate static schedule */
  kaapi_quark_sequence_t* qs;
  if (last_qs_free !=0)
  {
    qs = last_qs_free;
    last_qs_free = 0;
    qs->state_init &= ~QUARK_SEQUENCE_FREE;
  } 
  else 
  {
    qs = (kaapi_quark_sequence_t*)malloc( sizeof(kaapi_quark_sequence_t) );
    qs->state_init = 0;
  }
#if defined(STATIC)
  /* set state of thread to unstealable */
  qs->save_state = thread->unstealable;
  kaapi_thread_set_unstealable(1);
#endif

  /* push new frame for task */
  qs->save_fp = *(kaapi_frame_t*)thread->stack.sfp;
  thread->stack.sfp[1] = qs->save_fp;
  kaapi_writemem_barrier();
  ++thread->stack.sfp;

#if defined(TRACE)
  printf("Push Frame, in static sched: thread:%p, sfp:%p = {pc:%p, sp:%p, spdata:%p}\n", 
      thread, 
      thread->stack.sfp,
      thread->stack.sfp->pc,
      thread->stack.sfp->sp,
      thread->stack.sfp->sp_data
  );
  fflush(stdout);
#endif
  quark->sequence = qs;
  qs->state_init |= QUARK_SEQUENCE_INIT;
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
  sequence->state_init &= ~QUARK_SEQUENCE_INIT;
  sequence->state_init |= QUARK_SEQUENCE_FREE;

  /* restore thread state */
  kaapi_thread_set_unstealable(sequence->save_state);

  if (last_qs_free ==0)
  {
    last_qs_free = sequence;
  }
  else {
    free(sequence);
  }
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
//  kaapi_assert_debug( quark->thread == kaapi_self_thread_context() );

  kaapi_thread_context_t* thread = kaapi_self_thread_context();

#if defined(STATIC)
  if (quark_dump_dot)
  {
    static uint32_t counter = 0;
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"graph.stack.%s.%i.dot", getenv("USER"), counter++ );
    else
      sprintf(filename,"graph.%i.dot",counter++);
    FILE* filedot = fopen(filename, "w");
    kaapi_frame_print_dot( filedot,  thread->stack.sfp, 0 );
    fclose(filedot);
  }


  if ((sequence->state_init & QUARK_SEQUENCE_TASKLIST_INIT) == 0)
    kaapi_tasklist_init( &sequence->tasklist, thread );
  kaapi_thread_computereadylist( thread, &sequence->tasklist );

  /* see kaapi_stsched_tasksetstatic.c */
  /* populate tasklist with initial ready tasks */
  kaapi_thread_tasklistready_push_init( &sequence->tasklist, &sequence->tasklist.readylist );
  kaapi_thread_tasklist_commit_ready( &sequence->tasklist );
  sequence->tasklist.context.chkpt = 0;

  thread->stack.sfp->tasklist = &sequence->tasklist;

  if (quark_dump_dot)
  {
    static uint32_t counter = 0;
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"graph.%s.%i.dot", getenv("USER"), counter++ );
    else
      sprintf(filename,"graph.%i.dot",counter++);
    FILE* filedot = fopen(filename, "w");
    kaapi_thread_tasklist_print_dot( filedot,  &sequence->tasklist, 0 );
    fclose(filedot);
  }

  /* force serialisation of previous write with the next write to stealable flag */  
  kaapi_writemem_barrier();

  /* reset stealable flag on thread */
  kaapi_thread_set_unstealable(sequence->save_state);
#endif

  /* real execution of tasks: here */
  kaapi_sched_sync_(thread);

  /* Pop & restore the frame: should use popframe */
  kaapi_sched_lock(&thread->stack.lock);
  thread->stack.sfp->tasklist = 0;
  
  --thread->stack.sfp;
  *thread->stack.sfp = sequence->save_fp;
  kaapi_sched_unlock(&thread->stack.lock);

#if defined(STATIC)
  kaapi_tasklist_destroy( &sequence->tasklist );
#endif
  
#if defined(TRACE)
  printf("Pop Frame, in static sched: thread:%p, sfp:%p = {pc:%p, sp:%p, spdata:%p}\n", 
      thread, 
      thread->stack.sfp,
      thread->stack.sfp->pc,
      thread->stack.sfp->sp,
      thread->stack.sfp->sp_data
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

/* format definition for any QUARK task */
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
#if 0
  if (arg->param[i].mode & KAAPI_ACCESS_MODE_S)
    return KAAPI_ACCESS_MODE_V;
#endif
  return arg->param[i].mode;
}

static
void* kaapi_quark_task_format_get_off_param(
  const struct kaapi_format_t* fmt, unsigned int i, const void* sp
)
{
//DEPRECATED
abort();
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
abort();
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
    (kaapi_task_body_t)kaapi_wrapper_wh_quark_function,
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
    kaapi_quark_task_format_get_task_binding,
    0
  );
}
