/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#ifndef _KAAPI_TASK_H_
#define _KAAPI_TASK_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif


#include "config.h"
#include "kaapi_error.h"
#include "kaapi_atomic.h"
#include "kaapi_cpuset.h"

#include "kaapi_defs.h"

#if defined(KAAPI_USE_JMP)
#include <setjmp.h>
#endif

/* Maximal number of recursive calls used to store the stack of frames.
   The value indicates the maximal number of frames that can be pushed
   into the stackframe for each thread.
   
   If an assertion is thrown at runtime, and if this macro appears then
   it is necessary to increase the maximal number of frames in a stack.
*/
#define KAAPI_MAX_RECCALL 256

struct kaapi_listrequest_iterator_t;
struct kaapi_listrequest_t;

/* ============================= The stack of task data structure ============================ */
/** States of a task.
    The state diagram is the following:
    *  INIT -> EXEC -> TERM
    Normal execution of tasks, the task is created, under execution and then terminated.
    This path is tried to be as fast as possible.
    
    *  INIT -> STEAL -> TERM  
    Execution of the task stolen: the task is created, marked STEAL while the thief execute it,
    and then it is marked terminated if no merge of results are required.

    *  INIT -> STEAL -> MERGE [-> EXEC -> TERM]
    Execution of the task stolen: the task is created, marked STEAL while the thief execute it,
    and then it is marked MERGE if some merge of results are required. Then the victim will
    execute a post processing functions. The last two states between brackets are not explicitly
    represented. After the execution of the merge operation the task is considered as terminated.
    
    * ALLOCATED -> INIT -> ... then see above the possible transitions 
    This is the case of that occurs during a steal operation: the thief thread has a task in state
    ALLOCATED before it mades steal request. This task will store the information about the stolen 
    task to execute. Then, when the thief finish to get information about the stolen task, it
    made the transition to INIT, then the thief task can be executed or stolen using previous transitions.
    If the transition to INIT is impossible, then it means that the task has been preempted by the victim.
    
    * ALLOCATED -> PREEMPTED
    As in the previous transition, but the victim will preempt the thief task before the thief thread
    finish to steal the task and made the transition to INIT. 
    
    * INIT -> PREEMPTED
    Same as previous transition: the vicitm preempt the thief task before execution begin. Then the this
    will abort its tansition to EXEC.
    
    * TODO:  EXEC -> SIGNALED
    If a thief task receive a request for preemption during its execution
*/
#define KAAPI_TASK_STATE_INIT       0x0
#define KAAPI_TASK_STATE_EXEC       0x1
#define KAAPI_TASK_STATE_STEAL      0x2
#define KAAPI_TASK_STATE_TERM       0x4
#define KAAPI_TASK_STATE_MERGE      0x8
#define KAAPI_TASK_STATE_ALLOCATED  0x10
#define KAAPI_TASK_STATE_SIGNALED   0x20   /* mask: extra flag to set the task as signaled for preemption */
#define KAAPI_TASK_STATE_LOCKED     0x40   /* mask: lock the state of the task */

typedef void (*kaapi_task_body_internal_t)(void* /*task arg*/, kaapi_thread_t* /* thread or stream */, kaapi_task_t*);

/* ============================= The stack of task data structure ============================ */
/** The stack of tasks data structure
    This data structure is a stack of tasks with following method:
    - push (defined in kaapi.h)
    - pop : implicitely used in execframe
    - steal: defined in kaapi_stack_steal.

    This data structure may be extend in case where the C-stack is required to be suspended and resumed.
    
    WARNING: sfp should be the first field of the data structure in order to be able to recover in the public
    API sfp (<=> kaapi_thread_t*) from the kaapi_stack_t pointer stored in kaapi_current_thread_key.
*/
typedef struct kaapi_stack_t {
  kaapi_frame_t*        volatile sfp;            /** pointer to the current frame (in stackframe) */
  kaapi_frame_t*                 esfp;           /** first frame until to execute all frame  */
  kaapi_frame_t*                 stackframe;     /** for execution, see kaapi_thread_execframe */
  struct kaapi_processor_t*      proc;           /** access to the running processor */
  kaapi_task_t*                  task;           /** bottom of the stack of task */
  char*                          data;           /** begin of stack of data */ 
#if defined(KAAPI_USE_JMP)
  jmp_buf*                       jbuf;
#endif

  /* execution state for stack of task */
  kaapi_frame_t*        volatile thieffp __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the thief frame where to steal */
  kaapi_atomic_t                 lock;           // __attribute__((aligned(KAAPI_CACHE_LINE)));

} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_stack_t;


/* ===================== Default internal task body ==================================== */

/** Body of task used to mark a theft task into a victim stack
    \ingroup TASK
*/
extern void kaapi_anormal_body( void*, kaapi_thread_t*, kaapi_task_t* );

/** Body of the task used to mark a thread suspend in its execution
    \ingroup TASK
*/
extern void kaapi_execthread_body( void*, kaapi_thread_t*);

/** Merge result after a steal
    This body is set by a thief at the end of the steal operation in case of 
    results to merge. Else the thief set the task' steal body to kaapi_term_body
    \ingroup TASK
*/
extern void kaapi_aftersteal_body( void*, kaapi_thread_t*, kaapi_task_t* );

/** Body of the nop task that do nothing
    \ingroup TASK
*/
extern void kaapi_nop_body( void*, kaapi_thread_t*);

/** Body of the startup task 
    \ingroup TASK
*/
extern void kaapi_taskstartup_body( void*, kaapi_thread_t*, kaapi_task_t*);

/** Body of task steal created on thief stack to execute a task
    \ingroup TASK
*/
extern void kaapi_tasksteal_body( void*, kaapi_thread_t* );

/** Body of task steal created on thief stack to execute a task
    theft from ready tasklist
    \ingroup TASK
*/
extern void kaapi_taskstealready_body( void*, kaapi_thread_t* );

/** Write result after a steal 
    \ingroup TASK
*/
extern void kaapi_taskwrite_body( void*, kaapi_thread_t*, kaapi_task_t* );


/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_adapt_body( void*, kaapi_thread_t* );


/** Task with kaapi_move_arg_t as parameter in order to initialize a first R access
    from the original data in memory.
*/
extern void kaapi_taskmove_body( void*, kaapi_thread_t* );

/** Task with kaapi_move_arg_t as parameter in order to allocate data for a first CW access
    from the original data in memory.
*/
extern void kaapi_taskalloc_body( void*, kaapi_thread_t* );

/** Task with kaapi_move_arg_t as parameter in order to mark synchronization at the end of 
    a chain of CW access. It is the task that logically produce the data.
*/
extern void kaapi_taskfinalizer_body( void*, kaapi_thread_t* );


/** \ingroup TASK
*/
/**@{ */
static inline kaapi_task_body_t kaapi_task_getbody(const kaapi_task_t* task)
{
  return ((volatile kaapi_task_t*)task)->body;
}

static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_body_t body)
{
  ((volatile kaapi_task_t*)task)->body = body;
}

static inline uintptr_t kaapi_task_getstate(const kaapi_task_t* task)
{
  return ((volatile kaapi_task_t*)task)->state;
}

static inline void kaapi_task_setstate(kaapi_task_t* task, uintptr_t state)
{
  ((volatile kaapi_task_t*)task)->state = state;
}

static inline void kaapi_task_setbody_barrier(kaapi_task_t* task, kaapi_task_body_t body)
{
  kaapi_mem_barrier();
  ((volatile kaapi_task_t*)task)->body = body;
}

static inline uintptr_t kaapi_task_casstate(kaapi_task_t* task, uintptr_t oldstate, uintptr_t newstate)
{
  return KAAPI_ATOMIC_CASPTR( &task->state, oldstate, newstate);
}

/* return the old state before atomic op
*/
static inline uintptr_t kaapi_task_orstate(kaapi_task_t* task, uintptr_t newstate)
{
  return KAAPI_ATOMIC_ORPTR_ORIG( &task->state, newstate);
}

/* Return the state of the task
*/
static inline int kaapi_task_markexec( kaapi_task_t* task )
{
  return KAAPI_ATOMIC_CASPTR( &task->state, KAAPI_TASK_STATE_INIT, KAAPI_TASK_STATE_EXEC);
}


static inline kaapi_task_body_t kaapi_task_marksteal( kaapi_task_t* task )
{
  if (likely(KAAPI_ATOMIC_CASPTR( &task->state, KAAPI_TASK_STATE_INIT, KAAPI_TASK_STATE_STEAL)))
    return task->body;
  return 0;
}


static inline void kaapi_task_markterm( kaapi_task_t* task )
{
#if defined(KAAPI_DEBUG)
  uintptr_t state = kaapi_task_getstate(task);
  kaapi_assert( state != KAAPI_TASK_STATE_TERM );
  kaapi_assert( (state & ~KAAPI_TASK_STATE_LOCKED) == KAAPI_TASK_STATE_STEAL );
#endif
  while (!KAAPI_ATOMIC_CASPTR( &task->state, KAAPI_TASK_STATE_STEAL, KAAPI_TASK_STATE_TERM)) 
    kaapi_slowdown_cpu();
}

static inline void kaapi_task_markaftersteal( kaapi_task_t* task )
{
#if defined(KAAPI_DEBUG)
  uintptr_t state = kaapi_task_getstate(task);
  kaapi_assert( state != KAAPI_TASK_STATE_TERM );
  kaapi_assert( (state & ~KAAPI_TASK_STATE_LOCKED) == KAAPI_TASK_STATE_STEAL );
#endif
  while (!KAAPI_ATOMIC_CASPTR( &task->state, KAAPI_TASK_STATE_STEAL, KAAPI_TASK_STATE_MERGE)) ;
    kaapi_slowdown_cpu();
}



#if 0 // DEPRECATED
static inline unsigned int kaapi_task_state_isspecial(uintptr_t state)
{
  return !(state == 0);
}

static inline unsigned int kaapi_task_state_isnormal(uintptr_t state)
{
  return !kaapi_task_state_isspecial(state);
}

static inline unsigned int kaapi_task_state_isready(uintptr_t state)
{
  return (state & (KAAPI_MASK_BODY_AFTER | KAAPI_MASK_BODY_TERM)) != 0;
}

static inline unsigned int kaapi_task_state_isstealable(uintptr_t state)
{
  return (state & (KAAPI_MASK_BODY_STEAL | KAAPI_MASK_BODY_EXEC)) == 0;
}
#endif


inline static void kaapi_task_lock_adaptive_steal(kaapi_stealcontext_t* sc)
{
#if 0
  while (1)
  {
    if ((KAAPI_ATOMIC_READ(&sc->thieves.list.lock) == 0) && KAAPI_ATOMIC_CAS(&sc->thieves.list.lock, 0, 1))
      break ;
    kaapi_slowdown_cpu();
  }
#else
  kaapi_atomic_lock( &sc->thieves.list.lock );
#endif
}

inline static void kaapi_task_unlock_adaptive_steal(kaapi_stealcontext_t* sc)
{
#if 0
  KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
#else
  kaapi_atomic_unlock( &sc->thieves.list.lock );
#endif
}


/** \ingroup TASK
    The function kaapi_frame_isempty() will return non-zero value iff the frame is empty. Otherwise return 0.
    This method MUST be called by the owner thread of the kaapi_frame.
    \param stack IN the pointer to the kaapi_stack_t data structure. 
    \retval !=0 if the stack is empty
    \retval 0 if the stack is not empty or argument is an invalid stack pointer
*/
static inline int kaapi_frame_isempty(const kaapi_frame_t* frame)
{ return (frame->pc <= frame->sp); }



/** \ingroup TASK
    The function kaapi_stack_topframe() will return the top frame (the oldest pushed frame).
    If successful, the kaapi_stack_topframe() function will return a pointer to the top frame
    Otherwise, an 0 is returned to indicate the error.
    \param stack IN a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_frame_t* kaapi_stack_topframe(const kaapi_stack_t* stack)
{
  kaapi_assert_debug( stack != 0 );
  return stack->stackframe;
}


/** \ingroup TASK
    The function kaapi_stack_bottom() will return the top task.
    The bottom task is the first pushed task into the stack.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_stack_bottomtask(const kaapi_stack_t* stack) 
{
  kaapi_assert_debug( stack != 0 );
  return stack->task;
}


/** \ingroup TASK
    The initialize a new stack 
*/
static inline int kaapi_stack_init(
  kaapi_stack_t* stack, 
  kaapi_frame_t* stackframe, 
  kaapi_task_t* stacktask, 
  void* stackdata )
{
  kaapi_frame_t* fp = stackframe;
  stack->sfp        = fp;
  stack->esfp       = fp;
  fp->sp            = fp->pc  = stacktask; /* empty frame */
  fp->sp_data       = (char*)stackdata;                /* empty frame */
  stack->thieffp    = 0;
  kaapi_atomic_initlock( &stack->lock );
  stack->stackframe = stackframe;
  stack->task       = stacktask;
  stack->data       = (char*)stackdata;
  return 0;
}

static inline int kaapi_stack_reset(kaapi_stack_t* stack )
{
  kaapi_frame_t* fp = stack->stackframe;
  stack->sfp        = fp;
  stack->esfp       = fp;
  fp->sp            = fp->pc  = stack->task; /* empty frame */
  fp->sp_data       = stack->data;
  return 0;
}

/* more field are reset than in stack_reset 
*/
static inline int kaapi_stack_clear(kaapi_stack_t* stack )
{
  kaapi_stack_reset( stack );
  kaapi_atomic_initlock( &stack->lock );
  stack->sfp->tasklist= 0;
  stack->thieffp      = 0;
#if defined(KAAPI_USE_JMP)
  stack->jbuf         = 0;
#endif
  return 0;
}


/** Useful
*/
extern int kaapi_task_print( FILE* file, kaapi_task_t* task, kaapi_task_body_t body );

/** \ingroup TASK
    The function kaapi_thread_execframe() execute all the tasks in the thread' stack following
    the RFO order in the closures of the frame [frame_sp,..,sp[
    If successful, the kaapi_thread_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval 0 the execution of the stack frame is completed
*/
extern int kaapi_stack_execframe( kaapi_stack_t* thread );

/** \ingroup TASK
    The function kaapi_threadgroup_execframe() execute all the tasks in the thread' stack following
    using the list of ready tasks.
    If successful, the kaapi_threadgroup_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval EINTR the execution of the stack is interupt and the thread is detached to the kprocessor.
*/
extern int kaapi_threadgroup_execframe( struct kaapi_thread_context_t* thread );

/** \ingroup WS
    This method tries to steal work from the tasks of a stack passed in argument.
    The method iterates through all the tasks in the stack until it found a ready task
    or until the request count reaches 0.
    The current implementation is cooperative or concurrent depending of configuration flag.
    only exported for kaapi_stealpoint.
    \param stack the victim stack
    \param task the current running task (cooperative) or 0 (concurrent)
    \retval the number of positive replies to the thieves
*/
extern int kaapi_sched_stealstack  ( 
  struct kaapi_thread_context_t*       thread, 
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup WS
    Compute if the task is ready using the history of accessed stored in map.
    On return the bitset of paramaters with war dependencies or cw accesses are
    returned.
    \retval the number of non ready parameters
*/
extern size_t kaapi_task_computeready( 
  kaapi_task_t*         task,
  void*                 sp, 
  const kaapi_format_t* task_fmt, 
  unsigned int*         war_param, 
  unsigned int*         cw_param, 
  kaapi_hashmap_t*      map 
);

/** \ingroup TASK
    Try to steal a DFG task using the history of access stored in map.
    If map 0 then the dependencies are not computed, this is used to
    steal tasks from queues in hws because these tasks are assumed to
    be always ready.
*/
extern void kaapi_task_steal_dfg
(
  kaapi_hashmap_t*                     map, 
  struct kaapi_thread_context_t*       thread, 
  kaapi_task_t*                        task,
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup TASK
    Try to steal an adaptive task.
HACK: callback_empty is used for hws strategy where adaptive task may be put
    into queue without control flow that execute them.... see kaapi_hws_queue_fifo for instance.
*/
extern void kaapi_task_steal_adapt
(
  struct kaapi_thread_context_t*       thread, 
  kaapi_task_t*                        task,
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange,
  void                                 (*callback_empty)(kaapi_task_t*)
);

/** \ingroup WS
    Wrapper arround the user level Splitter for Adaptive task
*/
extern int kaapi_task_splitter_adapt( 
    struct kaapi_thread_context_t*       thread, 
    kaapi_task_t*                        task,
    kaapi_task_splitter_t                splitter,
    void*                                argsplitter,
    struct kaapi_listrequest_t*          lrequests, 
    struct kaapi_listrequest_iterator_t* lrrange
);


/** Args for tasksteal
*/
typedef struct kaapi_tasksteal_arg_t {
  struct kaapi_thread_context_t* origin_thread;     /* stack where task was stolen */
  kaapi_task_body_t              origin_body;       /* the stolen task into origin_stack */
  kaapi_task_t*                  origin_task;       /* the stolen task into origin_stack */
  const kaapi_format_t*          origin_fmt;        /* the format of the stolen taskx */
  unsigned int                   war_param;         /* bit i=1 iff it is a w mode with war dependency */
  unsigned int                   cw_param;          /* bit i=1 iff it is a cw mode */
  void*                          copy_task_args;    /* set by tasksteal a copy of the task args */
} kaapi_tasksteal_arg_t;

/** Args for taskstealready
*/
typedef struct kaapi_taskstealready_arg_t {
  struct kaapi_tasklist_t*   master_tasklist; /* the original task list */
  struct kaapi_tasklist_t*   victim_tasklist; /* the victim task list or 0 if steal from the master */
#if defined(TASKLIST_REPLY_ONETD)
  struct kaapi_taskdescr_t*  td;              /* the stolen td */
#else
  struct kaapi_taskdescr_t** td_beg;          /* range of stolen task into origin_tasklist */
  struct kaapi_taskdescr_t** td_end;          /* range of stolen task into origin_tasklist */
#endif
} kaapi_taskstealready_arg_t;

/** User task + args for kaapi_adapt_body. 
    This area represents the part to be sent by the combinator during the remote
    write of the adaptive reply.
    It is store into the field reply->udata of the kaapi_reply_t data structure.
*/
#define KAAPI_REPLY_USER_DATA_SIZE_MAX (KAAPI_REPLY_DATA_SIZE_MAX-sizeof(kaapi_adaptive_thief_body_t))
typedef struct kaapi_taskadaptive_user_taskarg_t {
  /* user defined body */
  kaapi_adaptive_thief_body_t ubody;
  /* user defined args for the body */
  unsigned char               udata[KAAPI_REPLY_USER_DATA_SIZE_MAX];
} kaapi_taskadaptive_user_taskarg_t;



/** \ingroup TASK
    The function kaapi_task_body_isstealable() will return non-zero value iff the task body may be stolen.
    All user tasks are stealable.
    \param body IN a task body
*/
static inline int kaapi_task_isstealable(const kaapi_task_t* task)
{ 
  uintptr_t state = kaapi_task_getstate(task);
  return (state == KAAPI_TASK_STATE_INIT);
}

/** \ingroup TASK
    The function kaapi_task_body_isstealable() will return non-zero value iff the task body may be stolen.
    All user tasks are stealable.
    \param body IN a task body
*/
static inline int kaapi_task_isready(const kaapi_task_t* task)
{ 
  return (((uintptr_t)task->state & ~KAAPI_TASK_STATE_TERM) == 0);
}


#if defined(__cplusplus)
}
#endif

#endif
