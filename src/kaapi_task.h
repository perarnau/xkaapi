/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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


/* Maximal number of recursive calls used to store the stack of frames.
   The value indicates the maximal number of frames that can be pushed
   into the stackframe for each thread.
   
   If an assertion is thrown at runtime, and if this macro appears then
   it is necessary to increase the maximal number of frames in a stack.
*/
#define KAAPI_MAX_RECCALL 131072

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
    If a thief task receive a request for preemption during its execution.
    
    Warning: sizeof state must be less or equal than 8bits.
*/
#define KAAPI_TASK_STATE_INIT       0x0
#define KAAPI_TASK_STATE_EXEC       0x1
#define KAAPI_TASK_STATE_STEAL      0x2
#define KAAPI_TASK_STATE_TERM       0x4
#define KAAPI_TASK_STATE_MERGE      0x8
#define KAAPI_TASK_STATE_SIGNALED   0x20   /* mask: flag to set the task as signaled for preemption */
#define KAAPI_TASK_STATE_LOCKED     0x40   /* mask: to lock the state of a task */

#define KAAPI_TASK_EXECUTION_STATE \
  (KAAPI_TASK_STATE_EXEC|KAAPI_TASK_STATE_STEAL|KAAPI_TASK_STATE_TERM|KAAPI_TASK_STATE_MERGE)


/** Scheduling information for the task:
    It is a bit field with following definition.
    See definition of kaapi_task_flag_t in kaapi.h.
    The internal representation avoid explit representation of exclusive case,
    for instance KAAPI_TASK_S_PREEMPTION and KAAPI_TASK_S_NOPREEMPTION)
    The value here
    - KAAPI_TASK_IS_UNSTEALABLE: defined iff the task is not stealable
    - KAAPI_TASK_IS_SPLITTABLE : defined iff the task is splittable
*/
#define KAAPI_TASK_UNSTEALABLE_MASK    0x01
#define KAAPI_TASK_SPLITTABLE_MASK     0x02  /* means that a splitter can be called */
#define KAAPI_TASK_PREEMPTION_MASK     0x04  /* task with preemption */
#define KAAPI_TASK_COOPERATIVE_MASK    0x08



/**
*/
typedef void (*kaapi_task_body_internal_t)(
  void*           /* task arg*/, 
  kaapi_thread_t* /* thread or stream */, 
  kaapi_task_t*   /* the current task */
);


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

  /* execution state for stack of task */
  kaapi_frame_t*        volatile thieffp __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the thief frame where to steal */
  kaapi_lock_t                   lock;           // __attribute__((aligned(KAAPI_CACHE_LINE)));

} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_stack_t;


/* experimental: to debug. Only call by libkomp */
static inline kaapi_frame_t* kaapi_push_frame( kaapi_stack_t* stack)
{
  kaapi_frame_t* fp = (kaapi_frame_t*)stack->sfp;
  kaapi_task_t* sp = fp->sp;

  /* init new frame for the next task to execute */
  fp[1].pc        = sp;
  fp[1].sp        = sp;
  fp[1].sp_data   = fp->sp_data;

  kaapi_writemem_barrier();
  stack->sfp = ++fp;
  kaapi_assert_debug_fmt( stack->sfp - stack->stackframe <KAAPI_MAX_RECCALL,
       "reccall limit: %i\n", KAAPI_MAX_RECCALL);
  return fp;
}

/* experimental: to debug. Only call by libkomp */
static inline void kaapi_pop_frame( kaapi_stack_t* stack )
{
  kaapi_frame_t* fp = (kaapi_frame_t*)stack->sfp;
#if defined(KAAPI_USE_LOCKTOPOP_FRAME)
  /* lock based pop */
  int tolock = 0;

  /* pop the frame */
  --fp;
  /* finish to execute child tasks, pop current task of the frame */
  stack->sfp = fp;
  tolock = (fp <= stack->thieffp);

  if (tolock)
    kaapi_atomic_lock(&stack->lock);


  if (tolock)
    kaapi_sched_unlock(&stack->lock);

#else //---------#if defined(KAAPI_USE_LOCKTOPOP_FRAME)
  /* THE based pop */
  --fp;
  stack->sfp = fp;
  if (fp <= stack->thieffp)
    kaapi_atomic_waitlock(&stack->lock);
#endif
}


/* ===================== Initialization of adaptive part =============================== */
extern void kaapi_init_adaptive(void);


/* ===================== Default internal task body ==================================== */

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

/** Body of task steal created on thief stack to execute a theft task.
    \ingroup TASK
*/
extern void kaapi_tasksteal_body( void*, kaapi_thread_t* );

/** Body of task steal created on thief stack to execute a task
    theft from ready tasklist
    \ingroup TASK
*/
extern void kaapi_taskstealready_body( void*, kaapi_thread_t* );

/** Write result after a steal and signal the theft task.
    This is used during a task stealing operation.
    This task is pushed into the thief thread by kaapi_tasksteal_body.
    \ingroup TASK
*/
extern void kaapi_taskwrite_body( void*, kaapi_thread_t*, kaapi_task_t* );

/** Body of the task in charge of signaling the end of the task created
    during the split operation. This task body is only used in case of the task
    returned to the thief thread in a splitter is a non adaptive task.
    \ingroup TASK
*/
extern void kaapi_tasksignaladapt_body( void*, kaapi_thread_t* );

/** Body of the merge task pushed after each adaptive task in order to terminate the computation.
    \ingroup TASK
*/
extern void kaapi_taskadaptmerge_body( void*, kaapi_thread_t* );

/** Body of the task to wrapper existing task to be adaptive
    \ingroup TASK
*/
extern void kaapi_taskadapt_body( void*, kaapi_thread_t*, kaapi_task_t* );

/** Body of the task used in interface kaapi_task_begin_adaptive/kaapi_task_end_adaptive
    \ingroup TASK
*/
extern void kaapi_taskbegendadapt_body( void*, kaapi_thread_t*, kaapi_task_t* );

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
static inline int kaapi_task_get_priority(const kaapi_task_t* task)
{ 
  return task->u.s.priority; 
}

/* return the 1+site where task is able to run or 0 if no site */
static inline uint32_t kaapi_task_get_site(const kaapi_task_t* task)
{ 
  return task->u.s.site;
}

/* return 1 if kid is in the site mask */
static inline int kaapi_task_has_arch(const kaapi_task_t* task, int arch)
{ 
  return (task->u.s.arch ==0) || ((task->u.s.arch & (1<<arch)) !=0);
}

static inline int kaapi_task_is_unstealable(kaapi_task_t* task)
{ 
  return (task->u.s.flag & KAAPI_TASK_UNSTEALABLE_MASK) !=0; 
}

static inline void kaapi_task_set_unstealable(kaapi_task_t* task)
{ 
  task->u.s.flag |= KAAPI_TASK_UNSTEALABLE_MASK; 
}

static inline void kaapi_task_unset_unstealable(kaapi_task_t* task)
{ 
  task->u.s.flag &= ~KAAPI_TASK_UNSTEALABLE_MASK; 
}

static inline int kaapi_task_is_splittable(kaapi_task_t* task)
{ 
  return (task->u.s.flag & KAAPI_TASK_SPLITTABLE_MASK) !=0; 
}

static inline void kaapi_task_set_splittable(kaapi_task_t* task)
{ 
  task->u.s.flag |= KAAPI_TASK_SPLITTABLE_MASK; 
}

static inline void kaapi_task_unset_splittable(kaapi_task_t* task)
{ 
  task->u.s.flag &= ~KAAPI_TASK_SPLITTABLE_MASK; 
}

static inline int kaapi_task_is_withpreemption(kaapi_task_t* task)
{ 
  return (task->u.s.flag & KAAPI_TASK_S_PREEMPTION) !=0;
}

static inline kaapi_task_body_t kaapi_task_getbody(const kaapi_task_t* task)
{
  return ((volatile kaapi_task_t*)task)->body;
}

static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_body_t body)
{
  ((volatile kaapi_task_t*)task)->body = body;
}

static inline int kaapi_task_getstate(const kaapi_task_t* task)
{
  return KAAPI_ATOMIC_READ(&task->u.s.state);
}

static inline void kaapi_task_setstate(kaapi_task_t* task, int state)
{
  KAAPI_ATOMIC_WRITE(&task->u.s.state, state);
}

static inline void kaapi_task_setbody_barrier(kaapi_task_t* task, kaapi_task_body_t body)
{
  kaapi_mem_barrier();
  ((volatile kaapi_task_t*)task)->body = body;
}

static inline uintptr_t kaapi_task_casstate(kaapi_task_t* task, uint32_t oldstate, uint32_t newstate)
{
  return KAAPI_ATOMIC_CAS( &task->u.s.state, oldstate, newstate);
}

/* return the old state before atomic op
*/
static inline uintptr_t kaapi_task_orstate(kaapi_task_t* task, uint32_t mask)
{
  return KAAPI_ATOMIC_OR_ORIG( &task->u.s.state, mask);
}

/* return the old state before atomic op
*/
static inline uintptr_t kaapi_task_andstate(kaapi_task_t* task, uint32_t mask)
{
  return KAAPI_ATOMIC_AND_ORIG( &task->u.s.state, mask);
}

/* Return the previous state before adding the exec state.
   If the state returned is 0 then it means that the task was mark exec first.
   Else it returns the previous execution state (steal or term or merge).
   Should ensure excluive mark with respect to marksteal.
*/
static inline int kaapi_task_markexec( kaapi_task_t* task )
{
  return KAAPI_ATOMIC_OR_ORIG( &task->u.s.state, KAAPI_TASK_STATE_EXEC) & KAAPI_TASK_EXECUTION_STATE;
}


/* Return the !=0 iff the task was marked first to be theft.
   Should ensure excluive mark with respect to markexec.
*/
static inline int kaapi_task_marksteal( kaapi_task_t* task )
{
  return 0==(KAAPI_ATOMIC_OR_ORIG( &task->u.s.state, KAAPI_TASK_STATE_STEAL) & KAAPI_TASK_STATE_EXEC);
}


static inline void kaapi_task_markterm( kaapi_task_t* task )
{
  KAAPI_ATOMIC_OR( &task->u.s.state, KAAPI_TASK_STATE_TERM);
}

static inline void kaapi_task_markaftersteal( kaapi_task_t* task )
{
  KAAPI_ATOMIC_OR( &task->u.s.state, KAAPI_TASK_STATE_MERGE);
}

static inline void kaapi_task_markpreempted( kaapi_task_t* task )
{
  KAAPI_ATOMIC_OR( &task->u.s.state, KAAPI_TASK_STATE_SIGNALED);
}

static inline void kaapi_task_lock( kaapi_task_t* task )
{
  while ( (KAAPI_ATOMIC_OR_ORIG( &task->u.s.state, KAAPI_TASK_STATE_LOCKED) & KAAPI_TASK_STATE_LOCKED) != 0)
    ;
}

static inline int kaapi_task_trylock( kaapi_task_t* task )
{
  return (KAAPI_ATOMIC_OR_ORIG( &task->u.s.state, KAAPI_TASK_STATE_LOCKED) & KAAPI_TASK_STATE_LOCKED) == 0;
}

static inline void kaapi_task_unlock( kaapi_task_t* task )
{
  KAAPI_ATOMIC_AND( &task->u.s.state, ~KAAPI_TASK_STATE_LOCKED);
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
  kaapi_frame_t* fp   = stack->stackframe;
  fp->sp              = fp->pc  = stack->task; /* empty frame */
  fp->sp_data         = stack->data;
  fp->tasklist        = 0;
  stack->sfp          = fp;
  stack->esfp         = fp;
  return 0;
}


/* more field are reset than in stack_reset 
*/
static inline int kaapi_stack_clear(kaapi_stack_t* stack )
{
  kaapi_stack_reset( stack );
  stack->thieffp      = 0;
  return 0;
}

/* more field are reset than in stack_reset 
*/
static inline int kaapi_stack_destroy(kaapi_stack_t* stack )
{
  kaapi_atomic_destroylock( &stack->lock );
  return 0;
}


/** Useful
*/
extern int kaapi_task_print( FILE* file, kaapi_task_t* task, kaapi_task_body_t body );

/** \ingroup TASK
    The function kaapi_stack_execframe() execute all the tasks in the thread' stack following
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
    The function kaapi_thread_execframe_tasklist() execute all the tasks in the thread' stack following
    using the list of ready tasks.
    If successful, the kaapi_threadgroup_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval EINTR the execution of the stack is interupt and the thread is detached to the kprocessor.
*/
extern int kaapi_thread_execframe_tasklist( struct kaapi_thread_context_t* thread );

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
  const struct kaapi_format_t* task_fmt, 
  unsigned int*         war_param, 
  unsigned int*         cw_param, 
  kaapi_hashmap_t*      map 
);


/** \ingroup TASK
    Try to steal some work from a frame that may contains tasks
    
    If map 0 then the dependencies are not computed, this is used to
    steal tasks from queues in hws because these tasks are assumed to
    be always ready.

    \param thread the thread that stores the frame
    \param frame the frame to steal
    \param map the history of previous accesses stored in map.
    \param lrequests the set of requests
    \param lrrange the iterator over the requests
*/
extern int kaapi_sched_stealframe
(
  struct kaapi_thread_context_t*       thread, 
  kaapi_frame_t*                       frame, 
  kaapi_hashmap_t*                     map, 
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup TASK
    Try to steal a DFG task using the history of access stored in map.
    If map 0 then the dependencies are not computed, this is used to
    steal tasks from queues in hws because these tasks are assumed to
    be always ready.
    
    \retval 0 if case of successfull steal of the task
    \retval ENOENT the task does not have format and cannot be stolen
    \retval EACCES the task is not ready due to data flow dependency
    \retval EINVAL the task cannot be execute on the any of the requests
    \retval EPERM the task is not stealable: either state is not INIT or flag unstealable is set
    \retval EBUSY the task has not been stolen because some body else has steal it or execute it.
*/
extern int kaapi_sched_steal_task
(
  kaapi_hashmap_t*                     map, 
  const struct kaapi_format_t*         task_fmt,
  kaapi_task_t*                        task,
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup TASK
    Try to split a DFG ready task (INIT and unstealable, or EXEC and splittable).

    \retval 0 if case of successfull split of the task
    \retval EPERM the task cannot be split because no splitter was defined.
*/
extern int kaapi_sched_splittask
(
  const struct kaapi_format_t*         task_fmt,
  kaapi_task_t*                        task,
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);

/** \ingroup TASK
    Try to steal or to split a DFG ready task.

    If map 0 then the dependencies are not computed, this is used to
    steal tasks from queues in hws because these tasks are assumed to
    be always ready.

    \retval 0 if case of successfull steal of the task
    \retval ENOENT the task does not have format and cannot be stolen
    \retval EACCES the task is not ready due to data flow dependency
    \retval EPERM the task is not stealable: either state is not INIT or flag unstealable is set
    \retval EBUSY the task has not been stolen because some body else has steal it or execute it.
*/
extern int kaapi_sched_steal_or_split_task
(
  kaapi_hashmap_t*                     map, 
  const struct kaapi_format_t*         task_fmt,
  kaapi_task_t*                        task,
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange,
  void                                 (*callback_empty)(kaapi_task_t*)  
);


#if 0 // DEPRECATED_ATTRIBUTE
{
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
}
#endif


/** \ingroup WS
    Wrapper arround the user level Splitter for Adaptive task
*/
__attribute__((deprecated))
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
  kaapi_task_body_t              origin_body;       /* the stolen task into origin_stack */
  kaapi_task_t*                  origin_task;       /* the stolen task into origin_stack */
  const struct kaapi_format_t*   origin_fmt;        /* the format of the stolen taskx */
  unsigned int                   war_param;         /* bit i=1 iff it is a w mode with war dependency */
  unsigned int                   cw_param;          /* bit i=1 iff it is a cw mode */
  void*                          copy_task_args;    /* set by tasksteal a copy of the task args */
} kaapi_tasksteal_arg_t;


/** Args for taskstealready
*/
typedef struct kaapi_taskstealready_arg_t {
  struct kaapi_tasklist_t*  master_tasklist;   /* the original task list to signal */
#if defined(TASKLIST_REPLY_ONETD)
  struct kaapi_taskdescr_t*  td;               /* the stolen td */
#else
  struct kaapi_taskdescr_t** td_beg;           /* range of stolen task into origin_tasklist */
  struct kaapi_taskdescr_t** td_end;           /* range of stolen task into origin_tasklist */
#endif
} kaapi_taskstealready_arg_t;


/** Argument for kaapi_taskadapt_body.
    This argument is used to encapsulated any kind of adaptative task in order 
    to add an extra access to the steal context.
    The format of this task is the same as the format of the encapsulated task.
*/
typedef struct kaapi_taskadaptive_arg_t {
  kaapi_access_t                shared_sc;
  kaapi_task_body_t             user_body; 
  void*                         user_sp;  
  kaapi_adaptivetask_splitter_t user_splitter;
} kaapi_taskadaptive_arg_t;


/** Argument for the companion task kaapi_merge_body.
*/
typedef struct kaapi_taskmerge_arg_t {
  kaapi_access_t                shared_sc;
} kaapi_taskmerge_arg_t;


/** Argument for kaapi_taskadapt_body.
    This argument data structure is used for compatibility with the (old) interface
    and the new interface where splittter/arg splitter is part of the task format
    accessors.
*/
typedef struct kaapi_taskbegendadaptive_arg_t {
  kaapi_access_t                shared_sc;      /* must be first as for kaapi_taskadaptive_arg_t */
  kaapi_adaptivetask_splitter_t splitter;
  void*                         argsplitter;
  kaapi_adaptivetask_splitter_t usersplitter;
} kaapi_taskbegendadaptive_arg_t;


/** Args for kaapi_taskstealadapt_body may be dynamically allocated or statically allocated
    in the thief stack.
*/
typedef struct kaapi_thiefadaptcontext_t {
  kaapi_lock_t                        lock;                    /* synchro between victim-thief */
  struct kaapi_thiefadaptcontext_t*   next;                    /* link fields in the victim steal context list */
  struct kaapi_thiefadaptcontext_t*   prev;                    /* */
  kaapi_task_t*                       thief_task;              /* thief task, !=0 until completion or preemption */
  void*                               arg_from_thief;
  void*                               arg_from_victim;
  struct kaapi_thiefadaptcontext_t*   thief_of_the_thief_head; /* list of the thief of the thief */
  struct kaapi_thiefadaptcontext_t*   thief_of_the_thief_tail;
} kaapi_thiefadaptcontext_t __attribute__((aligned(KAAPI_CACHE_LINE)));


/** Adaptive stealing context.
    This context is defined to delegate the management of thief
    and preemption from the user to the runtime.
    The steal context 
    \ingroup ADAPT
*/
typedef struct kaapi_stealcontext_t {
  struct kaapi_stealcontext_t*   msc;
  uintptr_t                      flag;  
  kaapi_thiefadaptcontext_t*     ktr;  /* !=0 iff preemption was used */ 

  /* thieves related context, 2 cases */
  union
  {
    /* 0) an atomic counter if preemption disabled */
    kaapi_atomic_t count;

    /* 1) a thief list if preemption enabled */
    struct
    {
      kaapi_lock_t lock;
      kaapi_thiefadaptcontext_t* head __attribute__((aligned(sizeof(void*))));
      kaapi_thiefadaptcontext_t* tail __attribute__((aligned(sizeof(void*))));
    } list;
  } thieves;
#if defined(KAAPI_DEBUG)
  int version;
  int volatile state;   /* 0 term */
#endif

} kaapi_stealcontext_t __attribute__((aligned(sizeof(intptr_t))));


/** \ingroup TASK
    The function kaapi_task_body_isstealable() will return non-zero value iff the task body may be stolen.
    All user tasks are stealable.
    \param body IN a task body
*/
static inline int kaapi_task_isstealable(const kaapi_task_t* task)
{ 
  uintptr_t state = kaapi_task_getstate(task);
  return (state == KAAPI_TASK_STATE_INIT) && ((task->u.s.flag & KAAPI_TASK_UNSTEALABLE_MASK) ==0);
}

/** \ingroup TASK
    The function kaapi_task_isready() will return non-zero value iff the task may be executed.
    The method assumes that dependency analysis has already verified that it does not exist unstatisfied
    data flow constraint.
    \param task IN a pointer to a task
*/
static inline int kaapi_task_isready(const kaapi_task_t* task)
{ 
  uintptr_t state = kaapi_task_getstate(task);
  return  ((state & KAAPI_TASK_EXECUTION_STATE) == 0)
       || ((state & KAAPI_TASK_STATE_TERM) != 0) 
       || ((state & KAAPI_TASK_STATE_MERGE) != 0);
}

/*
*/
extern void kaapi_init_adapfmt(void);
extern void kaapi_init_adaptmergefmt(void);
extern void kaapi_init_begendadapfmt(void);

/* */
extern void kaapi_register_staticschedtask_format(void);

extern kaapi_task_body_t kaapi_task_stsched_get_body_by_arch
(
  const struct kaapi_taskdescr_t* const td,
  unsigned int arch
);

extern kaapi_task_body_t kaapi_task_stsched_get_bodywh_by_arch
(
  const struct kaapi_taskdescr_t* const td,
  unsigned int arch
);

#if defined(__cplusplus)
}
#endif

#endif
