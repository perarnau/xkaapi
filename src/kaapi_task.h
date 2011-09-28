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
/** The stack of tasks data structure
    This data structure should be extend in case where the C-stack is required to be suspended and resumed.
    This data structure is always at position ((kaapi_thread_context_t*)stackaddr) - 1 of stack at address
    stackaddr.
    It was made opaque to the user API because we do not want to expose the way we execute stack in the
    user code.
    
    WARNING: sfp should be the first field of the data structure in order to be able to recover in the public
    API sfp (<=> kaapi_thread_t*) from the kaapi_stack_t pointer stored in kaapi_current_thread_key.
*/
typedef struct kaapi_stack_t {
  kaapi_frame_t*        volatile sfp;            /** pointer to the current frame (in stackframe) */
  kaapi_frame_t*                 esfp;           /** first frame until to execute all frame  */
  kaapi_atomic_t                 lock;           // __attribute__((aligned(KAAPI_CACHE_LINE)));
  kaapi_frame_t*                 stackframe;     /** for execution, see kaapi_thread_execframe */
  struct kaapi_processor_t*      proc;           /** access to the running processor */
  kaapi_task_t*                  task;           /** bottom of the stack of task */
  char*                          data;           /** begin of stack of data */ 

  /* execution state for stack of task */
  kaapi_frame_t*        volatile thieffp __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the thief frame where to steal */

} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_stack_t;


/* ===================== Default internal task body ==================================== */
/** Body of the nop task 
    \ingroup TASK
*/
extern void kaapi_nop_body( void*, kaapi_thread_t*);

/** Body of the startup task 
    \ingroup TASK
*/
extern void kaapi_taskstartup_body( void*, kaapi_thread_t*);

/** Body of the task that mark a task to suspend execution
    \ingroup TASK
*/
extern void kaapi_suspend_body( void*, kaapi_thread_t*);

/** Body of the task that mark a task as under execution
    \ingroup TASK
*/
extern void kaapi_exec_body( void*, kaapi_thread_t*);

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
extern void kaapi_taskwrite_body( void*, kaapi_thread_t* );

/** Merge result after a steal
    \ingroup TASK
*/
extern void kaapi_aftersteal_body( void*, kaapi_thread_t* );

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


/* ============================= Implementation method ============================ */
/** Note: a body is a pointer to a function. We assume that a body <=> void* and has
    64 bits on 64 bits architecture.
    The 4 highest bits are used to store the state of the task :
    - 0000 : the task has been pushed on the stack (<=> a user pointer function has never high bits == 11)
    - 0100 : the task has been execute either by the owner / either by a thief
    - 1000 : the task has been steal by a thief for execution
    - 1100 : the task has been theft by a thief for execution and it was executed, the body is "aftersteal body"
    
    Other reserved bits are :
    - 0010 : the task is terminated. This state if only put for debugging.
*/
#if (__SIZEOF_POINTER__ == 4)
#  define KAAPI_MASK_BODY_TERM    (0x1)
#  define KAAPI_MASK_BODY_PREEMPT (0x2) /* must be different from term */
#  define KAAPI_MASK_BODY_AFTER   (0x2)
#  define KAAPI_MASK_BODY_EXEC    (0x4)
#  define KAAPI_MASK_BODY_STEAL   (0x8)

#elif (__SIZEOF_POINTER__ == 8)

#  define KAAPI_MASK_BODY_TERM    (0x1UL << 60UL)
#  define KAAPI_MASK_BODY_PREEMPT (0x2UL << 60UL) /* must be different from term */
#  define KAAPI_MASK_BODY_AFTER   (0x2UL << 60UL)
#  define KAAPI_MASK_BODY_EXEC    (0x4UL << 60UL)
#  define KAAPI_MASK_BODY_STEAL   (0x8UL << 60UL)
#  define KAAPI_MASK_BODY         (0xFUL << 60UL)
#  define KAAPI_MASK_BODY_SHIFTR   58UL

#else
#  error "No implementation for pointer to function with size greather than 8 bytes. Please contact the authors."
#endif

/** \ingroup TASK
*/
/**@{ */

#if (__SIZEOF_POINTER__ == 4)
static inline uintptr_t kaapi_task_getstate(const kaapi_task_t* task)
{
  return KAAPI_ATOMIC_READ(&task->u.state);
}

static inline void kaapi_task_setstate(kaapi_task_t* task, uintptr_t state)
{
  KAAPI_ATOMIC_WRITE(&task->u.state, state);
}

static inline void kaapi_task_setstate_barrier(kaapi_task_t* task, uintptr_t state)
{
  KAAPI_ATOMIC_WRITE_BARRIER(&task->u.state, state);
}

static inline unsigned int kaapi_task_state_issteal(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_STEAL;
}

static inline unsigned int kaapi_task_state_isexec(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_EXEC;
}

static inline unsigned int kaapi_task_state_isterm(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_TERM;
}

static inline unsigned int kaapi_task_state_isaftersteal(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_AFTER;
}

static inline unsigned int kaapi_task_state_ispreempted(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_PREEMPT;
}

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

static inline unsigned int kaapi_task_state2int(uintptr_t state)
{
  return (unsigned int)state;
}

#define kaapi_task_state_setsteal(__state)	\
    ((__state) | KAAPI_MASK_BODY_STEAL)

#define kaapi_task_state_setexec(__state)       \
    ((__state) | KAAPI_MASK_BODY_EXEC)

#define kaapi_task_state_setterm(__state)       \
    ((__state) | KAAPI_MASK_BODY_TERM)

#define kaapi_task_state_setafter(__state)      \
    ((__state) | KAAPI_MASK_BODY_AFTER)

/** \ingroup TASK
    Set the body of the task
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body)
{
  KAAPI_ATOMIC_WRITE(&task->u.state, 0);
  task->u.body  = body;
}

/** \ingroup TASK
    Get the body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(const kaapi_task_t* task)
{
  return task->u.body;
}

#elif (__SIZEOF_POINTER__ ==8) /* __SIZEOF_POINTER__ == 8 */
#define kaapi_task_getstate(task)\
      KAAPI_ATOMIC_READ(&(task)->u.state)

#define kaapi_task_setstate(task, value)\
      KAAPI_ATOMIC_WRITE(&(task)->u.state,(value))

#define kaapi_task_setstate_barrier(task, value)\
      { kaapi_writemem_barrier(); (task)->u.state = (value); }

#define kaapi_task_state_issteal(state)       \
      ((state) & KAAPI_MASK_BODY_STEAL)

#define kaapi_task_state_isexec(state)        \
      ((state) & KAAPI_MASK_BODY_EXEC)

#define kaapi_task_state_isterm(state)        \
      ((state) & KAAPI_MASK_BODY_TERM)

#define kaapi_task_state_isaftersteal(state)  \
      ((state) & KAAPI_MASK_BODY_AFTER)

#define kaapi_task_state_ispreempted(state)  \
      ((state) & KAAPI_MASK_BODY_PREEMPT)

#define kaapi_task_state_isspecial(state)     \
      (state & KAAPI_MASK_BODY)

#define kaapi_task_state_isnormal(state)     \
      ((state & KAAPI_MASK_BODY) ==0)

/* this macro should only be called on a theft task to determine if it is ready */
#define kaapi_task_state_isready(state)       \
      ((((state) & KAAPI_MASK_BODY)==0) || (((state) & (KAAPI_MASK_BODY_AFTER|KAAPI_MASK_BODY_TERM)) !=0))

#define kaapi_task_state_isstealable(state)   \
      (((state) & (KAAPI_MASK_BODY_STEAL|KAAPI_MASK_BODY_EXEC)) ==0)

static inline unsigned int kaapi_task_state2int(uintptr_t state)
{
  return (unsigned int)state;
}

#define kaapi_task_state_setsteal(state)      \
    ((state) | KAAPI_MASK_BODY_STEAL)

#define kaapi_task_state_setexec(state)       \
    ((state) | KAAPI_MASK_BODY_EXEC)

#define kaapi_task_state_setterm(state)       \
    ((state) | KAAPI_MASK_BODY_TERM)

#define kaapi_task_state_setafter(state)       \
    ((state) | KAAPI_MASK_BODY_AFTER)

#define kaapi_task_state_setocr(state)       \
    ((state) | KAAPI_MASK_BODY_OCR)
    
#define kaapi_task_body2state(body)           \
    ((uintptr_t)body)

#define kaapi_task_state2body(state)           \
    ((kaapi_task_body_t)(state))

#define kaapi_task_state2int(state)            \
    ((int)(state >> KAAPI_MASK_BODY_SHIFTR))

/** Set the body of the task
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->u.body = body;
}

/** Get the body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(const kaapi_task_t* task)
{
  return kaapi_task_state2body( kaapi_task_getstate(task) & ~KAAPI_MASK_BODY );
}
/**@} */

#else /* __SIZEOF_POINTER__ == 8 */
#error "I'm here"
#endif 


/** Atomically: OR of the task state with the value in 'state' and return the previous value.
*/
static inline uintptr_t kaapi_task_andstate( kaapi_task_t* task, uintptr_t state )
{
  const uintptr_t retval =
  KAAPI_ATOMIC_ANDPTR_ORIG(&task->u.state, state);
  return retval;
}

static inline uintptr_t kaapi_task_orstate( kaapi_task_t* task, uintptr_t state )
{
#if defined(__i386__)||defined(__x86_64)
  /* WARNING: here we assume that the locked instruction do a writememory barrier */
#else
  kaapi_writemem_barrier();
#endif

  const uintptr_t retval =
  KAAPI_ATOMIC_ORPTR_ORIG(&task->u.state, state);

  return retval;
}

static inline int kaapi_task_teststate( kaapi_task_t* task, uintptr_t state )
{
  /* assume a mem barrier has been done */
  return (kaapi_task_getstate( task ) & state) !=0;
}

inline static void kaapi_task_lock_adaptive_steal(kaapi_stealcontext_t* sc)
{
  while (1)
  {
    if ((KAAPI_ATOMIC_READ(&sc->thieves.list.lock) == 0) && KAAPI_ATOMIC_CAS(&sc->thieves.list.lock, 0, 1))
      break ;
    kaapi_slowdown_cpu();
  }
}

inline static void kaapi_task_unlock_adaptive_steal(kaapi_stealcontext_t* sc)
{
  KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
}


/** Should be only use in debug mode
    - other bodies should be added
*/
#if !defined(KAAPI_NDEBUG)
static inline int kaapi_isvalid_body( kaapi_task_body_t body)
{
  return 
    (kaapi_format_resolvebybody( body ) != 0) 
      || (body == kaapi_taskmain_body)
      || (body == kaapi_tasksteal_body)
      || (body == kaapi_taskwrite_body)
      || (body == kaapi_aftersteal_body)
      || (body == kaapi_nop_body)
      || (body == kaapi_taskmove_body)
      || (body == kaapi_taskalloc_body)
  ;
}
#else
static inline int kaapi_isvalid_body( kaapi_task_body_t body)
{
  return 1;
}
#endif

/** \ingroup TASK
    The function kaapi_frame_isempty() will return non-zero value iff the frame is empty. Otherwise return 0.
    \param stack IN the pointer to the kaapi_stack_t data structure. 
    \retval !=0 if the stack is empty
    \retval 0 if the stack is not empty or argument is an invalid stack pointer
*/
static inline int kaapi_frame_isempty(volatile kaapi_frame_t* frame)
{ return (frame->pc <= frame->sp); }

/** \ingroup TASK
    The function kaapi_stack_bottom() will return the top task.
    The bottom task is the first pushed task into the stack.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_thread_bottomtask(kaapi_stack_t* stack) 
{
  kaapi_assert_debug( stack != 0 );
  return stack->task;
}



static inline int kaapi_stack_reset(kaapi_stack_t* st )
{
  kaapi_frame_t* fp = st->stackframe;
  st->sfp          = fp;
  st->esfp         = fp;
  fp->sp           = fp->pc  = st->task; /* empty frame */
  fp->sp_data      = st->data;                /* empty frame */
  st->thieffp      = 0;

  return 0;
}


static inline int kaapi_stack_clear(kaapi_stack_t* st )
{
  kaapi_stack_reset( st );
  kaapi_atomic_initlock( &st->lock );
  st->sfp->tasklist= 0;
  return 0;
}


/** Useful
*/
extern int kaapi_task_print( FILE* file, kaapi_task_t* task );

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
extern int kaapi_thread_execframe( struct kaapi_thread_context_t* thread );

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
  struct kaapi_thread_context_t* thread, 
  kaapi_task_t* curr, 
  struct kaapi_listrequest_t* lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup WS
    Splitter for DFG task
*/
extern int kaapi_task_splitter_dfg(
  struct kaapi_thread_context_t*       thread, 
  kaapi_task_t*                 task, 
  const kaapi_format_t*         task_fmt,
  unsigned int                  war_param, 
  unsigned int                  cw_param, 
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);

/** \ingroup TASK
    Splitter for a single DFG
*/
extern void kaapi_task_splitter_dfg_single
(
  struct kaapi_thread_context_t*       thread, 
  kaapi_task_t*                 task, 
  const kaapi_format_t*         task_fmt,
  unsigned int                  war_param, 
  unsigned int                  cw_param, 
  kaapi_request_t*		        request
);

/** \ingroup WS
    Wrapper arround the user level Splitter for Adaptive task
*/
extern int kaapi_task_splitter_adapt( 
    struct kaapi_thread_context_t*       thread, 
    kaapi_task_t*                 task,
    kaapi_task_splitter_t         splitter,
    void*                         argsplitter,
    struct kaapi_listrequest_t*          lrequests, 
    struct kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup WS
    Splitter arround tasklist stealing
*/
extern int kaapi_task_splitter_readylist( 
  struct kaapi_thread_context_t*       thread, 
  struct kaapi_tasklist_t*      tasklist,
  struct kaapi_taskdescr_t**    task_beg,
  struct kaapi_taskdescr_t**    task_end,
  struct kaapi_listrequest_t*          lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange,
  size_t                        countreq
);



/** Args for tasksteal
*/
typedef struct kaapi_tasksteal_arg_t {
  struct kaapi_thread_context_t* origin_thread;     /* stack where task was stolen */
  kaapi_task_t*                  origin_task;       /* the stolen task into origin_stack */
  const kaapi_format_t*          origin_fmt;        /* the format of the stolen taskx */
  unsigned int                   war_param;         /* bit i=1 iff it is a w mode with war dependency */
  unsigned int                   cw_param;          /* bit i=1 iff it is a cw mode */
  void*                          copy_task_args;    /* set by tasksteal a copy of the task args */
} kaapi_tasksteal_arg_t;

/** Args for taskstealready
*/
typedef struct kaapi_taskstealready_arg_t {
  struct kaapi_tasklist_t*   origin_tasklist; /* the original task list */
  struct kaapi_taskdescr_t** origin_td_beg;   /* range of stolen task into origin_tasklist */
  struct kaapi_taskdescr_t** origin_td_end;   /* range of stolen task into origin_tasklist */
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
static inline int kaapi_task_body_isstealable(kaapi_task_body_t body)
{ 
  return (body != kaapi_taskstartup_body) 
      && (body != kaapi_nop_body)
      && (body != kaapi_tasksteal_body) 
      && (body != kaapi_taskwrite_body)
      && (body != kaapi_taskfinalize_body) 
      && (body != kaapi_adapt_body)
      ;
}

/** \ingroup TASK
    The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
    All previous internal task body are not stealable. All user task are stealable.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isstealable(const kaapi_task_t* task)
{
#if (__SIZEOF_POINTER__ == 4)

  return kaapi_task_state_isstealable(kaapi_task_getstate(task)) &&
   kaapi_task_body_isstealable(task->u.body);

#else /* __SIZEOF_POINTER__ == 8 */

  const uintptr_t state = (uintptr_t)task->u.body;
  return kaapi_task_state_isstealable(state) &&
    kaapi_task_body_isstealable(kaapi_task_state2body(state));

#endif
}



#if defined(__cplusplus)
}
#endif

#endif
