/*
** kaapi_task.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#if !defined(KAAPI_TASK_H)
#define KAAPI_TASK_H
#include <stdint.h>
#include "kaapi_error.h"
//#include "kaapi_config.h"
//#include "kaapi_type.h"

#define KAAPI_MAX_ARCH 3
#define KAAPI_MAX_TASK_PARAMETERS 16

  /* Kaapi name for stdint typedefs.
   */
  typedef uint8_t  kaapi_uint8_t;
  typedef uint16_t kaapi_uint16_t;
  typedef uint32_t kaapi_uint32_t;
  typedef int8_t   kaapi_int8_t;
  typedef int16_t  kaapi_int16_t;
  typedef int32_t  kaapi_int32_t;

/* --------------------------------------------------------------- */

#if defined(__cplusplus)
extern "C" {
#endif

struct kaapi_task_t;
struct kaapi_stack_t;

/* ========================================================================= */
/** State of a task 
   \ingroup TASK
    - KAAPI_TASK_INIT: the task may be executed or stolen.
    - KAAPI_TASK_STOLEN: the task has been stolen.
    - KAAPI_TASK_EXEC: the task is under execution.
    - KAAPI_TASK_TERM: the task has been executed.
    - KAAPI_TASK_WAITING: the task as unsatisfied condition to be executed. 
      In this case, the body points to the function to evaluate the condition
      and pdata[0] points to data structure necessary to evalute the condition.
      After execution of this function, the state of the task may have changed to INIT
      and both the task body and pdata[0] has been restaured.
    - KAAPI_TASK_SUSPENDED: the task is suspending its execution. This normally occurs
      only for adaptive task when it cannot directly process interrupt.
*/
/*@{*/
#define KAAPI_TASK_INIT      0
#define KAAPI_TASK_STOLEN    1
#define KAAPI_TASK_EXEC      2
#define KAAPI_TASK_TERM      3
#define KAAPI_TASK_WAITING   4
#define KAAPI_TASK_SUSPENDED 5
/*@}*/

/** Flags for task
   \ingroup TASK 
   DEFAULT flags is for normal task that can be stolen and executed every where.
    - KAAPI_TASK_F_STICKY: if set, the task could not be theft else the task can (default).
    - KAAPI_TASK_F_ADAPTIVE: if set, the task is an adaptative task that could be stolen or preempted.
    - KAAPI_TASK_F_LOCALITY: if set, the task as locality constraint defined in locality data field.
    - KAAPI_TASK_F_SYNC: if set, the task do not engender synchronisation, victim should stop on a stolen task
    before continuing the fast execution using RFO schedule.
*/
/*@{*/
#define KAAPI_TASK_F_STICKY     (0x1 <<8)
#define KAAPI_TASK_F_ADAPTIVE   (0x2 <<8)
#define KAAPI_TASK_F_LOCALITY   (0x4 <<8)
#define KAAPI_TASK_F_SYNC       (0x8 <<8)
/*@}*/


/** Constants for number for fixed size parameter of a task
    \ingroup TASK
*/
/*@{*/
#define KAAPI_TASK_MAX_DATA  24 /* allows 4 double or 4 pointers of 8 byte size */
#define KAAPI_TASK_MAX_BDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint8_t))
#define KAAPI_TASK_MAX_SDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint16_t))
#define KAAPI_TASK_MAX_IDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint32_t))
#define KAAPI_TASK_MAX_DDATA (KAAPI_TASK_MAX_DATA/sizeof(double))
#define KAAPI_TASK_MAX_FDATA (KAAPI_TASK_MAX_DATA/sizeof(float))
#define KAAPI_TASK_MAX_PDATA (KAAPI_TASK_MAX_DATA/sizeof(void*))
/*@}*/

/* ========================================================================= */
struct kaapi_task_t;
struct kaapi_stack_t;


/** Kaapi stack of tasks definition
   \ingroup STACK
   The stack store list of tasks as well as a stack of data.
   Both sizes are fixed at initialization of the stack object.
   The stack is truly a stack when used in conjonction with frame.
   A frame capture the state (pc, sp, sp_data) of the stack in order
   to restore it. The implementation also used kaapi_retn_body in order 
   to postpone the restore operation after a set of tasks (see kaapi_stack_taskexecall).

   Before and after the execution of a task, the state of the computation is only
   defined by the stack state (pc, sp, sp_data and the content of the stack). 
   The C-stack doesnot need to be saved.
*/
typedef struct kaapi_stack_t {
  struct kaapi_task_t* pc;             /** task counter: next task to execute, 0 if empty stack */
  struct kaapi_task_t* sp;             /** stack counter: next free task entry */
#if defined(KAAPI_DEBUG)
  struct kaapi_task_t* end_sp;         /** past the last stack counter: next entry after the last task in stack array */
#endif
  struct kaapi_task_t* task;           /** stack of tasks */

  char*         sp_data;               /** stack counter for the data: next free data entry */
#if defined(KAAPI_DEBUG)
  char*         end_sp_data;           /** past the last stack counter: next entry after the last task in stack array */
#endif
  char*         data;                  /** stack of data with the same scope than task */
} kaapi_stack_t;


/* ========================================================================= */
/** Kaapi frame definition
   \ingroup STACK
*/
typedef struct kaapi_frame_t {
    struct kaapi_task_t* pc;
    struct kaapi_task_t* sp;
    char*                sp_data;
} kaapi_frame_t;


/* ========================================================================= */
/** Task body
    \ingroup TASK
*/
#if 0
typedef void (*kaapi_task_body_t)(struct kaapi_task_t* /*task*/, struct kaapi_stack_t* /* thread */);
#endif
//typedef void (*kaapi_task_body_t)(void* /*sp*/, struct kaapi_stack_t* /* thread */);
typedef void (*kaapi_task_body_t)();

typedef void (*kaapi_func_t)(void);

#if 0
/** Kaapi task definition
    \ingroup TASK
    A Kaapi task is the basic unit of computation. It has a constant size including some task's specific values.
    Variable size task has to store pointer to the memory where found extra data.
    The body field is the pointer to the function to execute. The special value 0 correspond to a nop instruction.
*/
typedef struct kaapi_task_t {
  /* state is the public member: initialized with the values above. It is used
     to initializ. istate is the internal state.
*/     
  kaapi_uint32_t     format;   /** */
  union {
    kaapi_uint16_t   locality; /** locality number see documentation */
    kaapi_uint16_t   event;    /** in case of adaptive task */
  } le; /* locality & event */
  union {
    kaapi_uint16_t   state;    /** State of the task see above + flags in order to initialze both in 1 op */
    struct { /* order of following fields may depend on the architecture: what should be defined is
                if state == KAAPI_TASK_TERM then xstate == KAAPI_TASK_TERM */
      kaapi_uint8_t  xstate;   /** State of the task see above */
      kaapi_uint8_t  flags;    /** flags of the task see above */
    } sf;
  } sf; /* state & flag */
  kaapi_task_body_t  body;     /** C function that represent the body to execute*/
/*  char*              sp_data;  /** data stack pointer of the data frame for the task  */
  union { /* union data to view task's immediate data with type information. Be carreful: this is an (anonymous) union  */
    kaapi_uint8_t  bdata[ KAAPI_TASK_MAX_BDATA ];
    kaapi_uint16_t sdata[ KAAPI_TASK_MAX_SDATA ];
    kaapi_uint32_t idata[ KAAPI_TASK_MAX_IDATA ];
    double         ddata[ KAAPI_TASK_MAX_DDATA ];
    float          fdata[ KAAPI_TASK_MAX_FDATA ];
    void*          pdata[ KAAPI_TASK_MAX_PDATA ];
    kaapi_func_t   rdata[ KAAPI_TASK_MAX_PDATA ];
    struct kaapi_frame_t           frame;
  } param;
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_task_t;
#else
typedef struct kaapi_task_t {
  /* state is the public member: initialized with the values above. It is used
     to initializ. istate is the internal state.
*/     
/*  kaapi_uint16_t    format;*/   /** */
#if 0  
union { /* union data to view task's immediate data with type information. Be carreful: this is an (anonymous) union  */
    kaapi_uint8_t  bdata[ KAAPI_TASK_MAX_BDATA ];
    kaapi_uint16_t sdata[ KAAPI_TASK_MAX_SDATA ];
    kaapi_uint32_t idata[ KAAPI_TASK_MAX_IDATA ];
    double         ddata[ KAAPI_TASK_MAX_DDATA ];
    float          fdata[ KAAPI_TASK_MAX_FDATA ];
    void*          pdata[ KAAPI_TASK_MAX_PDATA ];
    kaapi_func_t   rdata[ KAAPI_TASK_MAX_PDATA ];
    struct kaapi_frame_t           frame;
    char*             sp_data;  /** data stack pointer of the data frame for the task  */
  } param;
#endif
  int               event;     /** data stack pointer of the data frame for the task  */
  void*             sp;        /** data stack pointer of the data frame for the task  */
  kaapi_task_body_t body[4];   /** C function that represent the body to execute: 0 exec, 1 steal, 2: stop, 3: default */
} /*__attribute__((aligned (64)))*/ kaapi_task_t;
#endif


/** Kaapi Access mode
    \ingroup TASK
*/
typedef enum kaapi_access_mode_t {
  KAAPI_ACCESS_MODE_VOID= 0,        /* 0000 0000 : */
  KAAPI_ACCESS_MODE_V   = 1,        /* 0000 0001 : */
  KAAPI_ACCESS_MODE_R   = 2,        /* 0000 0010 : */
  KAAPI_ACCESS_MODE_W   = 4,        /* 0000 0100 : */
  KAAPI_ACCESS_MODE_CW  = 8,        /* 0000 1000 : */
  KAAPI_ACCESS_MODE_P   = 16        /* 0001 0000 : */
} kaapi_access_mode_t;

#define KAAPI_ACCESS_MASK_MODE   0x1F
#define KAAPI_ACCESS_MASK_MODE_R 0x2
#define KAAPI_ACCESS_MASK_MODE_W 0x4
#define KAAPI_ACCESS_MASK_MODE_P 0x10

#define KAAPI_ACCESS_GET_MODE( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE )

#define KAAPI_ACCESS_IS_READ( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_R)

#define KAAPI_ACCESS_IS_WRITE( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_W)

#define KAAPI_ACCESS_IS_POSTPONED( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_P)

#define KAAPI_ACCESS_IS_ONLYWRITE( m ) \
  (((m) & KAAPI_ACCESS_MASK_MODE_W) && !((m) & KAAPI_ACCESS_MASK_MODE_R))


/** Kaapi global data
    \ingroup TASK
*/
typedef struct kaapi_gd_t {
    void*               data;         /* access used by stolen task */
} kaapi_gd_t;

/** Kaapi access
    \ingroup TASK
*/
typedef struct kaapi_access_t {
    union {
      kaapi_gd_t* gd;       /* normal access */
      void*       data;     /* access used by stolen task */
    } a;
    /* optional field that depend on the format of the access, e.g. here range is allowed */
} kaapi_access_t;


/** Offset to access to parameter of a task
*/
typedef kaapi_uint32_t kaapi_offset_t;

/** Predefined proc type
*/
#define KAAPI_PROC_TYPE_CPU   0
#define KAAPI_PROC_TYPE_GPU   1
#define KAAPI_PROC_TYPE_MPSOC 2

/** Kaapi task format
    \ingroup TASK
    The format should be 1/ declared 2/ register before any use in task.
    The format object is only used in order to interpret stack of task.
    When a task is copied from one stack to an other one, the format allows to find memory address for each parameter.
    Given the size of each parameters, and the number of parameters the format object keep track where the parameter
    is located (in stack or in immediate parameter of the task structure).
    NOTE if a task should pass parameter onto the stack the offset values correspond to offset with respect to the
    data stack pointer where is first push the first data into the stack.
    
    After registration, the offsets of parameters are computed and:
    - if  offset & MASK_IN_STACK == 1 -> the data is in the stack at offset (offset & ~MASK_IN_STACK)
    - if  offset & MASK_IN_STACK == 0 -> the data is in the task structure at offset (offset & ~MASK_IN_STACK) of the address of the task
*/
typedef struct kaapi_task_format_t {
  kaapi_uint16_t             fmtid;                                   /* identifier of the format */
  short                      isinit;                                  /* ==1 iff initialize */
  const char*                name;                                    /* debug information */
  kaapi_task_body_t          entrypoint[KAAPI_MAX_ARCH];              /* maximum architecture considered in the configuration */
  const kaapi_access_mode_t  mode_params[KAAPI_MAX_TASK_PARAMETERS];  /* only consider value with mask 0xF0 */
  const kaapi_offset_t       params[KAAPI_MAX_TASK_PARAMETERS];       /* access to the i-th parameter: a value or a shared */
  const kaapi_uint32_t       size_params[KAAPI_MAX_TASK_PARAMETERS];  /* sizeof of each params */
  const kaapi_uint32_t       size_allparams;                          /* sizeof of all params */
  const kaapi_uint32_t       size_allaccess;                          /* sizeof of all params of mode access */
  int                        count_params;     /* */
} kaapi_dfg_format_closure_t;


  /* ======================= Required functions on task ============================== */

  /** Body of the nop task 
      \ingroup TASK
  */
  extern void kaapi_nop_body( void*, kaapi_stack_t*);

  /** Body of the task that restore the frame pointer 
      \ingroup TASK
  */
  extern void kaapi_retn_body( void*, kaapi_stack_t*);

  /** Body of the task that mark a task to suspend execution
      \ingroup TASK
  */
  extern void kaapi_suspend_body( void*, kaapi_stack_t*);


  /** \ingroup TASK
      Return a reference to the state of the task.
  */
  #define kaapi_task_state(task) ((task)->sf.sf.xstate)

  /** \ingroup TASK
      The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
      If the task pointer is an invalid pointer, then the function will return 0 as if the task may not be stolen.
      \param task IN a pointer to the kaapi_task_t to test.
  */
#if 0
  static inline int kaapi_task_isstealable(const kaapi_task_t* task)
  { return (task !=0) && !(task->sf.sf.flags & (KAAPI_TASK_F_STICKY>>8)) && (task->body != &kaapi_retn_body); }

  /** \ingroup TASK
      The function kaapi_task_haslocality() will return non-zero value iff the task has locality constraints.
      In this case, the field locality my be read to resolved locality constraints.
      If the task pointer is an invalid pointer, then the function will return 0 as if the task has no locality constraints.
      \param task IN a pointer to the kaapi_task_t to test.
  */
  inline extern int kaapi_task_haslocality(const kaapi_task_t* task)
  { return (task == 0 ? 0 : task->sf.sf.flags & (KAAPI_TASK_F_LOCALITY>>8)); }

  /** \ingroup TASK
      The function kaapi_task_isadaptive() will return non-zero value iff the task is an adaptive task.
      If the task pointer is an invalid pointer, then the function will return 0 as if the task is not an adaptive task.
      \param task IN a pointer to the kaapi_task_t to test.
  */
  inline extern int kaapi_task_isadaptive(const kaapi_task_t* task)
  { return (task == 0 ? 0 : task->sf.sf.flags & (KAAPI_TASK_F_ADAPTIVE>>8)); }

#endif

  /* ========================================================================= */


  /* ======================= Required functions on stack ============================== */

  /** \ingroup STACK
      The function kaapi_stack_alloc() allocates in the heap a stack with at most count number of tasks.
      If successful, the kaapi_stack_alloc() function will return zero.  
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t to allocate.
      \param count_task IN the maximal number of tasks in the stack.
      \param size_data IN the amount of stack data.
      \retval ENOMEM cannot allocate memory.
      \retval EINVAL invalid argument: bad stack pointer or capacity is 0.
  */
  extern int kaapi_stack_alloc( kaapi_stack_t* stack, kaapi_uint32_t count_task, kaapi_uint32_t size_data );

  /** \ingroup STACK
      The function kaapi_stack_free() free the stack successfuly allocated with kaapi_stack_alloc.
      If successful, the kaapi_stack_free() function will return zero.  
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t to allocate.
      \retval EINVAL invalid argument: bad stack pointer.
  */
  extern int kaapi_stack_free( kaapi_stack_t* stack );


  /** \ingroup STACK
      The function kaapi_stack_init() initializes the stack using the buffer passed in parameter. 
      The buffer must point to a memory region with at least count bytes allocated.
      If successful, the kaapi_stack_init() function will return zero and the buffer should
      never be used again.
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t to initialize.
      \param size_task_buffer IN the size in bytes of the buffer for the tasks.
      \param task_buffer INOUT the buffer to used to store the stack of tasks.
      \param size_data_buffer IN the size in bytes of the buffer for the data.
      \param data_buffer INOUT the buffer to used to store the stack of data.
      \retval EINVAL invalid argument: bad stack pointer or count is not enough to store at least one task or buffer is 0.
  */
  extern int kaapi_stack_init( kaapi_stack_t* stack,  
                               kaapi_uint32_t size_task_buffer, void* task_buffer,
                               kaapi_uint32_t size_data_buffer, void* data_buffer 
  );


  /** \ingroup STACK
      The function kaapi_stack_clear() clears the stack.
      If successful, the kaapi_stack_clear() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t to clear.
      \retval EINVAL invalid argument: bad stack pointer.
  */
  extern int kaapi_stack_clear( kaapi_stack_t* stack );

  /** \ingroup STACK
      The function kaapi_stack_isempty() will return non-zero value iff the stack is empty. Otherwise return 0.
      If the argument is a bad pointer then the function kaapi_stack_isempty returns a non value as if the stack was empty.
      \param stack IN the pointer to the kaapi_stack_t data structure. 
      \retval !=0 if the stack is empty
      \retval 0 if the stack is not empty or argument is an invalid stack pointer
  */
  static inline int kaapi_stack_isempty(const kaapi_stack_t* stack)
  {
    return (stack !=0) && (stack->pc >= stack->sp);
  }

  /** \ingroup TASK
      The function kaapi_stack_bottom() will return the top task.
      The top task is not part of the stack.
      If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
      Otherwise, an 0 is returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval a pointer to the next task to push or 0.
  */
#if defined(KAAPI_DEBUG)
  static inline kaapi_task_t* kaapi_stack_bottom(kaapi_stack_t* stack) 
  {
    if (stack ==0) return 0;
    if (stack->sp <= stack->pc) return 0;
    return stack->pc;
  }
#else
  #define kaapi_stack_bottom(stack) \
    (stack)->pc
#endif

  /** \ingroup TASK
      The function kaapi_stack_top() will return the top task.
      The top task is not part of the stack.
      If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
      Otherwise, an 0 is returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval a pointer to the next task to push or 0.
  */
#if defined(KAAPI_DEBUG)
  static inline kaapi_task_t* kaapi_stack_top(kaapi_stack_t* stack) 
  {
    if (stack ==0) return 0;
    if (stack->sp == stack->end_sp) return 0;
    return stack->sp;
  }
#else
  #define kaapi_stack_top(stack) \
    (stack)->sp
#endif

  /** \ingroup TASK
      The function kaapi_stack_push() pushes the top task into the stack.
      If successful, the kaapi_stack_push() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval EINVAL invalid argument: bad stack pointer.
  */
#if defined(KAAPI_DEBUG)
  static inline int kaapi_stack_push(kaapi_stack_t* stack)
  {
    if (stack ==0) return EINVAL;
    if (stack->sp == stack->end_sp) return EINVAL;
    ++stack->sp;
    return 0;
  }
#else
  #define kaapi_stack_push(stack) \
    ++(stack)->sp
  #define kaapi_stack_push2(stack, count) \
    (stack)->sp++; (stack)->sp_data += count
#endif

  /** \ingroup TASK
      The function kaapi_stack_pop() 
  */
#if defined(KAAPI_DEBUG)
  static inline int kaapi_stack_pop(kaapi_stack_t* stack)
  {
    if (stack ==0) return EINVAL;
    if (stack->sp == stack->pc) return EINVAL;
    --stack->sp;
    return 0;
  }
#else
  #define kaapi_stack_pop(stack) \
    --(stack)->sp
#endif

  /** \ingroup TASK
      The function kaapi_stack_pushdata() will return the pointer to the next top data.
      The top data is not yet into the stack.
      If successful, the kaapi_stack_pushdata() function will return a pointer to the next data to push.
      Otherwise, an 0 is returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval a pointer to the next task to push or 0.
  */
#if defined(KAAPI_DEBUG)
  static inline void* kaapi_stack_pushdata(kaapi_stack_t* stack, kaapi_uint32_t count)
  {
    if (stack ==0) return 0;
    if (stack->sp_data+count >= stack->end_sp_data) return 0;
    void* retval = stack->sp_data;
    stack->sp_data += count;
    return retval;
  }
#else
  static inline void* kaapi_stack_pushdata(kaapi_stack_t* stack, kaapi_uint32_t count)
  {
    void* retval = stack->sp_data;
    stack->sp_data += count;
    return retval;
  }
#define kaapi_stack_topdata2(stack) \
  (stack)->sp_data

#define kaapi_stack_pushdata2(stack, count) \
    (stack)->sp_data += (count)
#endif

  /** \ingroup TASK
      The function kaapi_stack_save_frame() saves the current frame of a stack into
      the frame data structure.
      If successful, the kaapi_stack_save_frame() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack IN a pointer to the kaapi_stack_t data structure.
      \param frame OUT a pointer to the kaapi_frame_t data structure.
      \retval EINVAL invalid argument: bad pointer.
  */
  static inline int kaapi_stack_save_frame( kaapi_stack_t* stack, kaapi_frame_t* frame)
  {
#if defined(KAAPI_DEBUG)
    if ((stack ==0) || (frame ==0)) return EINVAL;
#endif
    frame->pc      = stack->pc;
    frame->sp      = stack->sp;
    frame->sp_data = stack->sp_data;
    return 0;  
  }


  /** \ingroup TASK
      The function kaapi_stack_restore_frame() restores the frame context of a stack into
      the stack data structure.
      If successful, the kaapi_stack_restore_frame() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack OUT a pointer to the kaapi_stack_t data structure.
      \param frame IN a pointer to the kaapi_frame_t data structure.
      \retval EINVAL invalid argument: bad pointer.
  */
  static inline int kaapi_stack_restore_frame( kaapi_stack_t* stack, kaapi_frame_t* frame)
  {
#if defined(KAAPI_DEBUG)
    if ((stack ==0) || (frame ==0)) return EINVAL;
#endif
    stack->pc      = frame->pc;
    stack->sp      = frame->sp;
    stack->sp_data = frame->sp_data;
    return 0;  
  }

  /** \ingroup TASK
      The function kaapi_stack_execone() execute the given task only.
      If successful, the kaapi_stack_execone() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval EINVAL invalid argument: bad stack pointer
      \retval EWOULDBLOCK the execution of the task will block the control flow.
      \retval EINTR the control flow has received a KAAPI interrupt.
  */
  extern inline int kaapi_stack_execone(kaapi_stack_t* stack, kaapi_task_t* task)
  {
#if defined(KAAPI_DEBUG)
    if (stack ==0) return EINVAL;
    if (task ==0) return EINVAL;
#endif
    if (task->body[0] == &kaapi_suspend_body) 
      return EWOULDBLOCK;
    else if (task->body[0] !=0) 
      (*task->body)(task, stack);
    task->body[0] = 0;    
  }

  /** \ingroup TASK
      The function kaapi_stack_execchild() execute the given and all childs task.
      If successful, the kaapi_stack_execchild() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval EINVAL invalid argument: bad stack pointer
      \retval EWOULDBLOCK the execution of the task will block the control flow.
      \retval EINTR the control flow has received a KAAPI interrupt.
  */
  extern int kaapi_stack_execchild(kaapi_stack_t* stack, kaapi_task_t* task);

  /** \ingroup TASK
      The function kaapi_stack_execall() execute all the tasks in the RFO order.
      If successful, the kaapi_stack_execall() function will return zero.
      Otherwise, an error number will be returned to indicate the error.
      \param stack INOUT a pointer to the kaapi_stack_t data structure.
      \retval EINVAL invalid argument: bad stack pointer.
      \retval EWOULDBLOCK the execution of the stack will block the control flow.
      \retval ENOEXEC no task to execute.
      \retval EINTR the control flow has received a KAAPI interrupt.
  */
  extern int kaapi_stack_execall(kaapi_stack_t* stack);

  /* ========================================================================= */

extern inline int kaapi_finalize_steal( kaapi_stack_t* stack, kaapi_task_t* task )
{
  kaapi_stack_pop(stack);
  return ((task)->body[0] == &kaapi_suspend_body ? 1 : 0);
}
  
  /* ========================================================================= */

#if defined(__cplusplus)
}
#endif


#endif
