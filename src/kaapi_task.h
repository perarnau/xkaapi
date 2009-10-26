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
#include "kaapi_config.h"
#include "kaapi_type.h"
#include "kaapi_error.h"


/* TODO: 
  - implen / spec : DFG + flot de donnée
    * R, W, RW seem to be ok
    * CW ?
    * P ?
    
  - spec couche reseau:
      * com: AM + RDMA
      * proxy object pour fwd.
      * gestion des erreurs
*/


/* --------------------------------------------------------------- */

#if defined(__cplusplus)
extern "C" {
#endif

struct kaapi_task_t;
struct kaapi_stack_t;

/** \defgroup TASK Task
    This group defines functions to create and initialize and use task.
*/
/** \defgroup STACK Stack
    This group defines functions to create and initialize stack.
*/


/** Body of the nop task 
    \ingroup TASK
*/
extern void kaapi_nop_body( kaapi_task_t*, kaapi_stack_t*);

/** Body of the task that restore the frame pointer 
    \ingroup TASK
*/
extern void kaapi_retn_body( kaapi_task_t*, kaapi_stack_t*);


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
extern int kaapi_stack_alloc( kaapi_stack_t* stack, size_t count_task, size_t size_data );

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
                             size_t size_task_buffer, void* task_buffer,
                             size_t size_data_buffer, void* data_buffer 
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
    The function kaapi_stack_top() will return the top task.
    The top task is not part of the stack.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
#if defined(KAAPI_DEBUG)
static inline kaapi_stack_t* kaapi_stack_top(kaapi_stack_t* stack) 
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
  (++(stack)->sp, 0)
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
static inline void* kaapi_stack_pushdata(kaapi_stack_t* stack, size_t count)
{
  if (stack ==0) return 0;
  if (stack->sp_data+count >= stack->end_sp_data) return 0;
  void* retval = stack->sp_data;
  stack->sp_data += count;
  return retval;
}
#else
static inline void* kaapi_stack_pushdata(kaapi_stack_t* stack, size_t count)
{
  void* retval = stack->sp_data;
  stack->sp_data += count;
  return retval;
}
#endif


/** \ingroup TASK
    The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
    If the task pointer is an invalid pointer, then the function will return 0 as if the task may not be stolen.
    \param task IN a pointer to the kaapi_task_t to test.
*/
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
    The function kaapi_stack_taskexec() executes the next task.
    On return and after execution of the task, the PC is updated to the next task to execute.
    If successful, the kaapi_stack_taskexec() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval ENOEXEC no task to execute.
    \retval EINTR the control flow has received a KAAPI interrupt.
*/
#if defined(KAAPI_DEBUG)
extern int kaapi_stack_taskexec(kaapi_stack_t* stack);
#else
static inline int kaapi_stack_taskexec(kaapi_stack_t* stack) 
{
  kaapi_task_t* saved_sp;
  char*         saved_sp_data;
  kaapi_task_t* task = stack->pc;
  if (task->body ==0) return ENOEXEC;
  if (task->body == &kaapi_retn_body) 
  {
    kaapi_retn_body(task, stack);
    return 0;
  }
  saved_sp      = stack->sp;
  saved_sp_data = stack->sp_data;
  (*task->body)(task, stack);
  task->body = 0;

  /* push restore_frame task if pushed tasks */
  if (saved_sp < stack->sp)
  {
    kaapi_task_t* retn = kaapi_stack_top(stack);
    retn->body  = &kaapi_retn_body;
    /* next line is equiv to saving a frame. retn->pdata should be viewed as a kaapi_frame_t */
    retn->param.pdata[0] = task; /* <=> saved_pc */
    retn->param.pdata[1] = saved_sp;
    retn->param.pdata[2] = saved_sp_data;
    kaapi_stack_push(stack);

    /* update pc to the first forked task */
    stack->pc = saved_sp;
    return 0;
  }
  ++stack->pc;
  return 0;
}
#endif

/** \ingroup TASK
    The function kaapi_stack_taskexecall() execute all the tasks in the RFO order.
    If successful, the kaapi_stack_taskexecall() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval ENOEXEC no task to execute.
    \retval EINTR the control flow has received a KAAPI interrupt.
*/
extern int kaapi_stack_taskexecall(kaapi_stack_t* stack);


/** \ingroup TASK
    The function kaapi_task_condeval() executes the condition function pointed by body for
    a task in the state KAAPI_TASK_WAITING. And, if the condition
    becomes true, then it restaures body to the task body function and pdata[0].
    If condition was successfully evaluated, the kaapi_task_condeval() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    \param task INOUT a pointer to the kaapi_task_t.
    \retval EAGAIN the condition is not satisfied and should be re-evaluated.
    \retval EINVAL invalid argument: bad task pointer or task is not in state KAAPI_TASK_WAITING.
*/
extern int kaapi_task_condeval(kaapi_stack_t* stack, kaapi_task_t* task);


#if defined(__cplusplus)
}
#endif


#endif
