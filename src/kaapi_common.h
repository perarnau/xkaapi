/*
** kaapi_common.h
** xkaapi
** 
** Created on Tue Mar 31 15:16:12 2009
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
#ifndef _KAAPI_COMMON_TYPE_H
#define _KAAPI_COMMON_TYPE_H 1

#include <stdint.h> /* constant size type definitions */

/* ========================================================================= */
/* Kaapi name for stdint typedefs.
 */
typedef uint8_t  kaapi_uint8_t;
typedef uint16_t kaapi_uint16_t;
typedef uint32_t kaapi_uint32_t;
typedef int8_t   kaapi_int8_t;
typedef int16_t  kaapi_int16_t;
typedef int32_t  kaapi_int32_t;

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
    - KAAPI_TASK_F_LOCALITY: if set, the task as locality constraint defined in locality data field.
    - KAAPI_TASK_F_ADAPTIVE: if set, the task is an adaptative task that could be stolen or preempted.
*/
/*@{*/
#define KAAPI_TASK_F_STICKY     (0x1 <<8)
#define KAAPI_TASK_F_LOCALITY   (0x2 <<8)
#define KAAPI_TASK_F_ADAPTIVE   (0x4 <<8)
/*@}*/


/** Constants for number for fixed size parameter of a task
    \ingroup TASK
*/
/*@{*/
#define KAAPI_TASK_MAX_DATA  24 /* allows 3 double */
#define KAAPI_TASK_MAX_IDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint32_t))
#define KAAPI_TASK_MAX_SDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint16_t))
#define KAAPI_TASK_MAX_DDATA (KAAPI_TASK_MAX_DATA/sizeof(double))
#define KAAPI_TASK_MAX_FDATA (KAAPI_TASK_MAX_DATA/sizeof(float))
#define KAAPI_TASK_MAX_PDATA (KAAPI_TASK_MAX_DATA/sizeof(void*))
/*@}*/

/* ========================================================================= */
struct kaapi_task_t;
struct kaapi_stack_t;
struct kaapi_task_condition_t;

/** Task body
    \ingroup TASK
*/
typedef void (*kaapi_task_body_t)(struct kaapi_task_t* /*task*/, struct kaapi_stack_t* /* thread */);

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
    struct { 
      kaapi_uint8_t  xstate;   /** State of the task see above */
      kaapi_uint8_t  flags;    /** flags of the task see above */
    } sf;
  } sf; /* state & flag */
  kaapi_task_body_t  body;     /** C function that represent the body to execute*/
  union { /* union data to view task's immediate data with type information. Be carreful: this is an (anonymous) union  */
    kaapi_uint32_t idata[ KAAPI_TASK_MAX_IDATA ];
    kaapi_uint16_t sdata[ KAAPI_TASK_MAX_SDATA ];
    double         ddata[ KAAPI_TASK_MAX_DDATA ];
    float          fdata[ KAAPI_TASK_MAX_FDATA ];
    void*          pdata[ KAAPI_TASK_MAX_PDATA ];
    struct kaapi_task_condition_t* arg_condition;
  } param;
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_task_t;


/** Kaapi stack of task definition
   \ingroup STACK
*/
typedef struct kaapi_stack_t {
  kaapi_task_t* pc;             /** task counter: next task to execute, 0 if empty stack */
  kaapi_task_t* sp;             /** stack counter: next free task entry */
#if defined(KAAPI_DEBUG)
  kaapi_task_t* end_sp;         /** past the last stack counter: next entry after the last task in stack array */
#endif
  kaapi_task_t* task;           /** stack of tasks */

  char*         sp_data;        /** stack counter for the data: next free data entry */
#if defined(KAAPI_DEBUG)
  char*         end_sp_data;    /** past the last stack counter: next entry after the last task in stack array */
#endif
  char*         data;           /** stack of data with the same scope than task */
} kaapi_stack_t;

/** Kaapi frame definition
   \ingroup STACK
*/
typedef struct kaapi_frame_t {
    kaapi_task_t* pc;
    kaapi_task_t* sp;
    char*         sp_data;
} kaapi_frame_t;


#endif /* _KAAPI_COMMON_TYPE_H */
