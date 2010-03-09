/*
** kaapi_task_standard.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
#include <stdio.h>

/**
*/
void kaapi_nop_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
}

/** Dumy task pushed at startup into the main thread
*/
void kaapi_taskstartup_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
}

/**
*/
void kaapi_retn_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
#if 0
  kaapi_frame_t* frame = (kaapi_frame_t*)taskarg;
  kaapi_task_t* taskexec = frame->pc;

#if defined(KAAPI_CONCURRENT_WS)
  /* mark original task as executed, block until no more thief */
//  while (!kaapi_task_casstate(taskexec, kaapi_exec_body, kaapi_nop_body));
  while (!KAAPI_ATOMIC_CAS(&stack->lock, 0, 1));
#else
  /* in non concurrent version ...: stacks are not theft, the owner of a stack
     should execute it XOR it execute the steal op. 
  */
  kaapi_task_setbody(taskexec, kaapi_nop_body);
#endif

  kaapi_stack_restore_frame( stack, frame );
#if defined(KAAPI_CONCURRENT_WS)
  KAAPI_ATOMIC_WRITE(&stack->lock, 0);
#endif
  stack->errcode = -EAGAIN;
#endif
}

/*
*/
void kaapi_suspend_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
  stack->errcode |= EWOULDBLOCK << 8;
  printf( "[suspend] task: @=%p, stack: @=%p\n", task, stack);
}

/*
*/
void kaapi_exec_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* do not allow rexecuting already executed task */
  kaapi_assert_debug( 0 );
}

/*
*/
void kaapi_adapt_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
}


/*
*/
void kaapi_taskmain_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  kaapi_taskmain_arg_t* arg = kaapi_task_getargst(task, kaapi_taskmain_arg_t);
  arg->mainentry( arg->argc, arg->argv );
}
