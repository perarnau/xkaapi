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
#if defined(KAAPI_VERY_COMPACT_TASK)
void _kaapi_nop_body( kaapi_task_t* task, kaapi_stack_t* stack)
#else
void kaapi_nop_body( kaapi_task_t* task, kaapi_stack_t* stack)
#endif
{
}

/** Dumy task pushed at startup into the main thread
*/
#if defined(KAAPI_VERY_COMPACT_TASK)
void _kaapi_taskstartup_body( kaapi_task_t* task, kaapi_stack_t* stack)
#else
void kaapi_taskstartup_body( kaapi_task_t* task, kaapi_stack_t* stack)
#endif
{
}

/**
*/
#if defined(KAAPI_VERY_COMPACT_TASK)
void _kaapi_retn_body( kaapi_task_t* task, kaapi_stack_t* stack)
#else
void kaapi_retn_body( kaapi_task_t* task, kaapi_stack_t* stack)
#endif
{
  kaapi_frame_t* frame = kaapi_task_getargst( task, kaapi_frame_t);
  kaapi_task_setstate( frame->pc, KAAPI_TASK_S_TERM );
#if defined(KAAPI_CONCURRENT_WS)
  /* finer lock: lock for the frame only : put a lock on the task that create the frame */
  while (!KAAPI_ATOMIC_CAS(&stack->lock, 0, 1));
#endif
  kaapi_stack_restore_frame( stack, frame );
#if defined(KAAPI_CONCURRENT_WS)
  KAAPI_ATOMIC_WRITE(&stack->lock, 0);
#endif
}

/*
*/
#if defined(KAAPI_VERY_COMPACT_TASK)
void _kaapi_suspend_body( kaapi_task_t* task, kaapi_stack_t* stack)
#else
void kaapi_suspend_body( kaapi_task_t* task, kaapi_stack_t* stack)
#endif
{
}


