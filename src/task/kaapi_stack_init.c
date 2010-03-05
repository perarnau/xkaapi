/*
** kaapi_stack_init.c
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

/** kaapi_stack_init
    Initialize the Kaapi stack data structure.
    The stack is organized in two parts : the first, from address 0 to sp_data,
    contains the stack of data; the second contains the stack of tasks from address task down to sp.
    Pc points on the next task to execute.
    
   |   --------------  <- stack->data
   |   |            |
   |   |            |
   |   |            |
   |   |            |
  \|/  --------------  <- stack->sp_data
       |            |
       |  free zone |
       |            |
       --------------  <- stack->sp
  /|\  |            |  
   |   |            |  
   |   |            |  <- stack->pc
   |   |            |
   |   --------------  <- stack->task
  
  The stack is full when stack->sp_data == stack->sp.
*/
int kaapi_stack_init( kaapi_stack_t* stack, kaapi_uint32_t size, void* buffer )
{
  kaapi_task_t* pasttheend_task;
  
  if (stack == 0) return EINVAL;
  stack->haspreempt =0;
  stack->hasrequest =0; /* 0 means no thiefs */
  stack->errcode    =  0;
  KAAPI_ATOMIC_WRITE(&stack->lock, 0);
  
  if (size ==0) 
  { 
    stack->frame_sp = stack->sp = stack->task = 0; 
    stack->sp_data  = stack->data = 0;
    return 0;
  }

  if (size / sizeof(kaapi_task_t) ==0) return EINVAL;
  
  stack->sp_data  = stack->data = (char*)buffer;
  pasttheend_task = (kaapi_task_t*)((char*)buffer + size);
  stack->task     = pasttheend_task -1;
  stack->frame_sp = stack->sp = stack->task;
  stack->thiefsp  = stack->sp;
  
  stack->stackframe = malloc(sizeof(kaapi_frame_t)*KAAPI_MAX_RECCALL);
  if (stack->stackframe ==0) return ENOMEM;
  stack->pfsp = stack->stackframe;
  
  return 0;
}
