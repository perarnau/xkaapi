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
*/
int kaapi_stack_init( kaapi_stack_t* stack, 
                      kaapi_uint32_t size_task_buffer, void* task_buffer,
                      kaapi_uint32_t size_data_buffer, void* data_buffer 
)
{
  if (stack == 0) return EINVAL;
  stack->hasrequest =0;
  if (size_task_buffer ==0) 
  { 
    stack->pc = stack->sp = stack->task = 0; 
    stack->end_sp = 0;
  }
  else {
    if (size_task_buffer / sizeof(kaapi_task_t) ==0) return EINVAL;    
    stack->task   = (kaapi_task_t*)task_buffer;
    stack->pc     = stack->sp = stack->task;
    stack->end_sp = stack->task + size_task_buffer/sizeof(kaapi_task_t);
  }
  if (size_data_buffer ==0) 
  {
    stack->sp_data = stack->data = 0;
  }
  else 
  {
    stack->sp_data     = stack->data = data_buffer;
    stack->end_sp_data = stack->data + size_data_buffer;
  }
  return 0;
}
