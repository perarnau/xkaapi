/*
** kaapi_task_steal.c
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
void kaapi_tasksteal_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  int i;
  int countparam;
  kaapi_format_t* fmt;           /* format of the task */
  kaapi_task_t* copy_task;   
  kaapi_tasksteal_arg_t* arg = kaapi_task_argst( task, kaapi_tasksteal_arg_t );

  printf("IN %s\n", __PRETTY_FUNCTION__ );

  /* format of the original task */  
  fmt = kaapi_format_resolvebybody( arg->origin_body );
  kaapi_assert_debug( fmt !=0 );
  
  /* push a copy of the task in the stack */
  copy_task = kaapi_stack_toptask( stack );

  kaapi_task_init(stack, copy_task, arg->origin_task->flag );
  kaapi_task_setbody( copy_task, arg->origin_body );
  kaapi_task_setargs( copy_task, kaapi_stack_pushdata(stack, fmt->size ));
  kaapi_tasksteal_arg_t* copy_arg = kaapi_task_args( copy_task );

  countparam = fmt->count_params;
  
  /* recopy or allocate in the heap the shared objects in the arguments of the stolen task */
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    void* original_param = (void*)(fmt->off_params[i] + (char*)arg->origin_task->sp);
    void* copy_param = (void*)(fmt->off_params[i] + (char*)copy_arg);
    kaapi_format_t* fmt_param = fmt->fmt_params[i];

    if (m == KAAPI_ACCESS_MODE_V) 
    { /* copy pass by value parameter */
      kaapi_assert_debug( original_param == arg->origin_task_args[i] );
      (*fmt_param->cstorcopy)(copy_param, arg->origin_task_args[i]);
    } 
    else if (KAAPI_ACCESS_IS_ONLYWRITE(m))
    {
      /* copy_param points to the pointer on the shared data.
         It is a W shared, allocate a new one in the heap 
      */
      kaapi_access_t* shared_object = (kaapi_access_t*)malloc(sizeof(kaapi_access_t)+fmt_param->size);
      void* data_pointer = (void*)(shared_object + 1);
      shared_object->last_mode    = KAAPI_ACCESS_MODE_VOID;
      shared_object->last_version = 0;
      *(void**)copy_param = data_pointer;
    }
    else
    { /* copy_param points to the pointer on the shared data.
         assign the pointer to the original data
      */    
      *(void**)copy_param = arg->origin_task_args[i];
    }
  }

  /* execute the task: on return because it was pushed... */
  kaapi_stack_pushtask(stack);
}


