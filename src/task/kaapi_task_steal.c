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
void kaapi_taskwrite_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  int i;
  int countparam;
  kaapi_format_t* fmt;           /* format of the stolen task */
  void* stolen_task_sp;
  void** origin_task_args;
  void* copy_arg;

  kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( task, kaapi_tasksteal_arg_t );

  /* report data to the original task */
  fmt = arg->origin_fmt;

  stolen_task_sp = arg->origin_task->sp;
  origin_task_args = arg->origin_task_args;
  copy_arg = arg->copy_arg;

  countparam = fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    void* original_param = (void*)(fmt->off_params[i] + (char*)stolen_task_sp);
    void* copy_param = (void*)(fmt->off_params[i] + (char*)copy_arg);
    kaapi_format_t* fmt_param = fmt->fmt_params[i];

    if (m == KAAPI_ACCESS_MODE_V) 
    { /* copy pass by value parameter */
      kaapi_assert_debug( original_param == origin_task_args[i] );
      (*fmt_param->dstor)(copy_param);
    } 
    else if (KAAPI_ACCESS_IS_ONLYWRITE(m))
    {
      kaapi_access_t* original_access = (kaapi_access_t*)(original_param);
      kaapi_access_t* copy_access     = (kaapi_access_t*)(copy_param);
      original_access->version        = copy_access->data;
    }
    else if (KAAPI_ACCESS_IS_READ(m))
    { /* nothing to do ?
      */
      /* FOR DEBUG: ALSO COPY INTIAL VALUE INTO VERSION PARAMETER */
      kaapi_access_t* original_access = (kaapi_access_t*)(original_param);
      original_access->version        = origin_task_args[i];
    }
  }
}


/**
*/
void kaapi_tasksteal_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  int i;
  int countparam;
  kaapi_format_t* fmt;           /* format of the task */
  void* stolen_task_sp;
  void** origin_task_args;
  void* copy_arg;
  int push_write = 0;
  
  kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( task, kaapi_tasksteal_arg_t );

  KAAPI_LOG(100, "tasksteal: 0x%p -> task stolen: 0x%p\n", (void*)task, (void*)arg->origin_task );

  /* format of the original stolen task */  
  fmt = arg->origin_fmt;
  kaapi_assert_debug( fmt !=0 );
  
  /* push a copy of the task argument in the stack */
  stolen_task_sp = arg->origin_task->sp;
  origin_task_args = arg->origin_task_args;
  copy_arg = arg->copy_arg;
  
  /* recopy or allocate in the heap the shared objects in the arguments of the stolen task */
  countparam = fmt->count_params;
  push_write = 0;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    void* original_param = (void*)(fmt->off_params[i] + (char*)stolen_task_sp);
    void* copy_param = (void*)(fmt->off_params[i] + (char*)copy_arg);
    kaapi_format_t* fmt_param = fmt->fmt_params[i];

    if (KAAPI_ACCESS_IS_WRITE(m)) push_write = 1;

    if (m == KAAPI_ACCESS_MODE_V) 
    { /* copy pass by value parameter */
      kaapi_assert_debug( original_param == origin_task_args[i] );
      (*fmt_param->cstorcopy)(copy_param, origin_task_args[i]);
    } 
    else if (KAAPI_ACCESS_IS_ONLYWRITE(m))
    {
      /* copy_param points to the pointer on the shared data.
         It is a W shared, allocate a new one in the heap 
      */
      kaapi_gd_t* shared_object   = (kaapi_gd_t*)malloc(sizeof(kaapi_gd_t)+fmt_param->size);
      kaapi_access_t* copy_access = (kaapi_access_t*)(copy_param);
      void* data_pointer          = (void*)(shared_object + 1);
      shared_object->last_mode    = KAAPI_ACCESS_MODE_VOID;
      shared_object->last_version = 0;
      copy_access->data           = data_pointer;
      copy_access->version        = 0;
    }
    else
    { /* copy_param points to the pointer on the shared data.
         assign the pointer to the original shared data
      */    
      kaapi_access_t* copy_access = (kaapi_access_t*)(copy_param);
      copy_access->data           = origin_task_args[i];
      copy_access->version        = 0;
    }
  }

  /* mute myself... 
     switch should be atomic with iteration over the stack.... for concurrent impl.
     - normally -> no splitter, no possibility to call splitter...
  */
  kaapi_task_setargs  ( task, copy_arg );
  task->format = (kaapi_task_body_t)arg->origin_body; /* as if the current task was a fibo task */
  /* write barrier here */
  task->splitter = &kaapi_task_splitter_dfg;
  kaapi_task_setflags ( task, arg->origin_task->flag );

  /* ... and execute the  mutation */
  (*arg->origin_body)( task, stack );

  /* ... and push continuation if w, cw or rw mode */
  if (push_write)
  {
    task = kaapi_stack_toptask( stack );
    kaapi_task_init(stack, task, KAAPI_TASK_STICKY );
    kaapi_task_setargs( task, arg ); /* can keep pointer to kaapi_tasksteal_body arguments */
    kaapi_task_setbody( task, &kaapi_taskwrite_body );
    kaapi_stack_pushtask( stack );
  }

//  printf("IN %s: end exec/// task copy:@0x%p -> task stolen:@0x%p\n", __PRETTY_FUNCTION__, task, arg->origin_task );
  
  KAAPI_LOG(100, "tasksteal: 0x%p end exec, next task: 0x%p bodysignal: 0x%p, pc: 0x%p\n", 
      (void*)task, 
      (void*)(task+1), 
      (void*)(task+1)->body, 
      (void*)stack->pc );
}


