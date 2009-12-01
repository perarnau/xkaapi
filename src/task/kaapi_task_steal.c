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
  void* orig_task_args;
  void* copy_task_args;

  kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( task, kaapi_tasksteal_arg_t );

  /* report data to the original task */
  fmt = arg->origin_fmt;
  orig_task_args   = arg->origin_task->sp;
  copy_task_args = arg->copy_arg;

if (fmt->fmtid == 96)
{
  kaapi_stack_print( 0, stack );
  abort();
}

  countparam = fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    void* orig_param = (void*)(fmt->off_params[i] + (char*)orig_task_args);
    void* copy_param = (void*)(fmt->off_params[i] + (char*)copy_task_args);

    if (KAAPI_ACCESS_IS_ONLYWRITE(m))
    {
      kaapi_access_t* orig_access = (kaapi_access_t*)(orig_param);
      kaapi_access_t* copy_access = (kaapi_access_t*)(copy_param);
      orig_access->version        = copy_access->data;
    }
    /* read write has shared the origin access (and data) */
  }
}


/**
*/
void kaapi_tasksteal_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  int i;
  int countparam;
  kaapi_format_t* fmt;           /* format of the task */
  void* orig_task_args;
  void* copy_task_args;
  int push_write = 0;
  kaapi_tasksteal_arg_t* arg;

  kaapi_access_mode_t m;
  void*               orig_param;
  void*               copy_param;
  kaapi_format_t*     fmt_param;
  
  arg = kaapi_task_getargst( task, kaapi_tasksteal_arg_t );
#if 1
  printf("Recv: thiefstack:%p spdata:%p, arg:%p, task:%p, fmt:%p\n", stack, stack->sp_data, arg, arg->origin_task, arg->origin_fmt );
#endif

  KAAPI_LOG(100, "tasksteal: 0x%p -> task stolen: 0x%p\n", (void*)task, (void*)arg->origin_task );

  /* format of the original stolen task */  
  fmt = arg->origin_fmt;
  kaapi_assert_debug( fmt !=0 );
  
  /* push a copy of the task argument in the stack */
  orig_task_args = arg->origin_task->sp;
  copy_task_args = kaapi_stack_pushdata(stack, fmt->size);
  arg->copy_arg  = copy_task_args;
#if 0
  printf("After allocate: thiefstack:%p spdata:%p, arg:%p, task:%p, fmt:%p\n", stack, stack->sp_data, arg, arg->origin_task, arg->origin_fmt );
#endif
  
  /* recopy or allocate in the heap the shared objects in the arguments of the stolen task */
  countparam = fmt->count_params;
  push_write = 0;
  for (i=0; i<countparam; ++i)
  {
    m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    orig_param = (void*)(fmt->off_params[i] + (char*)orig_task_args);
    copy_param = (void*)(fmt->off_params[i] + (char*)copy_task_args);
    fmt_param = fmt->fmt_params[i];

    if (KAAPI_ACCESS_IS_WRITE(m)) push_write = 1;

    if (m == KAAPI_ACCESS_MODE_V) 
    { /* recopy pass by value parameter */
      (*fmt_param->cstorcopy)(copy_param, orig_param);
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
         assign the pointer to the original shared data contained in version field
      */    
      kaapi_access_t* orig_access = (kaapi_access_t*)(orig_param);
      kaapi_access_t* copy_access = (kaapi_access_t*)(copy_param);
      copy_access->data           = orig_access->version;   
      copy_access->version        = 0;
    }
  }

  /* mute myself... 
     switch should be atomic with iteration over the stack.... for concurrent impl.
     - normally -> no splitter, no possibility to call splitter...
  */
  /* \TODO: use the current architecture or move this file in machine repository */
  kaapi_task_setbody  ( task, fmt->entrypoint[KAAPI_PROC_TYPE_CPU] );
  kaapi_task_setargs  ( task, copy_task_args );
  /* update flag with original flag */
  kaapi_task_setflags ( task, arg->origin_task->flag );
  
  /* ... and execute the  mutation */
  (*task->body)( task, stack );

#if 0
  /* ... and push continuation if w, cw or rw mode */
  if (push_write)
  {
    task = kaapi_stack_toptask( stack );
    kaapi_task_init(stack, task, KAAPI_TASK_STICKY );
    kaapi_task_setargs( task, arg ); /* can keep pointer to kaapi_tasksteal_body arguments */
    kaapi_task_setbody( task, &kaapi_taskwrite_body );
    kaapi_stack_pushtask( stack );
  }
#endif

//  printf("IN %s: end exec/// task copy:@0x%p -> task stolen:@0x%p\n", __PRETTY_FUNCTION__, task, arg->origin_task );
  
  KAAPI_LOG(100, "tasksteal: 0x%p end exec, next task: 0x%p bodysignal: 0x%p, pc: 0x%p\n", 
      (void*)task, 
      (void*)(task+1), 
      (void*)(task+1)->body, 
      (void*)stack->pc );
}


