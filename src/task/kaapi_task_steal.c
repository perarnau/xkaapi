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

  const kaapi_format_t* fmt;
  void*                 orig_task_args;
  void*                 data_param;
  kaapi_access_t*       access_param;

  kaapi_access_mode_t   mode_param;
  const kaapi_format_t* fmt_param;

  void*                 copy_task_args;
  void*                 copy_data_param;
  kaapi_access_t*       copy_access_param;

  kaapi_tasksteal_arg_t* arg = kaapi_task_getargst(task, kaapi_tasksteal_arg_t);
  orig_task_args             = kaapi_task_getargs(arg->origin_task);
  copy_task_args             = arg->copy_task_args;
  fmt                        = arg->origin_fmt;
  countparam = fmt->count_params;
  
  printf( "[taskwrite] task: @=%p, stack: @=%p\n", arg->origin_task, stack);
  fflush(stdout);

  if (copy_task_args !=0)
  {
    /* for each parameter of the copy of the theft' task on mode:
       - V: we destroy the data
       - R,RW: do nothing
       - W,CW: set in field ->version of the original task args the field ->data of the copy args.
    */
    for (i=0; i<countparam; ++i)
    {
      mode_param = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
      copy_data_param = (void*)(fmt->off_params[i] + (char*)copy_task_args);
      fmt_param       = fmt->fmt_params[i];
      if (mode_param == KAAPI_ACCESS_MODE_V) 
      {
        (*fmt_param->dstor)(copy_data_param);
        continue;
      }

      if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param))
      {
        data_param        = (void*)(fmt->off_params[i] + (char*)orig_task_args);
        copy_data_param   = (void*)(fmt->off_params[i] + (char*)copy_task_args);
        access_param      = (kaapi_access_t*)(data_param);
        copy_access_param = (kaapi_access_t*)(copy_data_param);

        /* write the new version */
        access_param->version = copy_access_param->data;
      }
    }
  }
  else {
    /* copy args == 0: do nothing */
  }
}


/**
*/
void kaapi_tasksteal_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  int i;
  int                    countparam;
  int                    push_write;
  kaapi_tasksteal_arg_t* arg;
  kaapi_task_body_t      body;          /* format of the stolen task */
  kaapi_format_t*        fmt;

  void*                  orig_task_args;
  void*                  data_param;
  kaapi_access_t*        access_param;

  kaapi_access_mode_t    mode_param;
  kaapi_format_t*        fmt_param;

  void*                  copy_task_args;
  void*                  copy_data_param;
  kaapi_access_t*        copy_access_param;

  
  /* get information of the task to execute */
  arg = kaapi_task_getargst(task, kaapi_tasksteal_arg_t);
  arg->copy_task_args = 0;

  /* format of the original stolen task */  
  body            = kaapi_task_getextrabody(arg->origin_task);
  fmt             = kaapi_format_resolvebybody( body );
  kaapi_assert_debug( fmt !=0 );
  
  /* the the original task arguments */
  orig_task_args  = kaapi_task_getargs(arg->origin_task);
  countparam      = fmt->count_params;
  
  printf( "[tasksteal] exec task: @=%p, body:%p, stack: @=%p\n", arg->origin_task, body, stack);
  fflush(stdout);

  /* If it exist a W or CW access then recreate a new structure 
     of input arguments to execute the stolen task.
     Args passed by value are copied again into the stack.
  */
  push_write           = 0;
  for (i=0; i<countparam; ++i)
  {
    if (KAAPI_ACCESS_IS_ONLYWRITE(KAAPI_ACCESS_GET_MODE(fmt->mode_params[i])))
    {
      push_write=1;
      break;
    }
  }

  if (!push_write)
  {
    /* Execute the orinal body function with the original args
    */
    body(arg->origin_task, stack);
  }
  else 
  {
    copy_task_args       = kaapi_stack_pushdata(stack, fmt->size);
    arg->copy_task_args  = copy_task_args;
    arg->origin_fmt      = fmt;

    for (i=0; i<countparam; ++i)
    {
      mode_param      = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
      data_param      = (void*)(fmt->off_params[i] + (char*)orig_task_args);
      copy_data_param = (void*)(fmt->off_params[i] + (char*)copy_task_args);
      fmt_param       = fmt->fmt_params[i];
      
      if (mode_param == KAAPI_ACCESS_MODE_V) 
      {
        (*fmt_param->cstorcopy)(copy_data_param, data_param);
        continue;
      }

      access_param               = (kaapi_access_t*)(data_param);
      copy_access_param          = (kaapi_access_t*)(copy_data_param);
      copy_access_param->data    = access_param->data;
      copy_access_param->version = access_param->version; /* not required... */
      
      if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param))
      {
        /* store old value in version and allocate new data */
        copy_access_param->data    = malloc(fmt_param->size);
      }
    }
    /* Execute the orinal body function with the copy of args
    */
    kaapi_task_init( kaapi_stack_toptask(stack), body, copy_task_args);
    body( kaapi_stack_toptask(stack), stack);

    /* ... and we push the write task if w, cw or rw mode */
    kaapi_task_init( kaapi_stack_toptask( stack ), kaapi_taskwrite_body, arg );
    kaapi_stack_pushtask( stack );
  }
}
