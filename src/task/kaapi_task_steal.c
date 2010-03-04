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
void kaapi_taskwrite_body( void* taskarg, kaapi_stack_t* stack )
{
  int i;
  int countparam;
  void* orig_task_args;
  kaapi_access_mode_t mode_param;
  void*               data_param;
  kaapi_access_t*     access_param;
  void*               tmp;

  kaapi_tasksteal_arg_t* arg = (kaapi_tasksteal_arg_t*)taskarg;
  orig_task_args   = kaapi_task_getargs(arg->origin_task);

  countparam = arg->origin_fmt->count_params;
  
  /* for each parameter we only switch data & version field of the parameters
     of the original task
     - for all W mode parameter (should be also true for CW mode), the field ->version
     contains the newly produced data
  */
  for (i=0; i<countparam; ++i)
  {
    mode_param = KAAPI_ACCESS_GET_MODE(arg->origin_fmt->mode_params[i]);

    if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param))
    {
      data_param = (void*)(arg->origin_fmt->off_params[i] + (char*)orig_task_args);
      access_param = (kaapi_access_t*)(data_param);

      /* swap */
      tmp = access_param->data;
      access_param->data = access_param->version;
      access_param->version = tmp;
    }
  }
}


/**
*/
void kaapi_tasksteal_body( void* taskarg, kaapi_stack_t* stack )
{
  int i;
  int                    countparam;
  kaapi_tasksteal_arg_t* arg;
  kaapi_task_body_t      body;          /* format of the stolen task */

  void*               orig_task_args;
  kaapi_access_mode_t mode_param;
  void*               data_param;
  kaapi_format_t*     fmt_param;
  kaapi_access_t*     access_param;

  
  /* get information of the task to execute */
  arg = (kaapi_tasksteal_arg_t*)taskarg;

  /* format of the original stolen task */  
  body = kaapi_task_getextrabody(arg->origin_task);
  arg->origin_fmt = kaapi_format_resolvebybody( body );
  kaapi_assert_debug( arg->origin_fmt !=0 );
  
  /* the the original task arguments */
  orig_task_args = kaapi_task_getargs(arg->origin_task);
  
  /* during the stealing step the right arguments of the stolen task are put into ->version for the pointer type.
     - the original task body is kaapi_suspend_body (suspension if run on the victim side) 
     - we swap arg->data and arg->version of pointer args in order to avoid copy of arguments and we start the task
  */
  countparam = arg->origin_fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    mode_param = KAAPI_ACCESS_GET_MODE(arg->origin_fmt->mode_params[i]);
    
    if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param))
    {
      data_param = (void*)(arg->origin_fmt->off_params[i] + (char*)orig_task_args);
      access_param = (kaapi_access_t*)(data_param);
      fmt_param = arg->origin_fmt->fmt_params[i];

      /* store old value in version and allocate new data */
      access_param->version = access_param->data;
      access_param->data    = malloc(fmt_param->size);
    }
  }

  /* Execute the orinal body function
     - replace tasksteal_body by nopbody to allows thief to steal down the stack
     - normally -> no splitter, no possibility to call splitter...
  */
  body(orig_task_args, stack);
}
