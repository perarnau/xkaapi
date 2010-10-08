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
void kaapi_taskwrite_body( void* taskarg, kaapi_thread_t* thread  )
{
  int i;
  int countparam;

  const kaapi_format_t* fmt;
  void*                 orig_task_args;
  void*                 data_param;
  kaapi_access_t*       access_param;

  kaapi_access_mode_t   mode_param;
  const kaapi_format_t* fmt_param;
  unsigned int          war_param;     /* */

  void*                 copy_task_args;
  void*                 copy_data_param;
  kaapi_access_t*       copy_access_param;

  kaapi_tasksteal_arg_t* arg = (kaapi_tasksteal_arg_t*)taskarg;
  orig_task_args             = kaapi_task_getargs(arg->origin_task);
  copy_task_args             = arg->copy_task_args;
  war_param                  = arg->war_param;
  
  if (copy_task_args !=0)
  {
    /* for each parameter of the copy of the theft' task on mode:
       - V: we destroy the data
       - R,RW: do nothing
       - W,CW: set in field ->version of the original task args the field ->data of the copy args.
    */
    fmt         = arg->origin_fmt;
    countparam  = fmt->count_params;
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

  /* if signaled thread was suspended, move it to the local queue */
  kaapi_wsqueuectxt_cell_t* wcs = arg->origin_thread->wcs;
  if (wcs != 0) /* means thread has been suspended */
  { 
    kaapi_readmem_barrier();
    //kaapi_processor_t* kproc = arg->origin_thread->proc;
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    kaapi_assert_debug( kproc != 0);
    if (0)//  /*kaapi_sched_readyempty(kproc) &&*/ kaapi_thread_hasaffinity(wcs->affinity, kproc->kid))
    {
      kaapi_thread_context_t* kthread = kaapi_wsqueuectxt_steal_cell( wcs );
      if (kthread !=0)
      {
        /* Ok, here we have theft the thread and no body else can steal it
           Signal the end of execution of forked task: 
           -if no war => mark the task as terminated 
           -if war and due to copy => mark the task as aftersteal in order to merge value
        */
        if (war_param ==0)
          kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_TERM );
        else
          kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_AFTER );

        kaapi_sched_lock(&kproc->lock);
//        printf("Write signal wakeup ready task:%p, body:%p\n", (void*)arg->origin_task, (void*)arg->origin_task->body);
//        fflush(stdout);
        kaapi_sched_pushready(kproc, kthread );
        kaapi_sched_unlock(&kproc->lock);
      }
    }
    else {
      /* Ok, here we cannot theft the thread: only update state of the task */
      if (war_param ==0)
        kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_TERM );
      else
        kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_AFTER );
//      KAAPI_ATOMIC_WRITE(&wcs->state, KAAPI_WSQUEUECELL_READY);
    }
  }

  /* signal the task : mark it as executed, the old returned body should have steal flag */
  kaapi_assert_debug( kaapi_task_body_issteal( arg->origin_task->body ) );
  if (war_param ==0)
    kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_TERM );
  else
    kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_AFTER );
}


/**
*/
void kaapi_tasksteal_body( void* taskarg, kaapi_thread_t* thread  )
{
  int i;
  int                    countparam;
  int                    push_write;
  int                    w_param;
  kaapi_task_t*          task;
  kaapi_tasksteal_arg_t* arg;
  kaapi_task_body_t      body;          /* format of the stolen task */
  kaapi_format_t*        fmt;
  unsigned int           war_param;     /* */

  void*                  orig_task_args;
  void*                  data_param;
  kaapi_access_t*        access_param;

  kaapi_access_mode_t    mode_param;
  kaapi_format_t*        fmt_param;

  void*                  copy_task_args;
  void*                  copy_data_param;
  kaapi_access_t*        copy_access_param;

  
  /* get information of the task to execute */
  arg = (kaapi_tasksteal_arg_t*)taskarg;
  kaapi_assert_debug( kaapi_task_body_issteal( arg->origin_task->body ) );

  /* format of the original stolen task */  
  body            = kaapi_task_body2fnc(arg->origin_task->body);
  kaapi_assert_debug( kaapi_isvalid_body( body ) );

  fmt             = kaapi_format_resolvebybody( body );
  kaapi_assert_debug( fmt !=0 );
  
  /* the the original task arguments */
  orig_task_args  = kaapi_task_getargs(arg->origin_task);
  countparam      = fmt->count_params;
  arg->copy_task_args = 0;
  
  /**/
  war_param = arg->war_param;
  
  if (war_param)
  {
    printf( "[tasksteal] exec task steal with WAR dependencies: @=%p, thread: @=%p\n", (void*)arg->origin_task, (void*)thread);
    fflush(stdout);
  }

#if 0
  /* If it exist a W or CW access then recreate a new structure 
     of input arguments to execute the stolen task.
     Args passed by value are copied again into the stack.
  */
  push_write  = 0;
  w_param     = 0;
  for (i=0; i<countparam; ++i)
  {
    if (KAAPI_ACCESS_IS_ONLYWRITE(KAAPI_ACCESS_GET_MODE(fmt->mode_params[i])))
    {
      w_param = 1;
      if ((war_param & (1<<i)) !=0) 
        push_write=1;
      break;
    }
  }
#endif

  if (!war_param)
  {
    /* Execute the orinal body function with the original args */
    body(orig_task_args, thread);

    /* push task that will be executed after all created task by the user task */
    task = kaapi_thread_toptask( thread );
    kaapi_task_init( task, kaapi_taskwrite_body, arg );
    kaapi_thread_pushtask( thread );

#if 0
    /* if no barrier here: only read data signal the task */
    if (w_param != 0)
    {
      for (i=0; i<countparam; ++i)
      {
        mode_param      = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
        if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param))
        {
          data_param      = (void*)(fmt->off_params[i] + (char*)orig_task_args);
          fmt_param       = fmt->fmt_params[i];
          access_param    = (kaapi_access_t*)(data_param);
          access_param->version = access_param->data;
        }
      }
      kaapi_writemem_barrier();
    }
#endif
  }
  else /* it exists at least one w parameter with war dependency */
  {
    printf("Execute task after recopy some args\n");
    fflush(stdout);
    copy_task_args       = kaapi_thread_pushdata( thread, fmt->size);
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

      /* initialize all parameters ... */
      access_param               = (kaapi_access_t*)(data_param);
      copy_access_param          = (kaapi_access_t*)(copy_data_param);
      copy_access_param->data    = access_param->data;
      copy_access_param->version = 0; /*access_param->version; / * not required... * / */
      
      if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param) )
      {
        if ((war_param & (1<<i)) !=0)
        { 
          /* allocate new data */
#if defined(KAAPI_DEBUG)
          copy_access_param->data    = calloc(1,fmt_param->size);
#else
          copy_access_param->data    = malloc(fmt_param->size);
#endif
          if (fmt_param->cstor !=0) (*fmt_param->cstor)(copy_access_param->data);
        }
      }
    }

    /* Execute the orinal body function with the copy of args: do not push it... ?
       WHY to push a nop ?
    */
    task = kaapi_thread_toptask( thread );
    kaapi_task_init( task, kaapi_nop_body, copy_task_args);
    kaapi_thread_pushtask( thread );

    /* call directly the stolen body function */
    body( copy_task_args, thread);

    /* push task that will be executed after all created task by the user task */
    task = kaapi_thread_toptask( thread );
    kaapi_task_init( task, kaapi_taskwrite_body, arg );
    kaapi_thread_pushtask( thread );
  }
}
