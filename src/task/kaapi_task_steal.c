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

#if defined(KAAPI_USE_CUDA)
# include "../machine/cuda/kaapi_cuda_execframe.h"
#endif

/* toremove */
extern void kaapi_hws_sched_dec_sync(void);
/* toremove */

/**
*/
void kaapi_taskwrite_body( 
  void* taskarg, 
  kaapi_thread_t* thread  __attribute__((unused))
)
{
  unsigned int i;
  size_t count_params;

  const kaapi_format_t* fmt;
  void*                 orig_task_args;
  kaapi_access_t        access_param;

  kaapi_access_mode_t   mode_param;
  const kaapi_format_t* fmt_param;
  unsigned int          war_param;     /* */
  unsigned int          cw_param;     /* */

  void*                 copy_task_args;
  void*                 copy_data_param;
  kaapi_access_t        copy_access_param;

  kaapi_tasksteal_arg_t* arg = (kaapi_tasksteal_arg_t*)taskarg;
  orig_task_args             = kaapi_task_getargs(arg->origin_task);
  copy_task_args             = arg->copy_task_args;
  war_param                  = arg->war_param;
  cw_param                   = arg->cw_param;
  
  if (copy_task_args !=0)
  {
    /* for each parameter of the copy of the theft' task on mode:
       - V: we destroy the data
       - R,RW: do nothing
       - W,CW: set in field ->version of the original task args the field ->data of the copy args.
    */
    fmt         = arg->origin_fmt;
    count_params = kaapi_format_get_count_params( fmt, orig_task_args );
    for (i=0; i<count_params; ++i)
    {
      mode_param = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, copy_task_args) );
      if (mode_param == KAAPI_ACCESS_MODE_V) 
      {
        fmt_param       = kaapi_format_get_fmt_param(fmt, i, orig_task_args);
        copy_data_param = (void*)kaapi_format_get_data_param(fmt, i, copy_task_args);
        (*fmt_param->dstor)(copy_data_param);
        continue;
      }

      if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param) || KAAPI_ACCESS_IS_CUMULWRITE(mode_param))
      {
        access_param      = kaapi_format_get_access_param(fmt, i, orig_task_args); 
        copy_access_param = kaapi_format_get_access_param(fmt, i, copy_task_args); 

        /* write the value as the version */
        access_param.version = copy_access_param.data;
        kaapi_format_set_access_param(fmt, i, orig_task_args, &access_param );
      }
    }
  }

#if 0 // TG TO TEST IMPACT OF THIS OPTIMISATION
  /* if signaled thread was suspended, move it to the local queue */
  kaapi_wsqueuectxt_cell_t* wcs = arg->origin_thread->wcs;
  if ((wcs != 0) && (arg->origin_thread->stack.sfp->pc == arg->origin_task)) /* means thread has been suspended on this task */
  { 
    kaapi_readmem_barrier();
    kaapi_processor_t* kproc = arg->origin_thread->stack.proc;
    kaapi_assert_debug( kproc != 0);

    if (kaapi_cpuset_has( &wcs->affinity, kproc->kid))
    //  /*kaapi_sched_readyempty(kproc) &&*/ kaapi_thread_hasaffinity(wcs->affinity, kproc->kid))
    {
      kaapi_thread_context_t* kthread = kaapi_wsqueuectxt_steal_cell( wcs );
      if (kthread !=0)
      {
        kaapi_wsqueuectxt_finish_steal_cell(wcs);

        /* Ok, here we have theft the thread and no body else can steal it
           Signal the end of execution of forked task: 
           -if no war && no cw => mark the task as terminated 
           -if war or cw and due to copy => mark the task as aftersteal in order to merge value
        */
        if ((war_param ==0) && (cw_param ==0))
          kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_TERM );
        else
          kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_AFTER );

        kaapi_sched_lock(&kproc->lock);
        kaapi_sched_pushready(kproc, kthread );
        kaapi_sched_unlock(&kproc->lock);
        return;
      }
    }
  }
#endif

  /* signal the task : mark it as executed, the old returned body should have steal flag */
  kaapi_assert_debug( kaapi_task_getbody(arg->origin_task) == kaapi_steal_body );
  kaapi_mem_barrier();
  if ((war_param ==0) && (cw_param ==0))
    kaapi_task_setbody( arg->origin_task, kaapi_term_body );
//    kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_TERM );
  else 
    kaapi_task_setbody( arg->origin_task, kaapi_aftersteal_body );
//    kaapi_task_orstate( arg->origin_task, KAAPI_MASK_BODY_AFTER );

#if 0
  /* toremove */
  kaapi_hws_sched_dec_sync();
  /* toremove */
#endif
}


/**
*/
void kaapi_tasksteal_body( void* taskarg, kaapi_thread_t* thread  )
{
#if defined(KAAPI_USE_CUDA)
  kaapi_thread_context_t* const self_thread = kaapi_self_thread_context();
  kaapi_processor_t* const self_proc = self_thread->proc;
#endif

  unsigned int           i;
  size_t                 count_params;
  kaapi_task_t*          task;
  kaapi_tasksteal_arg_t* arg;
  kaapi_task_body_t      body;          /* format of the stolen task */
  kaapi_format_t*        fmt;
  unsigned int           war_param;     /* */
  unsigned int           cw_param;      /* */

  void*                  orig_task_args;
  void*                  data_param;
  kaapi_access_t         access_param;

  kaapi_access_mode_t    mode_param;
  const kaapi_format_t*  fmt_param;

  void*                  copy_task_args;
  void*                  copy_data_param;
  kaapi_access_t         copy_access_param;

  /* get information of the task to execute */
  arg = (kaapi_tasksteal_arg_t*)taskarg;
 
  /* format of the original stolen task */  
  body            = arg->origin_body;
  kaapi_assert_debug( kaapi_isvalid_body( body ) );

  fmt             = kaapi_format_resolvebybody( body );
  kaapi_assert_debug( fmt !=0 );

  /* the original task arguments */
  orig_task_args  = kaapi_task_getargs(arg->origin_task);

  kaapi_assert_debug( kaapi_task_getbody(arg->origin_task) == kaapi_steal_body );

  /* not a bound task */
  count_params    = kaapi_format_get_count_params(fmt, orig_task_args); 
  arg->copy_task_args = 0;
  
  /**/
  war_param = arg->war_param;
  cw_param  = arg->cw_param;
  
  if (!war_param && !cw_param)
  {
    /* Execute the orinal body function with the original args.
    */
#if defined(KAAPI_USE_CUDA)
    if (self_proc->proc_type == KAAPI_PROC_TYPE_CUDA)
    {
      /* todo_remove */
      if (fmt->entrypoint[KAAPI_PROC_TYPE_CUDA] == 0)
        body(orig_task_args, thread);
      else
        /* todo_remove */
        kaapi_cuda_exectask(self_thread, orig_task_args, fmt);
    }
    else
#endif
    body(orig_task_args, thread);
  }
  else /* it exists at least one w parameter with war dependency or a cw_param: recopies the arguments */
  {
    copy_task_args       = kaapi_thread_pushdata( thread, fmt->size);
    arg->copy_task_args  = copy_task_args;
    arg->origin_fmt      = fmt;

    /* there are possibly non formated params */
    memcpy(copy_task_args, orig_task_args, fmt->size);

    for (i=0; i<count_params; ++i)
    {
      mode_param      = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, orig_task_args) ); 
      fmt_param       = kaapi_format_get_fmt_param(fmt, i, orig_task_args);
      
      if (mode_param == KAAPI_ACCESS_MODE_V) 
      {
        data_param      = kaapi_format_get_data_param(fmt, i, orig_task_args);
        copy_data_param = kaapi_format_get_data_param(fmt, i, copy_task_args);
        (*fmt_param->cstorcopy)(copy_data_param, data_param);
        continue;
      }

      /* initialize all parameters ... */
      access_param              = kaapi_format_get_access_param(fmt, i, orig_task_args);
      copy_access_param         = kaapi_format_get_access_param(fmt, i, copy_task_args);
      copy_access_param.data    = access_param.data;
      copy_access_param.version = 0; /*access_param->version; / * not required... * / */
      
      if (KAAPI_ACCESS_IS_ONLYWRITE(mode_param) || KAAPI_ACCESS_IS_CUMULWRITE(mode_param) )
      {
        if (((war_param & (1<<i)) !=0) || ((cw_param & (1<<i)) !=0))
        { 
          /* allocate new data for the war or cw param, else points to the original data !*/
          kaapi_memory_view_t view = kaapi_format_get_view_param(fmt, i, orig_task_args);
#if defined(KAAPI_DEBUG)
          copy_access_param.data    = calloc(1, kaapi_memory_view_size(&view));
#else
          copy_access_param.data    = malloc(kaapi_memory_view_size(&view));
#endif
          if (fmt_param->cstor !=0) 
            (*fmt_param->cstor)(copy_access_param.data);
            
          /* if cw: init with neutral element with respect to the reduction */
          if ((cw_param & (1<<i)) !=0)
            kaapi_format_redinit_neutral(fmt, i, copy_task_args, copy_access_param.data );
          
          /* set new data to copy of the data with new view */
          kaapi_format_set_access_param(fmt, i, copy_task_args, &copy_access_param );
          kaapi_memory_view_reallocated( &view );
          kaapi_format_set_view_param( fmt, i, copy_task_args, &view );
        }
      }
    }

#if 0 // DEPRECATED
    /* Execute the orinal body function with the copy of args: do not push it... ?
       WHY to push a nop ?
    */
    task = kaapi_thread_toptask( thread );
    kaapi_task_init( task, kaapi_nop_body, copy_task_args);
    kaapi_thread_pushtask( thread );
#endif

    /* call directly the stolen body function */
#if defined(KAAPI_USE_CUDA)
    if (self_proc->proc_type == KAAPI_PROC_TYPE_CUDA)
    {
      /* todo_remove */
      if (fmt->entrypoint[KAAPI_PROC_TYPE_CUDA] == 0)
        body(copy_task_args, thread);
      else
        /* todo_remove */
        kaapi_cuda_exectask(self_thread, copy_task_args, fmt);
    }
    else
#endif
    body( copy_task_args, thread);
  }

  /* push task that will be executed after all created task by the user task */
  task = kaapi_thread_toptask( thread );
  kaapi_task_init( task, kaapi_taskwrite_body, arg );
  kaapi_thread_pushtask( thread );

}
