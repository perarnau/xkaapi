/*
** kaapi_task_steal.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
# include "../machine/cuda/kaapi_cuda_task_steal_body.h"
#endif

/* toremove */
extern void kaapi_hws_sched_dec_sync(void);
/* toremove */

/**
*/
void kaapi_taskwrite_body( 
  void*           taskarg, 
  kaapi_thread_t* thread  __attribute__((unused)),
  kaapi_task_t*   task
)
{
  unsigned int i;
  size_t count_params;

  const kaapi_format_t* fmt;
  void*                 orig_task_args;
  kaapi_access_t        access_param;

  kaapi_access_mode_t   mode;
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
      mode            = kaapi_format_get_mode_param(fmt, i, copy_task_args); 
      mode_param      = KAAPI_ACCESS_GET_MODE( mode ); 
      if (mode_param == KAAPI_ACCESS_MODE_V) 
      {
        fmt_param       = kaapi_format_get_fmt_param(fmt, i, orig_task_args);
        copy_data_param = (void*)kaapi_format_get_data_param(fmt, i, copy_task_args);
        (*fmt_param->dstor)(copy_data_param);
        continue;
      }

      if ( KAAPI_ACCESS_IS_ONLYWRITE(mode_param) 
       || (KAAPI_ACCESS_IS_CUMULWRITE(mode_param) && ((mode & KAAPI_ACCESS_MODE_IP) ==0)))
      {
        access_param      = kaapi_format_get_access_param(fmt, i, orig_task_args); 
        copy_access_param = kaapi_format_get_access_param(fmt, i, copy_task_args); 

        /* write the value as the version */
        access_param.version = copy_access_param.data;
        kaapi_format_set_access_param(fmt, i, orig_task_args, &access_param );
      }
      else if (KAAPI_ACCESS_IS_STACK(mode_param))
      { /* never merge result here */
        copy_access_param = kaapi_format_get_access_param(fmt, i, copy_task_args);
//Temporary delete free(copy_access_param.data);
        copy_access_param.data = 0;

        /* suppress war_param bit to avoid pushing a merge task */
        war_param &= ~(1<<i);
#if 0
kaapi_memory_view_t view = kaapi_format_get_view_param(fmt, i, copy_task_args);
printf("Delete temporary stack object: size=%i\n",(int)kaapi_memory_view_size(&view) );
#endif
      }
    }
  }

  /* order write to the memory before changing the origin task' state */
  kaapi_writemem_barrier();

  /* lock the original task to ensure exclusive access between preemption & end of task */
  kaapi_task_lock( arg->origin_task );

  /* signal the original task : mark it as executed, the old returned body should have steal flag */
  if ((war_param ==0) && (cw_param ==0))
    kaapi_task_markterm( arg->origin_task );
  else 
    kaapi_task_markaftersteal( arg->origin_task );

//  printf("Task: pc:%p end steal\n", (void*)arg->origin_task); fflush(stdout);
  kaapi_task_unlock( arg->origin_task );
}


/**
*/
void kaapi_tasksteal_body( void* taskarg, kaapi_thread_t* thread  )
{
  unsigned int           i;
  size_t                 count_params;
  kaapi_task_t*          task;
  kaapi_tasksteal_arg_t* arg;
  kaapi_task_body_t      body;          /* format of the stolen task */
  const kaapi_format_t*  fmt;
  unsigned int           war_param;     /* */
  unsigned int           cw_param;      /* */

  void*                  orig_task_args;
  void*                  data_param;
  kaapi_access_t         access_param;

  kaapi_access_mode_t    mode;
  kaapi_access_mode_t    mode_param;
  const kaapi_format_t*  fmt_param;

  void*                  copy_task_args;
  void*                  copy_data_param;
  kaapi_access_t         copy_access_param;

  /* get information of the task to execute */
  arg = (kaapi_tasksteal_arg_t*)taskarg;

  /* format of the original stolen task */  
  body            = arg->origin_body;
  fmt             = arg->origin_fmt;
  if (fmt == 0)
    arg->origin_fmt = fmt = kaapi_format_resolvebybody(body);
  kaapi_assert_debug( fmt !=0 );

  /* the original task arguments */
  orig_task_args  = kaapi_task_getargs(arg->origin_task);

  kaapi_assert_debug( kaapi_task_getbody(arg->origin_task) == body );

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
    if (kaapi_get_current_processor()->proc_type == KAAPI_PROC_TYPE_CUDA)
    {
      if (fmt->entrypoint[KAAPI_PROC_TYPE_CUDA] == 0)
        body( orig_task_args, thread );
      else
        kaapi_cuda_task_steal_body( thread, fmt, orig_task_args );
    }
    else
#endif
    {
//	if ( fmt != 0 )
//		kaapi_mem_host_map_sync_ptr( fmt, orig_task_args );
      body( orig_task_args, thread );
    }
  }
  else /* it exists at least one w parameter with war dependency or a cw_param: recopies the arguments */
  {
    size_t task_size = kaapi_format_get_size( fmt, orig_task_args );
    copy_task_args       = kaapi_thread_pushdata( thread, (uint32_t)task_size);
    arg->copy_task_args  = copy_task_args;
    arg->origin_fmt      = fmt;

    /* WARNING there are possibly non formated params */
    /* ERROR: do not work if variable size task: to be virtualized through the format object */
    kaapi_format_task_copy( fmt, copy_task_args, orig_task_args );
//    memcpy(copy_task_args, orig_task_args, fmt->size);

    for (i=0; i<count_params; ++i)
    {
      mode            = kaapi_format_get_mode_param(fmt, i, orig_task_args); 
      mode_param      = KAAPI_ACCESS_GET_MODE( mode ); 
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
      
      if (KAAPI_ACCESS_IS_STACK(mode_param))
      {
        copy_access_param.data = 0;
        kaapi_memory_view_t view = kaapi_format_get_view_param(fmt, i, orig_task_args);
        if (kaapi_memory_view_size(&view) < 1024 )
        { /* try to do stack allocation */
          copy_access_param.data = kaapi_thread_pushdata(thread, (int)kaapi_memory_view_size(&view));
        }
        if (copy_access_param.data ==0)
        {
#if defined(KAAPI_DEBUG)
          copy_access_param.data   = calloc(1, kaapi_memory_view_size(&view));
#else
          copy_access_param.data   = malloc(kaapi_memory_view_size(&view));
#endif
printf("Bad: heap allocation\n");
        }
        kaapi_assert_debug( copy_access_param.data != 0 );
        
        if (fmt_param->cstor !=0) 
          (*fmt_param->cstor)(copy_access_param.data);

        kaapi_memory_view_t view_dest = view;
        kaapi_memory_view_reallocated( &view_dest );
        (*fmt_param->assign)(copy_access_param.data, &view_dest, access_param.data, &view );
        
        /* set new data to copy with new view */
        kaapi_format_set_access_param(fmt, i, copy_task_args, &copy_access_param );
        kaapi_format_set_view_param( fmt, i, copy_task_args, &view_dest );
      }
      /* allocate new data for the war or cw param or stack data, else points to the original data
         if mode is CW and Inplace flag is set, then do not copy data.
      */
      else if ( KAAPI_ACCESS_IS_ONLYWRITE(mode_param) 
            || (KAAPI_ACCESS_IS_CUMULWRITE(mode_param) && ((mode & KAAPI_ACCESS_MODE_IP) ==0)) 
      )
      {
        if (((war_param & (1<<i)) !=0) || ((cw_param & (1<<i)) !=0))
        { 
          kaapi_memory_view_t view = kaapi_format_get_view_param(fmt, i, orig_task_args);
#if defined(KAAPI_DEBUG)
          copy_access_param.data   = calloc(1, kaapi_memory_view_size(&view));
#else
          copy_access_param.data   = malloc(kaapi_memory_view_size(&view));
#endif
          if (fmt_param->cstor !=0) 
            (*fmt_param->cstor)(copy_access_param.data);
          if (KAAPI_ACCESS_IS_STACK(mode_param))
          {
            kaapi_memory_view_t view_dest = view;
            kaapi_memory_view_reallocated( &view_dest );
            (*fmt_param->assign)(copy_access_param.data, &view_dest, access_param.data, &view );
          }
          else
            /* if cw: init with neutral element with respect to the reduction */
            if ((cw_param & (1<<i)) !=0)
              kaapi_format_redinit_neutral(fmt, i, copy_task_args, copy_access_param.data );
          
          /* set new data to copy with new view */
          kaapi_format_set_access_param(fmt, i, copy_task_args, &copy_access_param );
          kaapi_memory_view_reallocated( &view );
          kaapi_format_set_view_param( fmt, i, copy_task_args, &view );
        }
      }
    }


    /* call directly the stolen body function */
//	if ( fmt != 0 )
//		kaapi_mem_host_map_sync_ptr( fmt, copy_task_args );
      body( copy_task_args, thread);
  }

  /* push task that will be executed after all created tasks spawned
     by the user task 'body'
  */
  task = kaapi_thread_toptask( thread );
  kaapi_task_init( task, (kaapi_task_body_t)kaapi_taskwrite_body, arg );
  kaapi_thread_pushtask( thread );
}
