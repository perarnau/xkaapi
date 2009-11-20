/*
** kaapi_task_splitter_dfg.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
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

/** Return the number of splitted parts (at most 1 if the task may be steal)
*/
int kaapi_task_splitter_dfg(kaapi_stack_t* stack, kaapi_task_t* task, int count, struct kaapi_request_t* array)
{
  int i;
  int countparam;
  void** param_data = 0;         /* list of global data to read */
  kaapi_format_t* fmt;           /* format of the task */
  kaapi_task_body_t body;        /* body of the task */

  kaapi_assert_debug (task !=0);
  
  body = task->body;
  
  /* \TODO verify this line
     In case of executed task: nothing has to be. The DFG access chains will be with the next task
  */
  if (body ==0) return 0;
  
  /** CAS not required in this implementation (cooperative) 
  if (KAAPI_ATOMIC_CASPTR( &task->body, body, &kaapi_suspend_body))
  */
  if ((body ==0) ||(body == &kaapi_suspend_body)) fmt = kaapi_format_resolvebybody( task->format );
  else fmt = kaapi_format_resolvebybody( body );
  if (fmt ==0) return 0;
  
  /* allocate pointer to data to read, if the stack will be steal */
  param_data = alloca( sizeof(void*) * fmt->count_params );
  kaapi_assert( param_data !=0 );

  if ((body ==0) || (body == &kaapi_suspend_body))
  {
    countparam = fmt->count_params;
    for (i=0; i<countparam; ++i)
    {
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
      if (m != KAAPI_ACCESS_MODE_V) 
      {
        /* the access is just before the value pointed by the pointer (shared == pointer) returned
           by kaapi_stack_pushshareddata
        */
        void* data = *(void**)(fmt->off_params[i] + (char*)task->sp);
        kaapi_gd_t* access = ((kaapi_gd_t*)data) -1;
        if (KAAPI_ACCESS_IS_ONLYWRITE(m)) 
        {
          if (body ==0) 
          {
            access->last_version = data;                        /* this is the data */
            access->last_mode = KAAPI_ACCESS_MASK_MODE_R;       /* and it could be read */
          }
          else { 
            access->last_version = 0;
            access->last_mode = m;
          }
        }
        if (KAAPI_ACCESS_IS_READ(m)) /* also RW */
        { 
          /* test if concurrent access */
          if (   (access->last_version !=0)                     /* version produced */
              && (access->last_mode == m)                       /* same mode (R or RW) */
              && (m != KAAPI_ACCESS_MODE_RW)                    /* and not RW */
            )
          {
            param_data[i] = access->last_version;
            access->last_mode = m;
          }
          else /* no concurrent: set last_version to 0 ! */
          { 
            access->last_version = 0;
          }
        }
      }
    }
    return 0;
  }
  else 
  {
    int waitparam;
    
    /* not yet steal */
    /* TODO: all methods on format should take the task in parameter to deal variable number of parameters */

    /* update versions */
    waitparam = countparam = fmt->count_params;
    for (i=0; i<countparam; ++i)
    {
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
      if (m == KAAPI_ACCESS_MODE_V) 
      {
        --waitparam;
        param_data[i] = (void*)(fmt->off_params[i] + (char*)task->sp);
      } 
      else 
      {
        /* the access is just before the value pointed by the pointer (shared == pointer) returned
           by kaapi_stack_pushshareddata
        */
        kaapi_gd_t* access = ((kaapi_gd_t*)*(void**)(fmt->off_params[i] + (char*)task->sp)) -1;
        
        if (KAAPI_ACCESS_IS_ONLYWRITE(m)) 
        {
          --waitparam;
          access->last_version = 0;
          param_data[i] = 0; /* not use: but for correctness... & debugging */
        }
        if (KAAPI_ACCESS_IS_READ(m)) /* also RW */
        { 
          /* test if concurrent access */
          if (   (access->last_version !=0)                     /* version produced */
              && (access->last_mode == m)                       /* same mode (R or RW) */
              && (m != KAAPI_ACCESS_MODE_RW)                    /* and not RW */
            )
          {
            --waitparam;
            param_data[i] = access->last_version;
          }
          else /* no concurrent: set last_version to 0 ! */
          { 
            access->last_version = 0;
          }
        }
        access->last_mode = m;
      }
    }
    kaapi_assert_debug( waitparam >= 0);
    if ((waitparam ==0) && kaapi_task_isstealable(task) )
    {
       goto steal_the_task;
    }
    return 0;
  }
  
steal_the_task:
  KAAPI_LOG(50, "dfgsplitter task: 0x%x\n", task);
  kaapi_assert_debug( task->body !=0);
  kaapi_assert_debug( task->body !=kaapi_suspend_body);
  {
    kaapi_request_t* request   = 0;
    kaapi_stack_t* thief_stack = 0;
    kaapi_task_t* steal_task   = 0;
    
    /* find the first request in the list */
    for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
    {
      if (kaapi_request_ok( &array[i] )) 
      {
        request = &array[i];
        break;
      }
    }

    if (request ==0) return 0;
      
    task->body = &kaapi_suspend_body;
    task->format = body;
    
    /* - create the task steal that will execute the stolen task
       The task stealtask stores:
         - the original stack
         - the original task pointer
         - the original body
         - the pointer to shared data with R / RW access data
         - and at the end it reserve enough space to store original task arguments
    */
    thief_stack = request->stack;
    
    steal_task = kaapi_stack_toptask( thief_stack );
    kaapi_task_init(thief_stack, steal_task, KAAPI_TASK_STICKY );
    kaapi_task_setargs( steal_task, kaapi_stack_pushdata(thief_stack, sizeof(kaapi_tasksteal_arg_t)+sizeof(void*)*countparam) );
    kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( steal_task, kaapi_tasksteal_arg_t );
    arg->origin_stack     = stack;
    arg->origin_task      = task;
    arg->origin_body      = body;
    arg->origin_task_args = (void**)(arg+1);
    arg->copy_arg = kaapi_stack_pushdata(thief_stack, fmt->size);
    for (i=0; i<countparam; ++i)
      arg->origin_task_args[i] = param_data[i];
    
    kaapi_task_setbody( steal_task, &kaapi_tasksteal_body );
    kaapi_stack_pushtask( thief_stack );

    kaapi_request_reply( stack, task, request, thief_stack, 1 ); /* success of steal */
    return 1;
  }
}
