/*
** kaapi_task_signal.c
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
void kaapi_aftersteal_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
  int i, countparam;
  kaapi_format_t* fmt;           /* format of the task */
  void* arg;
  
  /* the task has been stolen: format contains the format of the task */
  fmt = task->format;
  kaapi_assert_debug( fmt !=0 );

  KAAPI_LOG(100, "aftersteal task: 0x%p\n", (void*)task );

  /* report data to version to global data */
  arg = kaapi_task_getargs(task);
  countparam = fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);

#if defined(KAAPI_DEBUG)
    if (m == KAAPI_ACCESS_MODE_V)
    {
      /* TODO: improve management of shared data, if it is a big data or not... */
      void* param_data __attribute__((unused)) = (void*)(fmt->off_params[i] + (char*)kaapi_task_getargs(task));
//      printf("After steal task:%p, name: %s, value: %i\n", (void*)task, fmt->name, *(int*)param_data );
    }    
    else 
#endif
    if (KAAPI_ACCESS_IS_ONLYWRITE(m))
    {
      void* param = (void*)(fmt->off_params[i] + (char*)arg);
      kaapi_format_t* fmt_param = fmt->fmt_params[i];
      kaapi_access_t* access = (kaapi_access_t*)(param);
      /* TODO: improve management of shared data, if it is a big data or not... */
//      printf("After steal task:%p, name: %s, W object: version: %i, data: %i\n", (void*)task, fmt->name, *(int*)access->version, *(int*)access->data );
kaapi_assert_debug( access->data != access->version );

      (*fmt_param->assign)( access->data, access->version );
      (*fmt_param->dstor) ( access->version );
      free(((kaapi_gd_t*)access->version)-1);
      access->version = 0;
    }
#if defined(KAAPI_DEBUG)
    else if (KAAPI_ACCESS_IS_READ(m)) /* rw : if above, not here */
    { /* nothing to do ?
      */    
      void* param __attribute__((unused)) = (void*)(fmt->off_params[i] + (char*)arg);
      kaapi_access_t* access __attribute__((unused)) = (kaapi_access_t*)(param);
//      printf("After steal task:%p, name: %s, R object: version: %i, data: %i\n", (void*)task, fmt->name, *(int*)access->version, *(int*)access->data );
    }
#endif
  }
}


/**
*/
void kaapi_tasksig_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
  /*
    printf("Thief end, @stack: 0x%p\n", stack);
    fflush( stdout );
  */
  kaapi_tasksig_arg_t* argsig;
  kaapi_task_t* task2sig;

  argsig = kaapi_task_getargst( task, kaapi_tasksig_arg_t);
  task2sig = argsig->task2sig;
  KAAPI_LOG(100, "signaltask: 0x%p -> task2signal: 0x%p\n", (void*)task, (void*)task2sig );

  if (!(argsig->flag & KAAPI_REQUEST_FLAG_PARTIALSTEAL))
  {
    kaapi_task_setbody(task2sig, &kaapi_aftersteal_body );
    KAAPI_LOG(100, "signaltask DFG task stolen: 0x%p\n", (void*)task2sig );
  }

  /* flush in memory all pending write ops */  
  kaapi_writemem_barrier();

  if (!(argsig->flag & KAAPI_REQUEST_FLAG_PARTIALSTEAL))
  {
    kaapi_task_setstate(task2sig, KAAPI_TASK_S_TERM );
    KAAPI_LOG(100, "signaltask DFG task stolen: 0x%p\n", (void*)task2sig );
  }
  else 
  {
    if ( !(argsig->flag & KAAPI_TASK_ADAPT_NOSYNC) )
    {
      kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task2sig->sp;/* do not use kaapi_task_getargs !!! */
      kaapi_assert_debug( ta !=0 );
      KAAPI_ATOMIC_DECR( &ta->thievescount );
    } 
    if ( !(argsig->flag & KAAPI_TASK_ADAPT_NOPREEMPT) )
    { /* mark result as term */
      argsig->result->flag = argsig->result->flag & ~KAAPI_RESULT_MASK_EXEC;
    }
  }
}

