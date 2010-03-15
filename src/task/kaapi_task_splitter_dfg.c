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
int kaapi_task_splitter_dfg( kaapi_thread_context_t* thread, kaapi_task_t* task, int count, struct kaapi_request_t* array)
{
  int i;
  kaapi_request_t*        request    = 0;
  kaapi_task_t*           thief_task   = 0;
  kaapi_thread_context_t* thief_thread = 0;
  kaapi_tasksteal_arg_t*  argsteal;
#if 0
  kaapi_tasksig_arg_t*    argsig;
#endif
  

#if defined(KAAPI_CONCURRENT_WS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&thread->proc->lock) == 1+_kaapi_get_current_processor()->kid );
#endif
  kaapi_assert_debug( task !=0 );
  kaapi_assert_debug( kaapi_task_getbody(task) ==kaapi_suspend_body );

  /* find the first request in the list */
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
  {
    if (kaapi_request_ok( &array[i] )) 
    {
      request = &array[i];
#if 0
      fprintf(stdout,"%i kproc reply ok to:%p, @req=%p\n", kaapi_get_current_kid(), (void*)kaapi_all_kprocessors[i], (void*)&array[i] );
      fflush(stdout);
#endif
      break;
    }
  }
  kaapi_assert(request !=0);

  char buffer[1024];
  size_t sz_write = 0;
  sz_write += snprintf( buffer, 1024, "[steal] task: @=%p, stack: @=%p", task, thread);
  fprintf(stdout, "%s\n", buffer);
  fflush(stdout);

  /* - create the task steal that will execute the stolen task
     The task stealtask stores:
       - the original stack
       - the original task pointer
       - the pointer to shared data with R / RW access data
       - and at the end it reserve enough space to store original task arguments
     The original body is saved as the extra body of the original task data structure.
  */
  thief_thread = request->thread;

  thief_task = _kaapi_thread_toptask( thief_thread );
  kaapi_task_init( thief_task, kaapi_tasksteal_body, _kaapi_thread_pushdata(thief_thread, sizeof(kaapi_tasksteal_arg_t)) );
  argsteal = kaapi_task_getargst( thief_task, kaapi_tasksteal_arg_t );
  argsteal->origin_thread         = thread;
  argsteal->origin_task           = task;
  
  _kaapi_thread_pushtask( thief_thread );

#if 0
  /* reply: several cases
     - if complete steal of the task -> signal sould pass the body to aftersteal body
     If steal an
  */
  thief_task       = _kaapi_thread_toptask( thief_thread );
  kaapi_task_init( thief_task, kaapi_tasksig_body, _kaapi_thread_pushdata(thief_thread, sizeof(kaapi_tasksig_arg_t)) );
  argsig           = kaapi_task_getargst( thief_task, kaapi_tasksig_arg_t);
  argsig->victim   = thread;
  argsig->task2sig = task;
  _kaapi_thread_pushtask( thief_thread );
#endif

  _kaapi_request_reply( request, thief_thread, 1 ); /* success of steal */
  return 1;
}
