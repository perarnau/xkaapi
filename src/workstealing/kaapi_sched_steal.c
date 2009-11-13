/*
** kaapi_sched_steal.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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

/** TO DO !!!!
- avoid cas ? 
- avoid incr ?

1/ entièrement coopératif !
  task -> pile de task
  begin_concurrent( event -> action ) & mask
  event= steal, stop, save, restore:
      steal(task) -> task
      stop() -> ()
      save(taks) -> continuation task
      restore(continuation task)->()

2/ comment le rendre concurrent ?
   ABA problème -> ll/sc ou cas + version
   1 cas: task: pc & sp ?  pc->sp
*/
int kaapi_sched_steal ( kaapi_thread_descr_t* victim_thread )
{
  kaapi_task_t* task;

  kaapi_assert_debug (victim_thread !=0);

  if (kaapi_stack_isempty( &victim_thread->_stack)) return ESRCH;
  if (kaapi_barrier_td_isterminated(&kaapi_barrier_term)) return EINTR; /* terminaison */

    
  /* iterate through all the tasks */
  task_bot = kaapi_stack_bottom(&victim_proc->_stack);
  task_top = kaapi_stack_top(&victim_proc->_stack);

  while (task_bot != task_top)
  {
    if ( (task_bot !=0) && (task_bot->body != &kaapi_retn_body) && (task_bot->body != 0) &&
         (kaapi_task_state(task_bot) == KAAPI_TASK_INIT) )
    {
      /* steal the task */
      *kaapi_stack_top( &victim_proc->_stack ) = *task_bot;
      kaapi_task_state(task_bot) = KAAPI_TASK_STOLEN;
      kaapi_stack_push( &victim_proc->_stack );
      return 0;
    }

    /* test next task */  
    ++task_bot;
  }
  return ESRCH;
}
