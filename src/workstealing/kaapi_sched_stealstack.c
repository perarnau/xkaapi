/*
** kaapi_sched_stealstack.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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

/** 
*/
int kaapi_sched_stealstack  ( kaapi_stack_t* stack )
{
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  int count;
  int replycount;  

  count = KAAPI_ATOMIC_READ( (kaapi_atomic_t*)stack->hasrequest );
  if (count ==0) return 0;

  if (kaapi_stack_isempty( stack)) return 0;

  /* reset dfg constraints evaluation */
  
  /* iterate through all the tasks from task_bot until task_top */
  task_bot = kaapi_stack_bottomtask(stack);
  task_top = kaapi_stack_toptask(stack);

  replycount = 0;

  while ((count >0) && (task_bot !=0) && (task_bot != task_top))
  {
    if (task_bot == 0) return replycount;

    kaapi_assert_debug( task_bot != 0 );
    /* task body == 0 no task after can stop 
       task body == retn : no steal
       
    */
    if (task_bot->splitter !=0)
    {
      int retval = (*task_bot->splitter)(stack, task_bot, count, stack->requests);
      count -= retval;
      replycount += retval;
      kaapi_assert_debug( count >=0 );
    }

    /* test next task */  
    ++task_bot;
  }

  if (replycount >0)
  {
    KAAPI_ATOMIC_SUB( (kaapi_atomic_t*)stack->hasrequest, replycount );
    kaapi_assert_debug( *stack->hasrequest >= 0 );
  }

  return replycount;
}
