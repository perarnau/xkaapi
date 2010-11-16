/*
** kaapi_task_finalize.c
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


/**
*/
void kaapi_taskfinalize_body( void* args, kaapi_thread_t* thread )
{
  kaapi_stealcontext_t* const sc = (kaapi_stealcontext_t*)args;

  kaapi_assert_debug(!(sc->header.flag & KAAPI_SC_PREEMPTION));

  /* ensure all working thieves are done. the steal
     sync has been done in kaapi_task_end_adaptive
   */
  while (KAAPI_ATOMIC_READ(&sc->thieves.count))
    kaapi_slowdown_cpu();

  /* avoid read reordering */
  kaapi_readmem_barrier();

  /* restore the upper frame */
  kaapi_thread_restore_frame(thread - 1, &sc->frame);
}


/**
*/
void kaapi_task_end_adaptive( kaapi_stealcontext_t* sc )
{
  /* end with the adapt dummy task -> change body with nop */

  kaapi_thread_t* const thread = kaapi_self_thread();

  /* avoid to steal old instance of this task */
  sc->splitter = 0;
  sc->argsplitter = 0;
  
  /* if this is a preemptive algorithm, it is assumed the
     user has preempted all the children (not doing so is
     an error). we restore the frame and return without
     waiting for anyting.
   */
  if (sc->header.flag & KAAPI_SC_PREEMPTION)
  {
    kaapi_assert_debug(sc->thieves.list.head == 0);
    kaapi_task_setbody(sc->ownertask, kaapi_nop_body);
    kaapi_thread_restore_frame(thread, &sc->frame);
    return ;
  }

  /* steal synchronization protocol. it is done
     here since we need to access the ownertask
     before the nop body is set.
     see comments in kaapi_task_preempt_head.c
  */
  kaapi_synchronize_steal(sc);

  kaapi_task_setbody(sc->ownertask, kaapi_nop_body);

  /* not a preemptive algorithm. push a finalization task
     to wait for thieves and block until finalization done.
   */
  kaapi_task_init
    (kaapi_thread_toptask(thread), kaapi_taskfinalize_body, sc);
  kaapi_thread_pushtask(thread);
  kaapi_sched_sync();
}
