/*
** kaapi_task_finalize.c
** xkaapi
** 
**
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
  kaapi_assert_debug(!(sc->flag & KAAPI_SC_PREEMPTION));

  /* avoid read reordering */
  kaapi_readmem_barrier();

  /* ensure all working thieves are done. the steal
     sync has been done in kaapi_task_end_adaptive
   */
  while (KAAPI_ATOMIC_READ(&sc->thieves.count))
    kaapi_slowdown_cpu();
}


/**
*/
int kaapi_task_end_adaptive( kaapi_task_t* mergetask )
{
  kaapi_taskmerge_arg_t* merge_arg;
  kaapi_stealcontext_t* sc;
  kaapi_thread_t* thread;
  kaapi_thread_context_t* self_thread;

  merge_arg = kaapi_task_getargst(mergetask, kaapi_taskmerge_arg_t);
  sc = (kaapi_stealcontext_t*)merge_arg->shared_sc.data;

  /* Synchronize with the theft on the current thread.
     After the following instruction, we have:
     - no more theft is under stealing and master counter or list of thief is
     correctly setted.
  */
  self_thread = kaapi_self_thread_context();
  thread = kaapi_threadcontext2thread( self_thread );
  kaapi_synchronize_steal(self_thread);

  if (sc->msc == sc)
  {
    /* if this is a preemptive algorithm, it is assumed the
       user has preempted all the children (not doing so is
       an error). we restore the frame and return without
       waiting for anyting.
     */
    if (sc->flag & KAAPI_SC_PREEMPTION)
    {

      if (sc->thieves.list.head != 0) 
        return EAGAIN;
      kaapi_sched_sync_(self_thread);
      kaapi_thread_restore_frame(thread, &merge_arg->saved_frame);
      return 0;
    }

    /* not a preemptive algorithm. push a finalization task
       to wait for thieves and block until finalization done.
    */
    kaapi_task_init(
      kaapi_thread_toptask(thread), 
      kaapi_taskfinalize_body, 
      sc
    );
    kaapi_thread_pushtask(thread);

    kaapi_sched_sync_(self_thread);
    kaapi_thread_restore_frame(thread, &merge_arg->saved_frame);

    return 0;
  }
  
  /* else merge of a non master context 
     - here no more thief
     Then flush memory & signal master context
  */
  kaapi_writemem_barrier();
  return 0;
}
