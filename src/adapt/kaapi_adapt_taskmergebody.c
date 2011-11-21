/*
 ** xkaapi
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
void kaapi_taskfinalize_body( void* sp, kaapi_thread_t* thread )
{
  kaapi_taskmerge_arg_t* const merge_arg = (kaapi_taskmerge_arg_t*)sp;
  kaapi_stealcontext_t* const sc = (kaapi_stealcontext_t*)merge_arg->shared_sc.data;
  kaapi_assert_debug(!(sc->flag & KAAPI_SC_PREEMPTION));

  kaapi_thread_restore_frame(thread, &merge_arg->saved_frame);
  kaapi_synchronize_steal(kaapi_self_thread_context());
  
  /* avoid read reordering */
  kaapi_readmem_barrier();

  /* ensure all working thieves are done. the steal
     sync has been done in kaapi_task_end_adaptive
   */
  while (KAAPI_ATOMIC_READ(&sc->thieves.count))
    kaapi_slowdown_cpu();
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->thieves.count) == 0);
}


/* Merge body task: arg is the steal context
*/
void kaapi_taskadaptmerge_body(void* sp, kaapi_thread_t* thread)
{
  kaapi_taskmerge_arg_t* const arg = (kaapi_taskmerge_arg_t*)sp;
  kaapi_stealcontext_t* const sc = (kaapi_stealcontext_t*)arg->shared_sc.data;
  kaapi_thread_context_t* const self_thread = kaapi_self_thread_context();

  /* Synchronize with the theft on the current thread.
     After the following instruction, we have:
     - no more theft is under stealing and master counter or list of thief is
     correctly setted.
  */
  kaapi_synchronize_steal(self_thread);

  /* If master thread */
  if (sc->msc == sc )
  {
    /* if this is a preemptive algorithm, it is assumed the
       user has preempted all the children (not doing so is
       an error). we restore the frame and return without
       waiting for anyting.
     */
    if (sc->flag & KAAPI_SC_PREEMPTION)
    {

      if (sc->thieves.list.head != 0) 
        return /*EAGAIN*/;
      return;
    }

    /* not a preemptive algorithm. push a finalization task
       to wait for thieves and block until finalization done.
    */
    kaapi_task_init(
      kaapi_thread_toptask(thread), 
      kaapi_taskfinalize_body, 
      sp
    );
    kaapi_thread_pushtask(thread);
    return;
  }
  
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->thieves.count) == 0);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->msc->thieves.count) >= 0);

  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->thieves.count) == 0);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->msc->thieves.count) >= 0);

  /* Else finalization of a thief: signal the master */
  if ( sc->flag == KAAPI_SC_PREEMPTION)
  {
    kaapi_assert_debug( sc->ktr != 0);
    /* report local list of thief to the remote ktr and finish */
    kaapi_atomic_lock(&sc->ktr->lock);
    sc->ktr->thief_task = 0;
    sc->ktr->thief_of_the_thief_head = sc->thieves.list.head;
    sc->ktr->thief_of_the_thief_tail = sc->thieves.list.tail;
    kaapi_atomic_unlock(&sc->ktr->lock);
  }
  else 
  {
    kaapi_assert_debug( sc->ktr == 0);

    /* Then flush memory & signal master context
    */
    kaapi_writemem_barrier();

    /* here a remote read of sc->msc->thieves may be avoided if
       sc stores a  pointer to the master count.
    */
    kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->msc->thieves.count) > 0);
    KAAPI_ATOMIC_DECR(&sc->msc->thieves.count);
  }
}
