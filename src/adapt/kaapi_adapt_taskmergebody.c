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

#if 0// OLD: code was inlined
/**
*/
void kaapi_taskfinalize_body( void* sp, kaapi_thread_t* thread )
{
  kaapi_taskmerge_arg_t* const merge_arg = (kaapi_taskmerge_arg_t*)sp;
  kaapi_stealcontext_t* const sc = (kaapi_stealcontext_t*)merge_arg->shared_sc.data;
  kaapi_assert_debug(!(sc->flag & KAAPI_SC_PREEMPTION));
  
  /* ensure all working thieves are done. the steal
     sync has been done in kaapi_task_end_adaptive
   */
  while (KAAPI_ATOMIC_READ(&sc->thieves.count))
    kaapi_slowdown_cpu();
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&sc->thieves.count) == 0);
}
#endif


/* Merge body task: arg is the steal context
*/
void kaapi_taskadaptmerge_body(void* sp, kaapi_thread_t* thread)
{
  kaapi_taskmerge_arg_t* const arg    = (kaapi_taskmerge_arg_t*)sp;
  kaapi_stealcontext_t* const sc      = (kaapi_stealcontext_t*)arg->shared_sc.data;
  kaapi_processor_t* const self_kproc = kaapi_get_current_processor();

  /* Not a master thread... 
    - merge task > adapt body task
    - adapt body task has setted unsplittable on it self
    - no theft is under stealing after synchronize point 
    - master counter or list of thief is correctly setted.
  */
#if 0
  kaapi_synchronize_steal(self_kproc);
#else
  kaapi_synchronize_steal_thread(self_kproc->thread);
#endif

  /* If I'm the master task */
  if (sc->msc == sc )
  {
    /* if this is a preemptive algorithm, it is assumed the
       user has preempted all the thieves (not doing so is
       an error). we restore the frame and return without
       waiting for anyting.
     */
    if (sc->flag & KAAPI_SC_PREEMPTION)
    {

      if (sc->thieves.list.head != 0) 
        return /*EAGAIN*/;
      return;
    }

    /* Ensure all thieves are done.
       Optimization: this mergetask must becomes steal when
     */
    while (KAAPI_ATOMIC_READ(&sc->thieves.count))
      kaapi_slowdown_cpu();
#if defined(KAAPI_DEBUG)
    sc->state = 0;
#endif

    return;
  }
    
  /* Else finalization of a thief: signal the master */
  if ( sc->flag == KAAPI_SC_PREEMPTION)
  {
    kaapi_assert_debug( 0 );
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

#if defined(KAAPI_DEBUG)
    kaapi_assert(sc->version == sc->msc->version );
    kaapi_assert(sc->msc->state == 1);
    sc->state = 0;
    kaapi_mem_barrier();
#endif

    /* here a remote read of sc->msc->thieves may be avoided if
       sc stores a  pointer to the master count.
    */
#if defined(KAAPI_DEBUG)
    int v0 = KAAPI_ATOMIC_READ(&sc->msc->thieves.count);
    kaapi_assert_debug( v0 >0 );
#endif

    /* Then flush memory & signal master context
    */
    kaapi_writemem_barrier();
    KAAPI_ATOMIC_DECR(&sc->msc->thieves.count);

#if defined(KAAPI_DEBUG)
    int v1 = KAAPI_ATOMIC_READ(&sc->msc->thieves.count);
    kaapi_assert_debug( v1 >=0);
    //printf("%i end, cnt=%i -> cnt=%i\n", self_kproc->kid, v0, v1);
#endif
  }
}
