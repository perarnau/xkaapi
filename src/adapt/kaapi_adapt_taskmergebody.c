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

  /* wait end of theives */
  
  /* prempt each a lived thieves and merge results */

  if (sc->msc == sc )
  {
  }

  /* signal the master */
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
    KAAPI_ATOMIC_DECR(&sc->msc->thieves.count);
  }
}
