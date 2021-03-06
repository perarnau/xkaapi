/*
 ** xkaapi
 ** 
 ** Copyright 2009,2010,2011,2012 INRIA.
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


/* Signal body task
   - used if the stolen task a non adaptive task and preemption flag was unset
   on the victim adaptive task
   - arg is the master steal context to signal (remotely)
*/
void kaapi_tasksignaladapt_body(void* sp, kaapi_thread_t* thread)
{
  kaapi_stealcontext_t* const msc  = (kaapi_stealcontext_t*)sp;
//printf("TO TEST\n");
  kaapi_assert_debug( msc->flag != KAAPI_SC_PREEMPTION );

  /* Then flush memory & signal master context
  */
  kaapi_writemem_barrier();

  /* here a remote read of sc->msc->thieves may be avoided if
     sc stores a  pointer to the master count.
  */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&msc->thieves.count) > 0);
  KAAPI_ATOMIC_DECR(&msc->thieves.count);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&msc->thieves.count) >= 0);
}
