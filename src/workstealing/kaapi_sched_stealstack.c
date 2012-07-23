/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributor :
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


/*
 */
void kaapi_synchronize_steal( kaapi_processor_t* kproc )
{
  kaapi_atomic_waitlock(&kproc->lock);
}

void kaapi_synchronize_steal_thread( kaapi_thread_context_t* thread )
{
  kaapi_atomic_waitlock(&thread->stack.lock);
}


/** Steal task in the stack from the bottom to the top.
 This signature MUST BE the same as a splitter function.
 */
int kaapi_sched_stealstack  
( 
 kaapi_thread_context_t*       thread, 
 kaapi_listrequest_t*          lrequests, 
 kaapi_listrequest_iterator_t* lrrange
 )
{
  kaapi_frame_t*           top_frame;  
  kaapi_hashmap_t          access_to_gd;
  kaapi_hashentries_bloc_t stackbloc;
  
  if ((thread ==0) || (thread->unstealable != 0)) 
    return 0;
  
  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &access_to_gd, &stackbloc );
  
  /* may be done by atomic write, see kaapi_thread_execframe */
  kaapi_atomic_lock(&thread->stack.lock);
  
  /* try to steal in each frame */
  for (  top_frame = thread->stack.stackframe; 
       (top_frame <= thread->stack.sfp) && !kaapi_listrequest_iterator_empty(lrrange); 
       ++top_frame)
  {
    /* TODO here: virtualization of the frame properties ? */
    if (top_frame->tasklist == 0)
    {
      thread->stack.thieffp = top_frame;
      if (top_frame->pc == top_frame->sp) continue;
      kaapi_sched_stealframe( thread, top_frame, &access_to_gd, lrequests, lrrange );
    } else 
      kaapi_sched_stealtasklist( thread, top_frame->tasklist, lrequests, lrrange );
  }
  
  thread->stack.thieffp = 0;
  
  kaapi_atomic_unlock(&thread->stack.lock);
  
  kaapi_hashmap_destroy( &access_to_gd );
  
  return 0;
}
