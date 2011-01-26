/*
 ** kaapi_wakeup.c
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:18:04 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** christophe.laferriere@imag.fr
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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

kaapi_thread_context_t* kaapi_sched_wakeup ( 
    kaapi_processor_t* kproc, 
    kaapi_processor_id_t kproc_thiefid, 
    kaapi_thread_context_t* cond_thread,
    kaapi_task_t* cond_task 
  )
{
  kaapi_thread_context_t* thread;
  kaapi_wsqueuectxt_cell_t* cell;
    
  /* Note on this line: in the current implementation, only the owner of the suspend queue
     is able to call kaapi_sched_wakeup.
  */
  kaapi_assert_debug( kproc->kid == kproc_thiefid );

  
  /* first test the thread that has been suspended 
     - this thread is initially put into a wsqueue (to be steal)
     - once the thief task is finished, the state of the cell is marked as
     ready and the thread cond_thread remains into the list.
  */ 
  if (cond_thread !=0)
  {
    if ( ((cond_thread->sfp->pc ==cond_task) && kaapi_thread_isready(cond_thread))
      || ((cond_thread->readytasklist !=0) && !kaapi_tasklist_ready_isempty(cond_thread->readytasklist) ) )
//             ((cond_thread->readytasklist->recvlist!=0) 
//          || (KAAPI_ATOMIC_READ(&cond_thread->readytasklist->count_recv) ==0))
//         )
//       )
    {
      /* should be atomic ? */
      cell = cond_thread->wcs;
      if (cell !=0) {
        cell->thread = 0;
        cond_thread->wcs = 0;
        KAAPI_ATOMIC_WRITE(&cell->state, KAAPI_WSQUEUECELL_OUTLIST);
        return cond_thread;
      }
      /* else the thread was put into the lready list */
    }
    /* the cell will be garbaged next times */
  }
  
  /* try to steal ready list */
  if (!kaapi_sched_readyempty(kproc))
  {
    /* lock if self wakeup to protect lready against thieves
       because in that case wakeup is called directly through
       sched_suspend or sched_idle, not by passing through 
       the emission of a request
    */
    kaapi_sched_lock( &kproc->lock );
    thread = kaapi_sched_stealready( kproc, kproc_thiefid );
    kaapi_sched_unlock( &kproc->lock );
    
    if (thread != 0)
    {
      kaapi_assert_debug( kaapi_cpuset_has(thread->affinity, kproc_thiefid)
        || (kaapi_cpuset_empty(thread->affinity) && (kproc->kid == kproc_thiefid)) );
      return thread;
    }
  }
  
  /* else... 
     - here only do garbage, because in the current version the 
     thread that becomes ready is pushed into an other queue.
     Thus, all cell which are marked KAAPI_WSQUEUECELL_OUTLIST or
     KAAPI_WSQUEUECELL_STEALLIST are garbaged.
  */
  cell = kproc->lsuspend.head;
  while (cell != 0)
  {
    /* not already wakeuped */
    int status = KAAPI_ATOMIC_READ(&cell->state);

    /* save the next cell */
    kaapi_wsqueuectxt_cell_t* const nextcell = cell->next;
    thread = cell->thread;
    
    /* add affinity here because in dfg the signal of suspended thread does not move it to ready list */
    if ((thread !=0) && kaapi_cpuset_has(thread->affinity, kproc_thiefid) 
          && kaapi_thread_isready(thread) && (thread == kaapi_wsqueuectxt_steal_cell(cell))
       ) 
//        || ((thread !=0) && !kaapi_tasklist_ready_isempty(thread->readytasklist)) 
    {
      kaapi_wsqueuectxt_finish_steal_cell(cell);
      return thread;
    }

    if (status == KAAPI_WSQUEUECELL_OUTLIST) /* not INLIST nor READY */
    {
      /* delete from the queue */
      if (nextcell != 0)
        nextcell->prev = cell->prev;
      else
        kproc->lsuspend.tail = cell->prev;
      
      if (cell->prev != 0)
        cell->prev->next = nextcell;
      else
        kproc->lsuspend.head = nextcell;
      /* should be atomic ? */
      cell->thread = 0;
      cell->next   = 0;
      cell->prev   = 0;
      
      /* insert it in the recycled queue */
      kaapi_wsqueuectxt_cell_t* const tailfreecell = kproc->lsuspend.tailfreecell;
      if (tailfreecell == 0)
        kproc->lsuspend.headfreecell = cell;
      else 
        tailfreecell->prev = cell;
      kproc->lsuspend.tailfreecell = cell;
    }
  
    /* next */
    cell = nextcell;
  }
  
  return 0; 
}
