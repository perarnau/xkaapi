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

kaapi_thread_context_t* kaapi_sched_wakeup
(
 kaapi_processor_t* kproc,
 kaapi_processor_id_t kproc_thiefid,
 kaapi_thread_context_t* cond_thread
)
{
  kaapi_thread_context_t* ctxt = NULL;
  kaapi_wsqueuectxt_cell_t* cell;
  int wakeupok = 0;
  int garbage;
  
  /* only steal the ready context if it's my ready list */
  if ((kproc->readythread !=0) && (kproc->kid == kproc_thiefid))
  {
    ctxt = kproc->readythread;
    if ((ctxt->affinity !=0) || (cond_thread ==0) || (ctxt == cond_thread)) 
    {
      kproc->readythread = NULL;
      return ctxt;
    }
  }
  
  /* ready list not empty */
  if (!kaapi_sched_isreadyempty(kproc))
  {
    /* lock if self wakeup to protect lready against thieves
       because in that case wakeup is called directly through
       sched_suspend or sched_idle, not by passing through 
       the emission of a request
    */

    kaapi_thread_context_t* thread;

    kaapi_sched_lock( kproc );
    thread = kaapi_sched_stealready( kproc, kproc_thiefid );
    kaapi_sched_unlock( kproc );

    if (thread != NULL)
      return thread;
  }

  cell = kproc->lsuspend.head;
  while (cell != NULL)
  {
    /* assume  will garbage */
    garbage = 1;

    /* not already wakeuped */
    int status = KAAPI_ATOMIC_READ(&cell->state);
    if (status != 2)
    {
      /* assume wont garbage */
      garbage = 0;

      ctxt = cell->thread;
      if ((ctxt->affinity !=0) || (cond_thread ==0) || (ctxt == cond_thread)) 
      {      
        kaapi_task_t* task = ctxt->sfp->pc;
        if ((kaapi_task_getbody(task) != kaapi_suspend_body))
        { 
          garbage = 1;
          /* wakeup the thread and try to steal it */
          while (status != 2)
          {
            if (KAAPI_ATOMIC_CAS(&cell->state, status, 2))
            {
	      /* wakeup success */
              cell->thread = 0;
              wakeupok = 1;
	      break ;
            }

	    status = KAAPI_ATOMIC_READ(&cell->state);
          }
        }
      }
    }

    /* save the next cell */
    kaapi_wsqueuectxt_cell_t* const nextcell = cell->next;

    /* recycle the cell */
    if (garbage)
    {
      kaapi_assert_debug(cell->thread == NULL); 

      /* delete from the queue */
      if (nextcell != NULL)
        nextcell->prev = cell->prev;
      else
        kproc->lsuspend.tail = cell->prev;
        
      if (cell->prev != NULL)
        cell->prev->next = nextcell;
      else
        kproc->lsuspend.head = nextcell;
      cell->next = NULL;
      cell->prev = NULL;

      /* insert it in the recycled queue */
      kaapi_wsqueuectxt_cell_t* const tailfreecell =
	kproc->lsuspend.tailfreecell;
      if (tailfreecell == NULL)
        kproc->lsuspend.headfreecell = cell;
      else 
        tailfreecell->prev = cell;
      kproc->lsuspend.tailfreecell = cell;

      if (wakeupok) 
        return ctxt;
    }

    /* next */
    cell = nextcell;
  }

  return 0; 
}
