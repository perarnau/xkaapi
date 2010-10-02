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
  ( kaapi_processor_t* kproc, kaapi_processor_id_t kproc_thiefid, struct kaapi_thread_context_t* cond_thread )
{
  kaapi_thread_context_t* ctxt = 0;
  kaapi_wsqueuectxt_cell_t* cell;
  int wakeupok = 0;
  int garbage;

  /* only steal the ready context if it's my ready list */
  if ((kproc->readythread !=0) && (kproc->kid == kproc_thiefid))
  {
    ctxt = kproc->readythread;
    if ((ctxt->affinity !=0) || (cond_thread ==0) || (ctxt == cond_thread)) 
    {
      kproc->readythread = 0;
#if defined(KAAPI_DEBUG_PRINT)
      printf("%i:: get thread:%p from ready thread\n", kproc->kid, (void*)ctxt);
      fflush( stdout );
#endif  
      return ctxt;
    }
  }
  
  /* ready list not empty */
  if (!kaapi_sched_isreadyempty(kproc))
    {
    kaapi_thread_context_t* thread = 0;

    /* lock if self wakeup to protect lready against thieves because in that case
       wakeup is called directly through sched_suspend or sched_idle, not by passing through 
       the emission of a request
    */
    kaapi_sched_lock( kproc );
    thread = kaapi_sched_stealready( kproc, kproc_thiefid );
    kaapi_sched_unlock( kproc );
    if (thread !=0) return thread;
  }

  cell = kproc->lsuspend.head;
  while (cell !=0)
  {
    int status = KAAPI_ATOMIC_READ( &cell->state );
    if (status != 2 ) /* else already wakeuped */
    {
      ctxt = cell->thread;
      if ( (ctxt->affinity !=0) || (cond_thread ==0) || (ctxt == cond_thread)) 
      {      
        kaapi_task_t* task = ctxt->sfp->pc;
        if ( (kaapi_task_getbody(task) != kaapi_suspend_body) )
        { 
          garbage  = 1;
          /* ok wakeup the thread and try to steal it */
          while (status !=2)
          {
            if (KAAPI_ATOMIC_CAS( &cell->state, status, 2 )) /* if already ==1 -> under stealing */
            {
              cell->thread = 0;
              wakeupok = 1;
              status = 2;
            }
            else status = KAAPI_ATOMIC_READ( &cell->state );
          }
          /* if wakeupok the caller has wakeup the thread */
        }
        else {
          wakeupok = 0;
          garbage  = 0;
        }
      } /* else ctxt->affinity: */
      else {
        wakeupok = 0;
        garbage  = 0;
      }
    } 
    else 
    { /* else state = 2: already wakeuped by someone */
      wakeupok = 0;
      garbage  = 1;
    }
    
    /* If the wakeup is ok or if the cell state is 1, the cell is recyled (push in tail):
    */
    kaapi_wsqueuectxt_cell_t* nextcell = cell->next;
    if (garbage)
    {
      kaapi_assert_debug(cell->thread ==0); 
      /* delete from the queue */
      if (nextcell !=0)
        nextcell->prev = cell->prev;
      else
        kproc->lsuspend.tail = cell->prev;
        
      if (cell->prev !=0)
        cell->prev->next = nextcell;
      else
        kproc->lsuspend.head = nextcell;
      cell->next =0;
      cell->prev = 0;

      /* insert it in the recycled queue */
      kaapi_wsqueuectxt_cell_t* tailfreecell = kproc->lsuspend.tailfreecell;
      if (tailfreecell ==0)
        kproc->lsuspend.headfreecell = cell;
      else 
        tailfreecell->prev = cell;
      kproc->lsuspend.tailfreecell = cell;

      if (wakeupok) 
      {
#if defined(KAAPI_DEBUG_PRINT)
        printf("%i:: wakeup thread:%p from suspend list\n", kproc->kid, (void*)ctxt);
        fflush( stdout );
#endif  
        return ctxt;
      }
    }

    cell = nextcell;
  }

  return 0; 
}
