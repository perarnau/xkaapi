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

kaapi_thread_context_t* kaapi_sched_wakeup ( kaapi_processor_t* kproc )
{
#if defined(KAAPI_VERY_COMPACT_TASK)
#warning TODO
  kaapi_thread_context_t* ctxt;
  kaapi_wsqueuectxt_cell_t* cell;
  
#if 0
  if (0 ==kaapi_wsqueuectxt_pop( &kproc->lready, &ctxt ))
    return ctxt;
#else
  /* only steal the ready context if any */
  if (kproc->ready !=0) {
    ctxt = kproc->ready;
    kproc->ready = 0;
    return ctxt;
  }    
#endif

  cell = kproc->lsuspend.head;
  while (cell !=0)
  {
    ctxt = cell->stack;
    kaapi_task_t*  task = ctxt->pc;
    if ( (KAAPI_ATOMIC_READ(&cell->state) == 0) && 
         (kaapi_task_getstate(task) != KAAPI_TASK_S_STEAL) || 
         (kaapi_task_getbody(task) == kaapi_aftersteal_body) )
    {
      int opok = KAAPI_ATOMIC_CAS( &cell->state, 0, 1);
      if (opok)
      {
        kaapi_wsqueuectxt_cell_t* nextcell = cell->next;
        cell->stack = 0;

        /* whatever is the result of the cas, the cell is recyled (push in tail):
           - cas is true: the pop sucess, we will return the stack
           - cas is false: the stack has been stolen, we recylce the cell
           Then if the cas is ok we return true 
        */
        if (nextcell !=0)
          nextcell->prev = cell->prev;
        if (cell->prev !=0)
          cell->prev->next = nextcell;
        cell->next =0;
        cell->prev = 0;
        kaapi_wsqueuectxt_cell_t* tailfreecell = kproc->lsuspend.tailfreecell;
        if (tailfreecell ==0)
          kproc->lsuspend.headfreecell = cell;
        else 
          tailfreecell->prev = cell;
        kproc->lsuspend.tailfreecell = cell;

        cell->stack = 0;
        return ctxt;
      }
    }

    cell = cell->next;
  }
#endif

  return 0; 
}
