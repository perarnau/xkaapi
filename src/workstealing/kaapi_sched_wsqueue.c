/*
** xkaapi
** 
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
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


/**
*/
int kaapi_wsqueuectxt_init( kaapi_wsqueuectxt_t* ls )
{
  ls->head          = 0;
  ls->tail          = 0;
  ls->headfreecell  = 0;
  ls->tailfreecell  = 0;
  ls->allocatedbloc = 0;
  return 0;
}


/**
*/
int kaapi_wsqueuectxt_destroy( kaapi_wsqueuectxt_t* ls )
{
  kaapi_wsqueuectxt_cellbloc_t* bloc;
  ls->head          = 0;
  ls->tail          = 0;
  while (ls->allocatedbloc !=0)
  {
    bloc = ls->allocatedbloc;
    ls->allocatedbloc = bloc->next;
    free(bloc->ptr);
  }
  return 0;
}


/** Allocate a new cell
*/
static kaapi_wsqueuectxt_cell_t* kaapi_wsqueuectxt_alloccell( kaapi_wsqueuectxt_t* ls )
{
  kaapi_wsqueuectxt_cell_t* cell = ls->headfreecell;
  if (cell !=0)
  {
    /* use ->prev for linking in free list, see steal */
    ls->headfreecell = cell->prev;
    if (ls->tailfreecell == cell)
      ls->tailfreecell = 0;
  }
  else /* allocate a new bloc */
  {
    int i;
    void* ptr;
    kaapi_wsqueuectxt_cellbloc_t* bloc = 
        (kaapi_wsqueuectxt_cellbloc_t*)kaapi_malloc_align( KAAPI_CACHE_LINE, sizeof(kaapi_wsqueuectxt_cellbloc_t), &ptr );
    if (bloc ==0) return 0;
    bloc->ptr         = ptr;
    bloc->next        = ls->allocatedbloc;
    ls->allocatedbloc = bloc;
    for (i=1; i<KAAPI_BLOCENTRIES_SIZE; ++i)
    {
      bloc->data[i].thread = 0;
      bloc->data[i].next  = 0;
      bloc->data[i].prev  = &bloc->data[i-1];
    }
    bloc->data[1].prev = 0;
    ls->headfreecell = &bloc->data[KAAPI_BLOCENTRIES_SIZE-1];
    ls->tailfreecell = &bloc->data[1];
    cell = &bloc->data[0];
  }
  return cell;
}



/** push: LIFO order with respect to pop. Only owner may push
*/
int kaapi_wsqueuectxt_push( kaapi_processor_t* kproc, kaapi_thread_context_t* thread )
{
  kaapi_wsqueuectxt_t* ls = &kproc->lsuspend;
  kaapi_wsqueuectxt_cell_t* cell = kaapi_wsqueuectxt_alloccell(ls);
#if defined(KAAPI_USE_STATICSCHED)  
  kaapi_task_t* task = thread->sfp->pc;
  kaapi_task_body_t task_body = kaapi_task_getbody(task);
  kaapi_taskrecv_arg_t* argrecv;
#endif
  kaapi_cpuset_copy(&cell->affinity, &thread->affinity);
  cell->thread   = 0;

  cell->thread   = thread;
  thread->wcs    = cell;
#if defined(KAAPI_USE_STATICSCHED)
  if ( (task_body == kaapi_taskrecv_body) || (task_body == kaapi_taskrecvbcast_body) || (task_body == kaapi_taskbcast_body) )
  {
    argrecv = (kaapi_taskrecv_arg_t*)kaapi_task_getargs(task);
    argrecv->wcs = cell;
  }
#endif  

  /* barrier + write in order thief view correct thread pointer if steal the struct */
  KAAPI_ATOMIC_WRITE_BARRIER(&cell->state, KAAPI_WSQUEUECELL_INLIST);

  /* push: LIFO, using next field */
  cell->prev = 0;
  cell->next = ls->head;

  /* avoid reordering of previous field... */
  kaapi_writemem_barrier();
  if (cell->next == 0) /* if empty head = tail = 0 */
    ls->tail = cell;
  else
    ls->head->prev = cell;
  ls->head = cell;
  return 0;
}


/** steal: FIFO with respect to push op
*/
kaapi_thread_context_t* kaapi_wsqueuectxt_steal_cell( kaapi_wsqueuectxt_cell_t* cell )
{
//  kaapi_wsqueuectxt_t* ls = &kproc->lsuspend;
  kaapi_thread_context_t* thread = 0;

  int opok = !kaapi_cpuset_empty(cell->affinity) && KAAPI_ATOMIC_CAS( &cell->state, KAAPI_WSQUEUECELL_INLIST, KAAPI_WSQUEUECELL_STEALLIST);
  if (opok)
  {
    thread       = cell->thread;
    cell->thread = 0;
    thread->wcs  = 0;
    return thread;
  }
  return 0;
}
