/*
** kaapi_hashmap.c
** xkaapi
** 
** 
** Copyright 2010 INRIA.
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
    free(bloc);
  }
  return 0;
}


/** push: LIFO order with respect to pop. Only owner may push
*/
int kaapi_wsqueuectxt_push( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t* stack )
{
  /*printf( "[sleep stack]: stack: @=%p\n", stack);*/

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
    kaapi_wsqueuectxt_cellbloc_t* bloc = malloc( sizeof(kaapi_wsqueuectxt_cellbloc_t) );
    if (bloc ==0) return ENOMEM;
    bloc->next = ls->allocatedbloc;
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

  /* set the state: write stack+set state/barrier/link the cell */
  cell->thread = stack;

  /* in order thief view correct stack pointer if steal the struct */
  kaapi_writemem_barrier();

  KAAPI_ATOMIC_WRITE(&cell->state, 0);

  /* push: LIFO, using next field */
  cell->prev = 0;
  cell->next = ls->head;
  if (cell->next == 0) /* if empty head = tail = 0 */
    ls->tail = cell;
  else
    ls->head->prev = cell;
  ls->head = cell;
  return 0;
}


/** pop: LIFO with respect to push op
*/
int kaapi_wsqueuectxt_pop( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t** stack )
{
  kaapi_wsqueuectxt_cell_t* cell = ls->head;
  while (cell !=0)
  {
    int opok = KAAPI_ATOMIC_CAS( &cell->state, 0, 1);
    if (opok)
    {
      /* R barrier ? */
      *stack = cell->thread;
    }

    kaapi_wsqueuectxt_cell_t* nextcell = cell->next;
    cell->thread = 0;

    /* whatever is the result of the cas, the cell is recyled (push in tail):
       - cas is true: the pop sucess, we will return the stack
       - cas is false: the stack has been stolen, we recylce the cell
       Then if the cas is ok we return true 
    */
    if (nextcell !=0)
      nextcell->prev = 0;
    cell->next =0;
    cell->prev = 0;
    kaapi_wsqueuectxt_cell_t* tailfreecell = ls->tailfreecell;
    if (tailfreecell ==0)
      ls->headfreecell = cell;
    else 
      tailfreecell->prev = cell;
    ls->tailfreecell = cell;

    if (opok) return 0;
    
    cell = nextcell;
  }
  *stack = 0;
  return EWOULDBLOCK;
}

/** steal: FIFO with respect to push op
*/
int kaapi_wsqueuectxt_steal( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t** stack )
{
  kaapi_wsqueuectxt_cell_t* cell = ls->tail;
  while (cell !=0)
  {
    int opok = KAAPI_ATOMIC_CAS( &cell->state, 0, 1);
    if (opok)
    {
      *stack = cell->thread;
      cell->thread = 0;
      return 0;
    }
    cell = cell->prev;
    /* note: here, if the owner has put the cell into free list then the steal  
       may 1/ read incorrect update and iterate through the list of stack or
       2/ read the correct update and will abort quickly because prev is 0
    */
  }
  return EWOULDBLOCK;
}
