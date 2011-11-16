/*
** xkaapi
** 
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
#include "kaapi_tasklist.h"

static inline kaapi_activationlink_t* _kaapi_alloc_link( kaapi_activationlink_t* freelist, kaapi_allocator_t* allocator)
{
  kaapi_activationlink_t* retval;
  if (freelist->next !=0)
  {
    retval = freelist->next;
    freelist->next = retval->next;
  }
  else 
  {
    retval = (kaapi_activationlink_t*)kaapi_allocator_allocate( allocator, sizeof(kaapi_activationlink_t) );
  }
  return retval;
}

static inline void _kaapi_free_link( kaapi_activationlink_t* freelist, kaapi_activationlink_t* l)
{
  l->next = freelist->next;
  freelist->next = l;
}

/** kaapi_threadgroup_execframe
    Use the list of ready task to execute program in an abstract sens.
    The function follows the readylist execution to iterates through the task
    in order to call 'taskdescr_execute' function onto each task descriptor.
    This method is used to interpret program execution and it is used
    in order to print the task (see dfgdot_print).
    
    If no more task exists in the readylist but their are tasks that are waiting
    for incomming synchronisation (such as in the recvlist), the abstract execution
    consider that they are ready.
*/
int kaapi_thread_abstractexec_readylist( 
  const kaapi_tasklist_t* tasklist, 
  void (*taskdescr_executor)(kaapi_taskdescr_t*, void*),
  void* arg_executor
)
{
  kaapi_taskdescr_t*         td;      /* cache */
  kaapi_recvactlink_t*       recv;    /* current recv task to consider */
  kaapi_activationlist_t     readylist;
  kaapi_allocator_t          allocator;
  kaapi_activationlink_t     free_al;

  if ((tasklist == 0) || (taskdescr_executor ==0)) 
    return EINVAL;
  
  kaapi_allocator_init( &allocator );
  kaapi_activationlist_clear( &readylist );
  free_al.next = 0;
  recv = tasklist->recvlist.front;

  /* populate initial ready td into readylist */
  kaapi_activationlink_t* curr = tasklist->readylist.front;
  while (curr !=0)
  {
    kaapi_activationlink_t* al = _kaapi_alloc_link(&free_al, &allocator);
    al->td = curr->td;
    al->queue = 0;
    kaapi_activationlist_pushback(&readylist, al);
    curr = curr->next;
  }

redo_once:
  while ( !kaapi_activationlist_isempty( &readylist ) )
  {
    curr = kaapi_activationlist_popfront( &readylist );
    td = curr->td;
    _kaapi_free_link( &free_al, curr );
    
    /* execute td->task */
    if (td !=0) 
    {
      taskdescr_executor( td, arg_executor );

      /* push in the front the activated tasks */
      if (!kaapi_activationlist_isempty(&td->u.acl.list))
      {
        kaapi_activationlink_t* curr_activated = td->u.acl.list.front;
        while (curr_activated !=0)
        {
          if (kaapi_taskdescr_activated(curr_activated->td))
          {
            kaapi_activationlink_t* al = _kaapi_alloc_link(&free_al, &allocator);
            al->td    = curr_activated->td;
            al->queue = 0;
            kaapi_activationlist_pushback(&readylist, al);
          }
          curr_activated = curr_activated->next;
        }
      }

      /* do bcast after child execution (they can produce output data) */
      if (td->u.acl.bcast !=0) 
      {
        kaapi_activationlink_t* curr_activated = td->u.acl.bcast->front;
        while (curr_activated !=0)
        {
          //always ready task.... after td if (kaapi_taskdescr_activated(curr_activated->td))
          {
            kaapi_activationlink_t* al = _kaapi_alloc_link(&free_al, &allocator);
            al->td    = curr_activated->td;
            al->queue = 0;
            kaapi_activationlist_pushback(&readylist, al);
          }
          curr_activated = curr_activated->next;
        }
      }
      else continue;
    }
    
  } /* while */

  /* pseudo recv incomming synchronisation 
  */
  if (recv !=0)
  {
    kaapi_activationlink_t* al = _kaapi_alloc_link(&free_al, &allocator);
    al->queue = 0;
    al->td    = recv->td;
    kaapi_activationlist_pushback(&readylist, al);
    recv = recv->next;
    goto redo_once;
  }
  

  kaapi_allocator_destroy(&allocator );
  
  return 0;
}
