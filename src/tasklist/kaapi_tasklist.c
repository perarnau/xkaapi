/*
** kaapi_staticsched.h
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com
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

int kaapi_frame_tasklist_init( kaapi_frame_tasklist_t* tl, struct kaapi_thread_context_t* thread )
{
    tl->recv            = 0;
    tl->count_recv      = 0;
    kaapi_activationlist_clear( &tl->readylist );
#if defined(KAAPI_DEBUG)
    kaapi_activationlist_clear( &tl->allocated_td );
#endif
    kaapi_recvlist_clear(&tl->recvlist);
    kaapi_allocator_init( &tl->td_allocator );
    kaapi_allocator_init( &tl->allocator );
    tl->t_infinity      = 0;
#if !defined(TASKLIST_REPLY_ONETD)
    KAAPI_ATOMIC_WRITE(&tl->pending_stealop, 0);
#endif

    kaapi_tasklist_init(&tl->tasklist, 0);
    tl->tasklist.frame_tasklist = tl;
    return 0;
}


/**/
int kaapi_frame_tasklist_destroy( kaapi_frame_tasklist_t* tl )
{
  kaapi_allocator_destroy( &tl->td_allocator );
  kaapi_allocator_destroy( &tl->allocator );
  kaapi_tasklist_destroy(&tl->tasklist);
  return 0;
}



/** Push a broadcast task attached to a writer task
*/
void kaapi_frame_tasklist_push_broadcasttask( 
    kaapi_frame_tasklist_t*  tl, 
    kaapi_taskdescr_t*       td_writer,
    kaapi_taskdescr_t*       td_bcast
)
{
  kaapi_activationlink_t* al = kaapi_tasklist_allocate_al(tl);
  al->td    = td_bcast;
#if 0
  al->queue = 0;
#endif
  al->next  = 0;
  if (td_writer->u.acl.bcast ==0) 
    td_writer->u.acl.bcast = (kaapi_activationlist_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_activationlist_t));
  if (td_writer->u.acl.bcast->back ==0)
    td_writer->u.acl.bcast->front = td_writer->u.acl.bcast->back = al;
  else {
    td_writer->u.acl.bcast->back->next = al;
    td_writer->u.acl.bcast->back = al;
  }
}



#if defined(KAAPI_DEBUG)
void kaapi_print_state_tasklist( kaapi_frame_tasklist_t* tl )
{
  kaapi_activationlink_t* curr_activated = tl->allocated_td.front;
  const char* str;
  while (curr_activated !=0)
  {
    uint64_t date = 0;
    if (curr_activated->td->u.acl.exec_date !=0)
      date =  curr_activated->td->u.acl.exec_date-kaapi_default_param.startuptime;
    if (curr_activated->td->wc ==0)
      str = "";
    else if ((KAAPI_ATOMIC_READ(&curr_activated->td->counter) % curr_activated->td->wc ==0) && (date ==0) )
      str = " -- ready";
    else 
      str = "";
    printf("td: %p task: %p, counter:%li wc:%li, date:%llu  %s\n", 
        (void*)curr_activated->td, (void*)&curr_activated->td->task, 
        (long)KAAPI_ATOMIC_READ(&curr_activated->td->counter),
        (long)curr_activated->td->wc,
        date, 
        str
    );
    curr_activated = curr_activated->next;
  }
}
#endif

