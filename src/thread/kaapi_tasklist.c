/*
** kaapi_staticsched.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
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

/**
*/
kaapi_version_t* kaapi_thread_newversion( 
    kaapi_metadata_info_t* kmdi, 
    kaapi_address_space_id_t kasid,
    void* data, const kaapi_memory_view_t* view 
)
{
#if 0//defined(KAAPI_USE_NUMA)
  kaapi_version_t* version = (kaapi_version_t*)numa_alloc_local( sizeof(kaapi_version_t) );
  version->handle       = (kaapi_data_t*)numa_alloc_local(sizeof(kaapi_data_t));
#else
  kaapi_version_t* version = (kaapi_version_t*)malloc( sizeof(kaapi_version_t) );
  version->handle       = (kaapi_data_t*)malloc(sizeof(kaapi_data_t));
#endif
  version->orig         = _kaapi_metadata_info_get_data( kmdi, kasid);
  version->handle->ptr  = kaapi_make_nullpointer(); /* or data.... if no move task is pushed */
  version->handle->view = *view;
  version->tag          = 0;
  version->last_mode    = KAAPI_ACCESS_MODE_VOID;
  version->last_task    = 0;
  version->last_tasklist= 0;
  version->writer_task  = 0;
  version->writer_asid  = kasid;
  version->writer_tasklist = 0;
#if defined(KAAPI_DEBUG)
  version->next         = 0;
#endif  
  return version;
}


/**
*/
kaapi_version_t* kaapi_thread_copyversion( 
    kaapi_metadata_info_t* kmdi, 
    kaapi_address_space_id_t kasid,
    kaapi_version_t* src
)
{
  kaapi_version_t* version = (kaapi_version_t*)malloc( sizeof(kaapi_version_t) );
  version->orig         = _kaapi_metadata_info_get_data( kmdi, kasid );
  version->handle       = (kaapi_data_t*)malloc(sizeof(kaapi_data_t));
  version->handle->ptr  = kaapi_make_nullpointer(); /* or data.... if no move task is pushed */
  version->handle->view = src->orig->view;
  version->tag          = src->tag;
  version->last_mode    = KAAPI_ACCESS_MODE_VOID;
  version->last_task    = 0;
  version->last_tasklist = src->last_tasklist;
  version->writer_task  = 0;
  version->writer_asid  = kasid;
  version->writer_tasklist = 0;

  /* link copy in from of mdi */
  version->next         = 0;
  src->next             = version;
  
  return version;
}


/* activate and push all ready tasks in the activation list to their allocated queue
*/
int kaapi_tasklist_doactivationlist( kaapi_activationlist_t* al )
{
  kaapi_activationlink_t* curr = al->front;
  while (curr !=0)
  {
    if (KAAPI_ATOMIC_INCR(&curr->td->counter) % curr->td->wc == 0)
    {
      kaapi_assert (0); //curr->queue != 0)
    }
    curr = curr->next;
  }

  return 0;
}



/** Push a broadcast task attached to a writer task
*/
void kaapi_tasklist_push_broadcasttask( 
    kaapi_tasklist_t*  tl, 
    kaapi_taskdescr_t* td_writer,
    kaapi_taskdescr_t* td_bcast
)
{
  kaapi_activationlink_t* al = kaapi_tasklist_allocate_al(tl);
  al->td    = td_bcast;
  al->queue = 0;
  al->next  = 0;
  if (td_writer->bcast ==0) 
    td_writer->bcast = (kaapi_activationlist_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_activationlist_t));
  if (td_writer->bcast->back ==0)
    td_writer->bcast->front = td_writer->bcast->back = al;
  else {
    td_writer->bcast->back->next = al;
    td_writer->bcast->back = al;
  }
}



#if defined(KAAPI_DEBUG)
void kaapi_print_state_tasklist( kaapi_tasklist_t* tl )
{
  kaapi_activationlink_t* curr_activated = tl->allocated_td.front;
  const char* str;
  while (curr_activated !=0)
  {
    uint64_t date = 0;
    if (curr_activated->td->exec_date !=0)
      date =  curr_activated->td->exec_date-kaapi_default_param.startuptime;
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

