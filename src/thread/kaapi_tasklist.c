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

int kaapi_tasklist_init( kaapi_tasklist_t* tl, struct kaapi_thread_context_t* thread )
{
  int i;
  
  kaapi_atomic_initlock(&tl->lock);
  KAAPI_ATOMIC_WRITE(&tl->count_thief, 0);

  //kaapi_readytasklist_init( &tl->rtl, (kaapi_taskdescr_t**)thread->stack.sfp->sp, &thread->stack.lock );
  kaapi_readytasklist_init( &tl->rtl, &thread->stack.proc->lock );

  tl->master          = 0;
  tl->thread          = thread;
  tl->recv            = 0;
  tl->context.chkpt   = 0;
#if defined(KAAPI_DEBUG)  
  tl->context.fp      = 0;
  tl->context.td      = 0;
#endif  
  tl->count_recv      = 0;
  kaapi_activationlist_clear( &tl->readylist );
#if defined(KAAPI_DEBUG)
  kaapi_activationlist_clear( &tl->allocated_td );
#endif
  kaapi_recvlist_clear(&tl->recvlist);
  kaapi_allocator_init( &tl->td_allocator );
  kaapi_allocator_init( &tl->allocator );
  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i)
    tl->cnt_tasks[i]  = 0;
  tl->t_infinity      = 0;
#if defined(TASKLIST_ONEGLOBAL_MASTER) && !defined(TASKLIST_REPLY_ONETD)
  KAAPI_ATOMIC_WRITE(&tl->pending_stealop, 0);
#endif
  KAAPI_ATOMIC_WRITE(&tl->cnt_exec, 0);
  tl->total_tasks     = 0;
  return 0;
}


/**/
extern int kaapi_tasklist_destroy( kaapi_tasklist_t* tl )
{
  int i;
  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i)
  {
    kaapi_workqueue_destroy( &tl->rtl.prl[i].wq );
    if (tl->rtl.prl[i].dynallocated)
#if defined(KAAPI_USE_NUMA)
      numa_free( tl->rtl.prl[i].base - tl->rtl.prl[i].size, tl->rtl.prl[i].size );
#else
      free( tl->rtl.prl[i].base - tl->rtl.prl[i].size );
#endif
    tl->rtl.prl[i].base        = 0;
  }
  kaapi_allocator_destroy( &tl->td_allocator );
  kaapi_allocator_destroy( &tl->allocator );
  kaapi_atomic_destroylock(&tl->lock);
  return 0;
}


/*
*/
int kaapi_readytasklist_reserve( 
  kaapi_readytasklist_t* rtl,
  uint64_t               cnt_tasks[KAAPI_TASKLIST_NUM_PRIORITY]
)
{
  int i;  
  size_t capacity = KAAPI_TASKLIST_NUM_PRIORITY*KAAPI_TASKLIST_INITIAL_CAPACITY;
  kaapi_taskdescr_t** base = rtl->staticcontainer;

  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i)
  {
    if (capacity < cnt_tasks[i])
    {
#if defined(KAAPI_USE_NUMA)
      rtl->prl[i].base         = (kaapi_taskdescr_t**)numa_alloc_local( sizeof(kaapi_taskdescr_t*)*cnt_tasks[i]);
#else
      rtl->prl[i].base         = (kaapi_taskdescr_t**)malloc( sizeof(kaapi_taskdescr_t*)*cnt_tasks[i]);
#endif
      rtl->prl[i].base         += cnt_tasks[i];
      rtl->prl[i].dynallocated = 1;
      rtl->prl[i].size         = (int)cnt_tasks[i];
    }
    else if (cnt_tasks[i] >0)
    {
      rtl->prl[i].base         = base+cnt_tasks[i]; /* because index are negative */
      rtl->prl[i].size         = (int)cnt_tasks[i];
      base += cnt_tasks[i];
      capacity -= cnt_tasks[i];
    }
    else 
      rtl->prl[i].base = 0;
  }
  return 0;
}


int _kaapi_readylist_extend_wq( kaapi_onereadytasklist_t* onertl )
{
  kaapi_taskdescr_t** oldbase;
  kaapi_taskdescr_t** newbase;
  int newsize;
  
  oldbase = onertl->base;
  newsize = onertl->size;
  if (newsize ==0) newsize = KAAPI_TASKLIST_INITIAL_CAPACITY;
  else newsize = 2*onertl->size;
  
#if defined(KAAPI_USE_NUMA)
  newbase = (kaapi_taskdescr_t**)numa_alloc_local( sizeof(kaapi_taskdescr_t*) * newsize);
#else
  newbase = (kaapi_taskdescr_t**)malloc( sizeof(kaapi_taskdescr_t*) * newsize);
#endif
  if (newbase ==0) return ENOMEM;
  
#if defined(KAAPI_DEBUG)
  memset( newbase, 0, sizeof(kaapi_taskdescr_t*) * newsize );
#endif

  /* recopy old value */
  if (onertl->size >0)
    memcpy( newbase + onertl->size, 
            oldbase - onertl->size, 
            onertl->size*sizeof(kaapi_taskdescr_t*) 
    );
  newbase += newsize;

#if defined(KAAPI_DEBUG)
{
  long i;
  for (i= onertl->next+1; i<0; ++i)
    kaapi_assert( newbase[i] == oldbase[i] );
}
#endif
  
  /* the only part that need to be linearizable with stealer */
  kaapi_atomic_lock( onertl->wq.lock );
  onertl->base = newbase;
  kaapi_mem_barrier();
  kaapi_atomic_unlock( onertl->wq.lock );  
  
  if (onertl->dynallocated)
  {
    kaapi_assert_debug(onertl->size >0);
#if defined(KAAPI_USE_NUMA)
    numa_free( oldbase-onertl->size, onertl->size );
#else
    free( oldbase - onertl->size);
#endif
  }
  onertl->dynallocated = 1;
  onertl->size = newsize;
  return 0;
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
void kaapi_print_state_tasklist( kaapi_tasklist_t* tl )
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

