/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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


static inline uint64_t _kaapi_max(uint64_t d1, uint64_t d2)
{ return (d1 < d2 ? d2 : d1); }

static inline kaapi_activationlink_t* _kaapi_alloc_link( 
  kaapi_activationlink_t* freelist, 
  kaapi_allocator_t* allocator
)
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
  l->td   = 0;
  l->next = freelist->next;
  freelist->next = l;
}


/* */
static int kaapi_explore_successor( 
  kaapi_big_hashmap_t*    task2task_khm, 
  kaapi_activationlist_t* rootlist,
  kaapi_activationlink_t* free_al,
  kaapi_allocator_t*      allocator,
  kaapi_taskdescr_t*      td,
  kaapi_taskdescr_t*      td_predgraph 
)
{
  kaapi_taskdescr_t*      tdsucc;
  kaapi_taskdescr_t*      tdsucc_predgraph;
  kaapi_activationlink_t* al;
  int err;
  int visited;


  kaapi_activationlink_t* curr = td->list.front;
  if (curr == 0) 
    return ESRCH;

  while (curr !=0)
  {
    tdsucc = curr->td;
    visited = 1;
    kaapi_hashentries_t* entry = kaapi_big_hashmap_findinsert( task2task_khm, tdsucc );
    if (entry->u.td == 0 )
    {
      tdsucc_predgraph = 
          (kaapi_taskdescr_t*)kaapi_allocator_allocate( allocator, sizeof(kaapi_taskdescr_t) );
      /* store in ->task the original td */
      memset( tdsucc_predgraph, 0, sizeof(kaapi_taskdescr_t) );
      tdsucc_predgraph->task = (kaapi_task_t*)tdsucc;
      entry->u.td = tdsucc_predgraph;
      visited = 0;
    }
    else 
      tdsucc_predgraph = entry->u.td;
    
    /* add link between tdsucc_predgraph -> td_predgraph */
    al = _kaapi_alloc_link( free_al, allocator);
    al->td = td_predgraph;
    kaapi_activationlist_pushback( &tdsucc_predgraph->list, al);
    ++td_predgraph->wc;
  
    if (!visited)
    {
      err = kaapi_explore_successor( task2task_khm, rootlist, free_al, allocator, tdsucc, tdsucc_predgraph );
      if (err == ESRCH) 
      { /* no successor: put it into list of empty task successor */
        al = _kaapi_alloc_link( free_al, allocator);
        al->td = tdsucc_predgraph;
        kaapi_activationlist_pushback( rootlist, al);
      }
    }
    curr = curr->next;
  }
  return 0;
}


/* compute the critical path of each task : length to the final execution execution
*/
int kaapi_staticschedtask_critical_path( kaapi_tasklist_t* tasklist )
{
  kaapi_taskdescr_t*         td;         
  kaapi_big_hashmap_t        task2task_khm;  /* map td -> td in predecessor graph */
  kaapi_activationlist_t     rootlist;
  kaapi_allocator_t          allocator;
  kaapi_activationlink_t     free_al;
  kaapi_taskdescr_t*         td_predgraph;
  kaapi_taskdescr_t*         tdpred_predgraph;
  kaapi_hashentries_t*       entry;
  kaapi_activationlink_t*    al;
  int err;

  if (tasklist == 0)
    return EINVAL;

#if 0
int ctp[KAAPI_MAX_PRIORITY_WQ];
memset(ctp, 0, sizeof(ctp) );    
#endif

  kaapi_allocator_init( &allocator );
  kaapi_activationlist_clear( &rootlist );
  memset(&free_al, 0, sizeof(free_al));
  kaapi_big_hashmap_init( &task2task_khm, 0 );
  
  /* iterate over all tasks and:
     - associated a td in task2task_khm were list points to the list of predecessors
     - populate initial tasks with no sucessor 
  */
  kaapi_activationlink_t* curr = tasklist->readylist.front;
  while (curr !=0)
  {
    td = curr->td;
    entry = kaapi_big_hashmap_findinsert( &task2task_khm, td );
    kaapi_assert( entry->u.td == 0 );
    td_predgraph = 
        (kaapi_taskdescr_t*)kaapi_allocator_allocate( &allocator, sizeof(kaapi_taskdescr_t) );
    entry->u.td = td_predgraph;
    
    /* store in ->task the original td */
    memset( td_predgraph, 0, sizeof(kaapi_taskdescr_t) );
    td_predgraph->task = (kaapi_task_t*)td;

    err = kaapi_explore_successor( &task2task_khm, &rootlist, &free_al, &allocator, td, td_predgraph );
    if (err == ESRCH) 
    { /* no successor: put it into list of empty task successor */
      al = _kaapi_alloc_link(&free_al, &allocator);
      al->td = td_predgraph;
      kaapi_activationlist_pushback(&rootlist, al);
    }
    curr = curr->next;
  }
  
  /* compute the critical path */
  while ( !kaapi_activationlist_isempty( &rootlist ) )
  {
    curr = kaapi_activationlist_popfront( &rootlist );
    td_predgraph = curr->td;
    //_kaapi_free_link( &free_al, curr );
    
    /* process td  */
    if (td_predgraph !=0) 
    {
      td        = (kaapi_taskdescr_t*)td_predgraph->task;
      td->date  = td_predgraph->date;
#if 0
      ++ctp[ kaapi_tasklist_getpriority(td) ];
#endif
//printf("Eval CT task:%p is:%lli\n", (void*)td->task, td->date);

      if (!kaapi_activationlist_isempty(&td_predgraph->list))
      {
//printf("Pred of task:%p are: ", (void*)td->task);
        kaapi_activationlink_t* curr_activated = td_predgraph->list.front;
        while (curr_activated !=0)
        {
          tdpred_predgraph     = curr_activated->td;
          tdpred_predgraph->date = _kaapi_max( tdpred_predgraph->date, 1+td_predgraph->date );
//printf("%p  ct:%i  ", (void*)((kaapi_taskdescr_t*)tdpred_predgraph->task)->task, tdpred_predgraph->date);
          if (kaapi_taskdescr_activated(tdpred_predgraph))
          {
            al = _kaapi_alloc_link(&free_al, &allocator);
            al->td    = tdpred_predgraph;
            kaapi_activationlist_pushback(&rootlist, al);
          }
          curr_activated = curr_activated->next;
        }
//printf("\n\n" );
      }
    }
    
  } /* while */

#if 0
printf("# ctp  #tasks\n");
for (int i=0; i<KAAPI_MAX_PRIORITY_WQ; ++i)
  printf("%i  %i\n", i, ctp[i] );
#endif
  kaapi_allocator_destroy(&allocator );

  return 0;  
}
