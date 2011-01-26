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
** threadctxts.
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
#include "kaapi_network.h"

/** Collective operation:
    - the threads are mapped with round robin mapping among the processors
    - the thread i is on processor i % #procs.
    - the master thread is on the globalid 0.
*/
int kaapi_threadgroup_create(kaapi_threadgroup_t* pthgrp, int size )
{
  int i;
  int error = 0;
  kaapi_threadgroup_t thgrp = 0;
  kaapi_processor_t*   proc = 0;

  kaapi_globalid_t     mygid;
  uint32_t             nodecount;
  
  if (pthgrp ==0) return EINVAL;
  thgrp = (kaapi_threadgroup_t)malloc(sizeof(kaapi_threadgrouprep_t));
  kaapi_assert(thgrp !=0);
  
  /* */
  proc  = kaapi_get_current_processor();
  mygid = kaapi_network_get_current_globalid();
  nodecount = kaapi_network_get_count();
  thgrp->localgid = mygid;
  thgrp->nodecount = nodecount;

  thgrp->group_size  = size;
  thgrp->startflag   = 0;
  KAAPI_ATOMIC_WRITE(&thgrp->countend, 0);
  thgrp->waittask    = 0;
  thgrp->threadctxts = malloc( (1+size)* sizeof(kaapi_thread_context_t*) );
  kaapi_assert(thgrp->threadctxts !=0);

  thgrp->threadctxts[0] = proc->thread;
  thgrp->threadctxts[0]->partid = -1;
  
  thgrp->threads    = malloc( (1+size)* sizeof(kaapi_thread_t*) );
  kaapi_assert(thgrp->threads !=0);
  
  /* create mapping 
  */
  thgrp->tid2gid  = (kaapi_globalid_t*)malloc( (1+size) * sizeof(kaapi_globalid_t) );
  kaapi_assert(thgrp->tid2gid !=0);
  thgrp->tid2asid = (kaapi_address_space_t*)malloc( (1+size) * sizeof(kaapi_address_space_t) );
  kaapi_assert(thgrp->tid2asid !=0);

  /* shift +1, -1 == main thread */
  ++thgrp->threads;
  ++thgrp->tid2gid;     /* shift such that -1 == index 0 of allocate array */
  ++thgrp->tid2asid;    /* shift such that -1 == index 0 of allocate array */
  ++thgrp->threadctxts; /* shift, because -1 ==> main thread */
  
  /* map threads onto globalid */
  for (i=-1; i<size; ++i)
  {
    /* map thread i on processor node count */
    thgrp->tid2gid[i]  = (kaapi_globalid_t)( (1+i) % nodecount);
    /* assigned address space identifier for thread i */
    thgrp->tid2asid[i] = kaapi_threadgroup_tid2asid( thgrp, i);
    //(uint32_t)(thgrp->tid2gid[i] << 16) | (uint32_t)((1+i) / nodecount);
    printf("Tid: %i as asid: %i\n", i, thgrp->tid2asid[i] );
  }

  /* here allocate thread -1 == main thread */
  if (mygid == thgrp->tid2gid[-1])
  {
    thgrp->threadctxts[-1] = kaapi_get_current_processor()->thread;
    thgrp->threads[-1] = kaapi_threadcontext2thread(thgrp->threadctxts[-1]);
    kaapi_threadgroup_initthread( thgrp, -1 );
  }

  /* here may be dispatch allocation of all processors */
  for (i=0; i<size; ++i)
  {
    if (mygid == thgrp->tid2gid[i]) 
    {
      thgrp->threadctxts[i] = kaapi_context_alloc( proc );
      kaapi_assert(thgrp->threadctxts[i] != 0);
      
      /* init the thread from the thread context */
      thgrp->threadctxts[i]->partid = i;
      thgrp->threads[i] = kaapi_threadcontext2thread(thgrp->threadctxts[i]);
      kaapi_threadgroup_initthread( thgrp, i );
    }
    else 
      thgrp->threadctxts[i] = 0;
  }
  
  error =pthread_mutex_init(&thgrp->mutex, 0);
  kaapi_assert(error ==0);

  error =pthread_cond_init(&thgrp->cond, 0);
  kaapi_assert (error ==0);

  /* ok */
  thgrp->maxstep            = -1;
  thgrp->step               = -1;
  thgrp->state              = KAAPI_THREAD_GROUP_CREATE_S;
  thgrp->flag               = 0;
  thgrp->tag_count          = 0;
  kaapi_assert( kaapi_allocator_init(&thgrp->allocator) ==0);
  thgrp->free_dataversion_list=0;
  thgrp->save_readylists    = 0;
  thgrp->size_readylists    = 0;
  *pthgrp                   = thgrp;
  return 0;
}


/**
*/
int kaapi_threadgroup_set_iteration_step(kaapi_threadgroup_t thgrp, int maxstep )
{
  thgrp->maxstep = maxstep;
  return 0;
}

