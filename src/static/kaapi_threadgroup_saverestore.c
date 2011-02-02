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


/**
*/
int kaapi_threadgroup_save(kaapi_threadgroup_t thgrp )
{
  printf("%i::[kaapi_threadgroup_save] begin\n", thgrp->localgid);
  if (thgrp->state != KAAPI_THREAD_GROUP_PARTITION_S) return EINVAL;

  if (thgrp->save_readylists ==0)
  {
    thgrp->save_readylists = (char**)malloc( (1+thgrp->group_size) * sizeof(char*) );
    thgrp->size_readylists = (size_t*)malloc( (1+thgrp->group_size) * sizeof(size_t) );
    ++thgrp->save_readylists;
    ++thgrp->size_readylists;
  }
  
  for (int i=-1; i<thgrp->group_size; ++i)
  {
    if (thgrp->localgid == thgrp->tid2gid[i])
    {
      kaapi_tasklist_t* tasklist = thgrp->threadctxts[i]->sfp->tasklist; 
      thgrp->size_readylists[i] = tasklist->sp;
      thgrp->save_readylists[i] = malloc( thgrp->size_readylists[i] + 2*sizeof(void*) + sizeof(uintptr_t) );
      memcpy( thgrp->save_readylists[i], 
              tasklist->stack,
              thgrp->size_readylists[i] 
      );
      
      /* save head and tail of the list */
      memcpy( thgrp->save_readylists[i] + thgrp->size_readylists[i], 
              &tasklist->front,
              2*sizeof(void*)
      );

      /* save the count recv */
      memcpy( thgrp->save_readylists[i] + thgrp->size_readylists[i]+2*sizeof(void*), 
              &tasklist->count_recv,
              sizeof(uintptr_t)
      );
    }
    else {
      thgrp->save_readylists[i] = 0;
    }
  }
  printf("%i::[kaapi_threadgroup_save] end\n", thgrp->localgid);
  return 0;
}


/**
*/
int kaapi_threadgroup_restore_thread( kaapi_threadgroup_t thgrp, int tid )
{
  printf("%i::[kaapi_threadgroup_restore_thread] tid:%i begin\n", thgrp->localgid, tid);
  kaapi_assert( (tid >=-1) && (tid < thgrp->group_size) );
  kaapi_assert_debug(thgrp->localgid == thgrp->tid2gid[tid]);

  kaapi_tasklist_t* tasklist = thgrp->threadctxts[tid]->sfp->tasklist;
  memcpy( tasklist->stack,
          thgrp->save_readylists[tid], 
          thgrp->size_readylists[tid] 
  );

  /* restore head and tail of the list */
  memcpy( &tasklist->front,
          thgrp->save_readylists[tid] + thgrp->size_readylists[tid],
          2*sizeof(void*)
  );

  memcpy( &tasklist->count_recv,
          thgrp->save_readylists[tid] + thgrp->size_readylists[tid] + 2*sizeof(void*),
          sizeof(uintptr_t)
  );

  tasklist->sp = thgrp->size_readylists[tid];
  tasklist->recvlist = 0;
  printf("%i::[kaapi_threadgroup_restore_thread] tid:%i end\n", thgrp->localgid, tid);
  return 0;
}


/**
*/
int kaapi_threadgroup_restore(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_WAIT_S) return EINVAL;

  /* do the same for each worker */
  for (int i=-1; i<thgrp->group_size; ++i)
  {
    if (thgrp->localgid == thgrp->tid2gid[i])
      kaapi_threadgroup_restore_thread(thgrp, i);
  }
  thgrp->state = KAAPI_THREAD_GROUP_MP_S;
  return 0;
}
