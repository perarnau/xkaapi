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
  char* buffer;
#if 0
  printf("%i::[kaapi_threadgroup_save] begin\n", thgrp->localgid);
#endif
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
      ptrdiff_t sz_alloc = thgrp->size_readylists[i] + 2*sizeof(void*) + sizeof(uintptr_t);
      if (i == -1) 
      {
        /* also save sfp, sfp[0] and sfp[-1] */
        sz_alloc += sizeof(kaapi_frame_t)*2 + sizeof(kaapi_frame_t*);
      }

      thgrp->save_readylists[i] = buffer = malloc( sz_alloc );
      
      if (i == -1)
      {
        memcpy( buffer, (const void*)&thgrp->threadctxts[-1]->sfp, sizeof(kaapi_frame_t*) );
        buffer += sizeof(kaapi_frame_t*);
        kaapi_assert_debug( (buffer - thgrp->save_readylists[i]) < sz_alloc );

        memcpy( buffer, &thgrp->threadctxts[-1]->sfp[-1], 2*sizeof(kaapi_frame_t) );
        buffer += 2*sizeof(kaapi_frame_t);
        kaapi_assert_debug( (buffer - thgrp->save_readylists[i]) < sz_alloc );
      }
      
      memcpy( buffer, tasklist->stack, thgrp->size_readylists[i] );
      buffer += thgrp->size_readylists[i];
      kaapi_assert_debug( (buffer - thgrp->save_readylists[i]) < sz_alloc );
      
      /* save head and tail of the list */
      memcpy( buffer,  &tasklist->front, 2*sizeof(void*) );
      buffer += 2*sizeof(void*);
      kaapi_assert_debug( (buffer - thgrp->save_readylists[i]) < sz_alloc );

      /* save the count recv */
      memcpy( buffer, &tasklist->count_recv, sizeof(uintptr_t) );
      buffer += sizeof(uintptr_t);
      kaapi_assert_debug( (buffer - thgrp->save_readylists[i]) == sz_alloc );
    }
    else {
      thgrp->size_readylists[i] = 0;
      thgrp->save_readylists[i] = 0;
    }
  }
#if 0
  printf("%i::[kaapi_threadgroup_save] end\n", thgrp->localgid);
#endif
  return 0;
}


/**
*/
int kaapi_threadgroup_restore_thread( kaapi_threadgroup_t thgrp, int tid )
{
  char* buffer;
  kaapi_tasklist_t* tasklist;
  
#if 0
  printf("%i::[kaapi_threadgroup_restore_thread] tid:%i begin\n", thgrp->localgid, tid);
#endif
  kaapi_assert( (tid >=-1) && (tid < thgrp->group_size) );
  kaapi_assert_debug(thgrp->localgid == thgrp->tid2gid[tid]);

  buffer = thgrp->save_readylists[tid];
  if (tid == -1)
  {
    memcpy( (void*)&thgrp->threadctxts[-1]->sfp, buffer, sizeof(kaapi_frame_t*) );
    buffer += sizeof(kaapi_frame_t*);
    memcpy( &thgrp->threadctxts[-1]->sfp[-1], buffer, 2*sizeof(kaapi_frame_t) );
    buffer += 2*sizeof(kaapi_frame_t);

    /* do not forget to reset the waittask with correct state */
    kaapi_task_init_with_state( thgrp->waittask, kaapi_taskwaitend_body, KAAPI_MASK_BODY_STEAL, thgrp );
  }
  
  tasklist = thgrp->threadctxts[tid]->sfp->tasklist;
  memcpy( tasklist->stack, buffer, thgrp->size_readylists[tid] );
  buffer += thgrp->size_readylists[tid];

  /* restore head and tail of the list */
  memcpy( &tasklist->front, buffer, 2*sizeof(void*) );
  buffer += 2*sizeof(void*);

  memcpy( &tasklist->count_recv, buffer, sizeof(uintptr_t) );
  buffer += sizeof(uintptr_t);

#if 0
  printf("%i::[kaapi_threadgroup_restore_thread] tid:%i end\n", thgrp->localgid, tid);
#endif
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
