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
int kaapi_threadgroup_begin_partition(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_CREATE_S) return EINVAL;
  thgrp->state = KAAPI_THREAD_GROUP_PARTITION_S;
  thgrp->mainctxt   = kaapi_get_current_processor()->thread;
  thgrp->threads[-1]= kaapi_threadcontext2thread(thgrp->mainctxt);
  
  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &thgrp->ws_khm, 0 );
  kaapi_vector_init( &thgrp->ws_vect_input, 0 );
  kaapi_versionallocator_init( &thgrp->ver_allocator );
  
  /* same the main thread frame to restore it at the end of parallel computation */
  kaapi_thread_save_frame(thgrp->threads[-1], &thgrp->mainframe);
  
  /* avoid thief to steal the main thread will tasks are added */
  thgrp->mainctxt->unstealable = 1;
  kaapi_mem_barrier();
  
  /* wait thief get out the thread */
#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
  while (!KAAPI_ATOMIC_CAS(&thgrp->mainctxt->lock, 0, 1))
    ;
#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
  while (thgrp->mainctxt->thieffp != 0)
    ;
#endif
  
#if 0
  fprintf(stdout, "Save frame:: pc:%p, sp:%p, spd:%p\n", 
    (void*)thgrp->mainframe.pc, 
    (void*)thgrp->mainframe.sp, 
    (void*)thgrp->mainframe.sp_data 
  );
#endif

  return 0;
}


/**
*/
int kaapi_threadgroup_end_partition(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_PARTITION_S) 
    return EINVAL;
  kaapi_task_t* task;
  
  /* for all threads add a signalend task */
  for (int i=0; i<thgrp->group_size; ++i)
  {
    task = kaapi_thread_toptask( thgrp->threads[i] );
    kaapi_task_init(task, kaapi_tasksignalend_body, thgrp );
    kaapi_thread_pushtask(thgrp->threads[i]);    
  }
  
  /* free hash map entries: they are destroy by destruction of the version allocator */
  kaapi_hashmap_destroy( &thgrp->ws_khm );
  kaapi_versionallocator_destroy( &thgrp->ver_allocator );

  kaapi_mem_barrier();
  
  thgrp->state = KAAPI_THREAD_GROUP_MP_S;
  return 0;
}
