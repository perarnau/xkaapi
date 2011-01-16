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
  if (thgrp->state != KAAPI_THREAD_GROUP_MP_S) return EINVAL;

  /* same the main thread frame to restore it at the end of parallel computation */
  kaapi_thread_save_frame( thgrp->threads[-1], &thgrp->save_maintopframe);
  
  /* recopy the task data structure: WARNING Stack growth down */
  thgrp->size_mainthread = (uint32_t)(thgrp->save_maintopframe.pc - thgrp->save_maintopframe.sp);
  thgrp->save_mainthread = malloc( thgrp->size_mainthread * sizeof(kaapi_task_t) );
  memcpy(thgrp->save_mainthread, thgrp->save_maintopframe.sp+1, thgrp->size_mainthread*sizeof(kaapi_task_t) );
  
  /* do the same for each worker */
  thgrp->save_workerthreads  = malloc( sizeof(kaapi_task_t*) * thgrp->group_size );
  thgrp->save_workertopframe = malloc( sizeof(kaapi_frame_t) * thgrp->group_size );
  thgrp->size_workerthreads  = malloc( sizeof(int) * thgrp->group_size );
  for (int i=0; i<thgrp->group_size; ++i)
  {
    kaapi_thread_save_frame(thgrp->threads[i], &(thgrp->save_workertopframe[i]) );
    thgrp->size_workerthreads[i] 
      = (int)(thgrp->save_workertopframe[i].pc - thgrp->save_workertopframe[i].sp);
    thgrp->save_workerthreads[i] = malloc( sizeof(kaapi_task_t) * thgrp->size_workerthreads[i] );
    memcpy( thgrp->save_workerthreads[i], 
            thgrp->save_workertopframe[i].sp+1, 
            thgrp->size_workerthreads[i]*sizeof(kaapi_task_t) );
  }
  
  return 0;
}




/**
*/
int kaapi_threadgroup_restore_thread( kaapi_threadgroup_t thgrp, int tid )
{
  kaapi_assert( (tid >=-1) && (tid < thgrp->group_size) );

  if (tid == -1) {
    /* recopy the main thread */
    memcpy( thgrp->save_maintopframe.sp+1, 
            thgrp->save_mainthread, 
            thgrp->size_mainthread*sizeof(kaapi_task_t) );

    /* restore the main frame */
    kaapi_thread_restore_frame( thgrp->threads[-1], &thgrp->save_maintopframe);
  }
  else {
    memcpy( thgrp->save_workertopframe[tid].sp+1, 
            thgrp->save_workerthreads[tid], 
            thgrp->size_workerthreads[tid]*sizeof(kaapi_task_t) );

    kaapi_thread_restore_frame(thgrp->threads[tid], &(thgrp->save_workertopframe[tid]) );

    /* reset frame pointer to the first frame: assume only one thread */
    thgrp->threadctxts[tid]->sfp = thgrp->threadctxts[tid]->stackframe;
  }
  return 0;
}



/**
*/
int kaapi_threadgroup_restore(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_WAIT_S) return EINVAL;

#if 0
  /* do the same for each worker */
  for (int i=-1; i<thgrp->group_size; ++i)
  {
    kaapi_threadgroup_restore_thread(thgrp, i);
  }
#else
  /* worker threads are already restored in signalend */
  kaapi_threadgroup_restore_thread(thgrp, -1);
#endif
  thgrp->state = KAAPI_THREAD_GROUP_MP_S;
  return 0;
}
