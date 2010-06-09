/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** theo.trouillon@imag.fr
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


/** 2 gros bug dans le code de Theo:
    hashmap avec access au lieu de access->data
    ATOMIC_DECR
*/

/** task is the top task not yet pushed.
 */
int kaapi_threadgroup_computedependencies(kaapi_threadgroup_t thgrp, int threadindex, kaapi_task_t* task)
{
  kaapi_thread_t*      thread;
  kaapi_format_t*      task_fmt;
  
  /* pass in parameter ? cf C++ thread interface */
  kaapi_assert_debug( (threadindex >=-1) && (threadindex < thgrp->group_size) );

  if(task->body==kaapi_suspend_body && task->ebody!=0)
    task_fmt= kaapi_format_resolvebybody(task->ebody);
  else if (task->body!=0)
    task_fmt= kaapi_format_resolvebybody(task->body);
  else
    task_fmt = 0;

  if (task_fmt ==0) return EINVAL;
  
  /* get the thread where to push the task */
  thread = kaapi_threadgroup_thread( thgrp, threadindex );
  
  /* initialize the pad for every pushed task */
  task->pad = 0;

  /* find the last writer for each args and in which partition it is 
     -> if all writers are in the same partition do nothing, push the task in the i-th partition
     -> if one of the writer is in a different partition, then change the body of the writer
     in order to add information to signal the task that are waiting for parameter
     
    ASSUMPTIONS:
    1- all the threads in the group are inactive and not subject to steal operation
    2- we only consider R,W and RW dependencies, no CW that implies multiple writers
  */
  kaapi_hashentries_t* entry; //Current argument's entry in the Hashmap
  void* sp = task->sp;
  for (int i=0;i<task_fmt->count_params;i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(task_fmt->mode_params[i]);
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* its an access */
    kaapi_access_t* access = (kaapi_access_t*)(task_fmt->off_params[i] + (char*)sp);
    entry = 0;

    /* find the last writer (task & thread) using the hash map */
    entry = kaapi_hashmap_find(&thgrp->ws_khm, access->data);
    if (entry ==0)
    {
      /* no entry -> new version object */
      entry = kaapi_threadgroup_newversion( thgrp, &thgrp->ws_khm, threadindex, access );
      if (KAAPI_ACCESS_IS_READ(m))
        kaapi_threadgroup_version_addfirstreader( thgrp, &thgrp->ws_vect_input, threadindex, task, access, i );
    }

    if (KAAPI_ACCESS_IS_READ(m))
    {
      task = kaapi_threadgroup_version_newreader( thgrp, entry->u.dfginfo, threadindex, task, access, i );
    }
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      task = kaapi_threadgroup_version_newwriter( thgrp, entry->u.dfginfo, threadindex, task, access, i );
    }
    
  } /* end for all arguments of the task */

  return 0;
}
