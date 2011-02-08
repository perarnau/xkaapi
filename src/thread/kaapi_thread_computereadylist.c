/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@imag.fr
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
int kaapi_sched_computereadylist( )
{
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;
  return kaapi_thread_computereadylist( thread );
}


/** task is the top task not yet pushed.
    This function is called is task is pushed into a specific thread using
    the C++ ka::SetPartition(site) attribut or the thread group access.
    
 */
int kaapi_thread_computereadylist( kaapi_thread_context_t* thread )
{
  kaapi_frame_t*          frame;
  kaapi_task_t*           task_top;
  kaapi_format_t*         task_fmt;
  kaapi_task_body_t       task_body;
  kaapi_taskdescr_t*      taskdescr =0;
  kaapi_tasklist_t*       tasklist =0;
  kaapi_hashentries_t*    entry;
  kaapi_hashentries_bloc_t stackbloc;
  kaapi_version_t*        version;

  /* assume no task list or task list is empty */
  frame    = thread->sfp;
  tasklist = frame->tasklist;
  kaapi_assert_debug( (tasklist==0) || (kaapi_tasklist_isempty(tasklist)) );
  kaapi_tasklist_init( tasklist, frame );
  
  /* new history of visited data */
  kaapi_hashmap_t dep_khm;
  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &dep_khm, &stackbloc );
  
  /* iteration over all tasks of the current top frame thread->sfp */
  task_top  = frame->pc;
  while (task_top > frame->sp)
  {
    task_body = kaapi_task_getbody(task_top);
    if (task_body!=0)
      task_fmt= kaapi_format_resolvebybody(task_body);
    else
      task_fmt = 0;
    if (task_fmt ==0) return EINVAL;

    /* new task descriptor */
    taskdescr = kaapi_tasklist_allocate_td( tasklist, task_top );
    
    /* compute if the i-th access is ready or not.
       If all accesses of the task are ready then push it into readylist.
    */
    void* sp = task_top->sp;
    size_t count_params = kaapi_format_get_count_params(task_fmt, sp );
    for (unsigned int i=0; i < count_params; i++) 
    {
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(task_fmt, i, sp) );
      if (m == KAAPI_ACCESS_MODE_V) 
        continue;
      
      /* its an access */
      kaapi_access_t access = kaapi_format_get_access_param(task_fmt, i, sp);
      entry = 0;

      /* find the version info of the data using the hash map */
      entry = kaapi_hashmap_findinsert(&dep_khm, access.data);
      if (entry->u.version ==0)
      {
        kaapi_memory_view_t view = kaapi_format_get_view_param(task_fmt, i, task_top->sp);

        /* no entry -> new version object: no writer */
        entry->u.version = kaapi_thread_newversion( access.data, &view );
      }
      else {
        /* have a look at the version and detect dependency or not etc... */
        version = entry->u.version;
        kaapi_thread_computeready_access( tasklist, version, taskdescr, m );
      }
      
      /* change the data in the task by the handle */
      access.data = version->handle;
      kaapi_format_set_access_param(task_fmt, i, task_top->sp, &access);

      /* change body by handle's body version */
#warning "      kaapi_task_setbody(task_top, kaapi_format_get_handle_body(task_fmt) ); "
      
    } /* end for all arguments of the task */
    
    /* if counter ==0, push the task into the ready list */
    if (KAAPI_ATOMIC_READ(&taskdescr->counter) == 0)
      kaapi_tasklist_pushback_ready(tasklist, taskdescr);

    --task_top;
  } /* end while task */
  
  return 0;
}
