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

static inline uint64_t _kaapi_max(uint64_t d1, uint64_t d2)
{ return (d1 < d2 ? d2 : d1); }

/** Call by attribut ka::SetPartition of the user level C++ 
*/
int kaapi_thread_online_computedep(kaapi_thread_t* frame, int pid, kaapi_task_t* task)
{
  int err;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;
  err = kaapi_thread_computedep_task( thread, frame->tasklist, task );
  return err;
}

#ifndef KAAPI_USE_PERFCOUNTER
#define KAAPI_USE_PERFCOUNTER 1
#endif
/**
*/
int kaapi_thread_computedep_task(kaapi_thread_context_t* thread, kaapi_tasklist_t* tasklist, kaapi_task_t* task)
{
  kaapi_format_t*         task_fmt;
  kaapi_task_body_t       task_body;
  kaapi_taskdescr_t*      taskdescr;
  kaapi_handle_t          handle;
  kaapi_version_t*        version;

  /* assume task list  */
  kaapi_assert( tasklist != 0);
  
  task_body = kaapi_task_getbody(task);
  if (task_body!=0)
    task_fmt= kaapi_format_resolvebybody(task_body);
  else
    task_fmt = 0;
  if (task_fmt ==0) 
    return EINVAL;

  /* new task descriptor in the task list */
  taskdescr = kaapi_tasklist_allocate_td( tasklist, task, task_fmt );
  
  /* Compute for each access i if it is ready or not.
     If all accesses of the task are ready then the taskdescr is pushed into readylist
     of the tasklist.
  */
  void* sp = task->sp;
  size_t count_params = kaapi_format_get_count_params(task_fmt, sp );
  
  for (unsigned int i=0; i < count_params; i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(task_fmt, i, sp) );
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* it is an access */
    kaapi_access_t access = kaapi_format_get_access_param(task_fmt, i, sp);

    /* find the version info of the data using a thread specific hash map 
       If no access exists, then add in the tasklist a task to move or allocate
       initial data for all the remainding tasks on the same tasklist.
    */
    version = kaapi_version_findinsert( thread, tasklist, access.data );
    if (version->last_mode == KAAPI_ACCESS_MODE_VOID)
    {
      kaapi_memory_view_t view = kaapi_format_get_view_param(task_fmt, i, task->sp);
      kaapi_version_add_initialaccess( version, tasklist, m, access.data, &view );
    }

    /* compute readiness of the access and return the handle to assign to the global data 
       to assign to the task
    */
    handle = kaapi_thread_computeready_access( tasklist, version, taskdescr, m );

    /* replace the pointer to the data in the task argument by the pointer to the global data */
    access.data = handle;
    kaapi_format_set_access_param(task_fmt, i, task->sp, &access);

  } /* end for all arguments of the task */
  
  /* store the format to avoid lookup */
  taskdescr->fmt = task_fmt;

  /* if wc ==0, push the task into the ready list */
  if (taskdescr->wc == 0)
    kaapi_tasklist_pushback_ready(tasklist, taskdescr);

  return 0;
}
