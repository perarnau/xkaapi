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

/** Call by attribut of the user level C++ ka::SetPartition
    The task should be pushed into the thread.
*/
int kaapi_thread_online_computedep(kaapi_thread_t* frame, int pid, kaapi_task_t* task)
{
  int err;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;
  err = kaapi_thread_computedep_task( thread, frame->tasklist, task );
  if (err ==0)
    kaapi_thread_pushtask(frame);
  return err;
}

#define KAAPI_USE_PERFCOUNTER 1
/**
*/
int kaapi_thread_computedep_task(kaapi_thread_context_t* thread, kaapi_tasklist_t* tasklist, kaapi_task_t* task)
{
  kaapi_format_t*         task_fmt;
  kaapi_task_body_t       task_body;
  kaapi_taskdescr_t*      taskdescr =0;
  kaapi_metadata_info_t*  mdi;
  kaapi_version_t*        version;
  kaapi_version_t*        all_versions[32];

  /* assume task list  */
  kaapi_assert( tasklist != 0);
  
  task_body = kaapi_task_getbody(task);
  if (task_body!=0)
    task_fmt= kaapi_format_resolvebybody(task_body);
  else
    task_fmt = 0;
  if (task_fmt ==0) 
  {
    return EINVAL;
  }

  /* new task descriptor */
  taskdescr = kaapi_tasklist_allocate_td( tasklist, task );
  ++tasklist->cnt_tasks;
  
  /* compute if the i-th access is ready or not.
     If all accesses of the task are ready then push it into readylist.
  */
  void* sp = task->sp;
  size_t count_params = kaapi_format_get_count_params(task_fmt, sp );
  kaapi_assert( count_params <= 32 );
  
  for (unsigned int i=0; i < count_params; i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(task_fmt, i, sp) );
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* its an access */
    kaapi_access_t access = kaapi_format_get_access_param(task_fmt, i, sp);

    /* find the version info of the data using the hash map */
    mdi = kaapi_mem_findinsert_metadata( access.data );
    if ( !_kaapi_metadata_info_is_valid(mdi, thread->asid) )
    {
      kaapi_memory_view_t view = kaapi_format_get_view_param(task_fmt, i, task->sp);
      _kaapi_metadata_info_bind_data( mdi, thread->asid, access.data, &view );
      /* no version -> new version object: no writer */
      mdi->version[0] = version = kaapi_thread_newversion( mdi, thread->asid, access.data, &view );
    }
    else {
      /* have a look at the version and detect dependency or not etc... */
      version = mdi->version[0];
    }
    all_versions[i] = version;

    /* compute the deepth (locagical date) to avoid some WAR or WAW */
    kaapi_thread_computeready_date( version, taskdescr, m );

    /* change the data in the task by the handle */
    access.data = version->handle;
    kaapi_format_set_access_param(task_fmt, i, task->sp, &access);

    /* store the format to avoid lookup */
    taskdescr->fmt = task_fmt;
  }
  
  tasklist->t_infinity = _kaapi_max(taskdescr->date, tasklist->t_infinity);

  for (unsigned int i=0; i < count_params; i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(task_fmt, i, sp) );
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* get version of the i-th parameter find in the previous iteration over args */
    version = all_versions[i];

    kaapi_thread_computeready_access( tasklist, version, taskdescr, m );
  } /* end for all arguments of the task */
  
  /* if wc ==0, push the task into the ready list */
  if (taskdescr->wc == 0)
    kaapi_tasklist_pushback_ready(tasklist, taskdescr);

  return 0;
}
