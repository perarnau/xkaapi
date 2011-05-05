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


/** task is the top task not yet pushed.
    This function is called is task is pushed into a specific thread using
    the C++ ka::SetPartition(site) attribut or the thread group access.
    
    The code is the same code as kaapi_thread_computereadylist except that
    the code generates bcast and recv task between different address spaces.
 */
int kaapi_threadgroup_computedependencies(kaapi_threadgroup_t thgrp, int tid, kaapi_task_t* task)
{
  kaapi_thread_context_t* thread;
  kaapi_format_t*         fmt;
  kaapi_task_body_t       body;
  kaapi_taskdescr_t*      taskdescr =0;
  kaapi_tasklist_t*       tasklist =0;
  kaapi_metadata_info_t*  mdi;
  kaapi_version_t*        version;
  kaapi_version_t*        all_version[32];
  kaapi_metadata_info_t*  all_mdi[32];
  
  /* assume task exists */
  thread = thgrp->threadctxts[tid];
  tasklist = thread->sfp->tasklist;
  kaapi_assert_debug( (tasklist !=0) );

  body = kaapi_task_getbody(task);
  if (body!=0)
    fmt= kaapi_format_resolvebybody(body);
  else
    fmt = 0;
  if (fmt ==0) 
    return EINVAL;

  /* new task descriptor */
  taskdescr = kaapi_tasklist_allocate_td( tasklist, task );
  kaapi_thread_pushtask( kaapi_threadcontext2thread(thread) );
  
  /* compute if the i-th access is ready or not.
     If all accesses of the task are ready then push it into readylist.
  */
  void* sp = task->sp;
  size_t count_params = kaapi_format_get_count_params(fmt, sp );
  kaapi_assert( count_params <= 32 );
  
  for (unsigned int i=0; i < count_params; i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, sp) );
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* its an access */
    kaapi_access_t access = kaapi_format_get_access_param(fmt, i, sp);

    /* find the version info of the data using the hash map */
    all_mdi[i] = mdi = kaapi_mem_findinsert_metadata( access.data );
    if ( _kaapi_metadata_info_is_novalid(mdi))
    {
      kaapi_version_t** ptrversion;
      kaapi_memory_view_t view = kaapi_format_get_view_param(fmt, i, task->sp);
      /* no version -> new version object: the writer will be put by the first task... 
      */
      ptrversion  = _kaapi_metadata_info_bind_data( mdi, thgrp->tid2asid[-1], access.data, &view );
      *ptrversion = version = kaapi_thread_newversion( mdi, thgrp->tid2asid[-1], access.data, &view );
      version->writer_tasklist = thread->sfp->tasklist;
      
      /* call it to create the initial task if access mode is read */
      if (KAAPI_ACCESS_IS_READ(m))
        kaapi_threadgroup_create_initialtask( thgrp->threadctxts[-1]->sfp->tasklist, version, m );
      version->writer_asid = thgrp->tid2asid[-1];
    }
    if (_kaapi_metadata_info_is_valid(mdi, thread->asid) )
    {
      /* have a already a valid version in this asid */
      version = _kaapi_metadata_info_get_version(mdi, thread->asid);
      
      /* reused the standard local computeready access */
      kaapi_thread_computeready_access( tasklist, version, taskdescr, m );
    }
    else 
    { /* The mdi does not have a valid copy on this asid.
         Add a bcast / recv between version-> writer_asid and thread->asid 
         iff the thread requires the data in input mode (r or rw)
      */
      if (KAAPI_ACCESS_IS_READ(m))
      {
        /* find one of the writers (which has a valid copy of the data) */
        kaapi_version_t* version_writer = _kaapi_metadata_info_find_onewriter(mdi, thread->asid);

        /* create a new version on receiver thread */
        kaapi_version_t** ptrversion 
          = _kaapi_metadata_info_copy_data( mdi, thread->asid, &version_writer->orig->view  );
        *ptrversion = version = kaapi_thread_copyversion( mdi, thread->asid, version_writer );
        version->writer_tasklist = tasklist;
        kaapi_taskdescr_t* td_recv;
        kaapi_task_t* task_recv;

        kaapi_recv_arg_t* argrecv 
          = (kaapi_recv_arg_t*)kaapi_tasklist_allocate(tasklist, sizeof(kaapi_recv_arg_t) );

        /* recv data into recv_data */
        argrecv->tag         = version_writer->tag; 
        argrecv->from        = kaapi_memory_address_space_getgid(version_writer->writer_asid);
        argrecv->data        = 0; /* to allocate on remote site */
        argrecv->dest        = version->handle;
        task_recv            = kaapi_tasklist_allocate_task( tasklist, kaapi_taskrecv_body, argrecv);
        td_recv              = kaapi_tasklist_allocate_td( tasklist, task_recv );
        
        version->last_mode   = KAAPI_ACCESS_MODE_W;
        version->last_task   = td_recv;
        version->writer_task = td_recv;
        version->writer_asid = thread->asid;
        
        /* Add a bcast on the writer side */
        kaapi_taskdescr_t* td_writer = version_writer->writer_task;
        if (version_writer->tag ==0)
        {
          version_writer->tag = ++thgrp->count_tag;
          argrecv->tag = version_writer->tag;
          
          /* not bcast task exist, do not search for them */
          kaapi_task_t* task_bcast;
          kaapi_taskdescr_t* td_bcast;
          kaapi_bcast_arg_t* argbcast /* to allocate onto the tasklist of the writer */
            = (kaapi_bcast_arg_t*)kaapi_tasklist_allocate(
                version_writer->writer_tasklist, sizeof(kaapi_bcast_arg_t) 
          );
          /* recv data into recv_data */
          argbcast->tag             = version_writer->tag; 
          argbcast->src             = version_writer->handle;
          argbcast->front.dest.ptr  = 0; /* to allocate on remote site + exchange pointer */
          argbcast->front.dest.asid = thread->asid;
          argbcast->front.rsignal   = 0; /* exchange to known receiver task */
          argbcast->front.next      = 0;
          argbcast->back            = &argbcast->front;
          task_bcast                = kaapi_tasklist_allocate_task( tasklist, kaapi_taskbcast_body, argbcast);
          td_bcast               /* to allocate writer tasklist */
            = kaapi_tasklist_allocate_td( version_writer->writer_tasklist, task_bcast );
          argbcast->td_bcast        = td_bcast;
          
          kaapi_tasklist_push_broadcasttask( version_writer->writer_tasklist, td_writer, td_bcast );
        }
        else {
          /* else the version tag != 0 => a bcast has already been pushed */
          kaapi_assert( version_writer->writer_task->bcast != 0);
          /* find the bcast for the parameter version_writer->handle */
          kaapi_activationlink_t* onebcast = version_writer->writer_task->bcast->front;
          kaapi_bcast_arg_t* argbcast = 0;
          kaapi_assert( onebcast != 0)
          while (onebcast != 0)
          {
            argbcast = (kaapi_bcast_arg_t*)onebcast->td->task.sp;
            if (argbcast->src == version_writer->handle) break;
            onebcast = onebcast->next;
          }
          
          kaapi_assert (onebcast !=0);
          /* add a new destination: thread->asid  */
          kaapi_bcast_onedest_t* onemoredest 
            = (kaapi_bcast_onedest_t*)kaapi_tasklist_allocate(version_writer->writer_tasklist, 
                    sizeof(kaapi_bcast_onedest_t) );
          onemoredest->dest.ptr  = 0; /* to allocate on remote site + exchange pointer */
          onemoredest->dest.asid = thread->asid;
          onemoredest->rsignal   = 0; /* exchange to known receiver task */
          onemoredest->next      = 0;
          argbcast->back->next = onemoredest;
          argbcast->back = onemoredest;
        }

        /* add taskdescr as a successor of td_recv */
        kaapi_tasklist_push_successor( tasklist, td_recv, taskdescr );

        /* push the receive task */
        kaapi_tasklist_push_receivetask( tasklist, version_writer->tag, td_recv );
        version->last_mode    = m;
        version->last_task    = taskdescr;

        /* invalidate old copies _kaapi_metadata_info_set_writer( mdi, thread->asid ); */
      }
      else 
      {
        /* only create a new local version: invalidation will be done after computing 
           the local dependencies
        */
        kaapi_version_t** ptrversion;
        kaapi_memory_view_t view = kaapi_format_get_view_param(fmt, i, task->sp);
        _kaapi_metadata_info_unbind_alldata( mdi );
        ptrversion = _kaapi_metadata_info_bind_data( mdi, thread->asid, access.data, &view );

        /* no version -> new version object: the writer will be put by the first task... 
        */
        *ptrversion = version = kaapi_thread_newversion( mdi, thread->asid, 0, &view );

        kaapi_taskdescr_t* td_alloc;
        kaapi_task_t* task_alloc;
        kaapi_move_arg_t* argalloc 
            = (kaapi_move_arg_t*)kaapi_tasklist_allocate(tasklist, sizeof(kaapi_move_arg_t) );
        argalloc->src_data  = version->orig;
        argalloc->dest      = version->handle;
        task_alloc = kaapi_tasklist_allocate_task( tasklist, kaapi_taskalloc_body, argalloc);
        td_alloc =  kaapi_tasklist_allocate_td( tasklist, task_alloc );
        version->writer_task= td_alloc;

        /* put task as successor the the ready task td_alloc */
        kaapi_tasklist_push_successor( tasklist, td_alloc, taskdescr );
        kaapi_tasklist_pushback_ready( tasklist, td_alloc);

        version->last_task     = taskdescr;
        version->last_mode     = m;
        version->last_tasklist = tasklist;
      }
    }
    
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      version->writer_asid     = thread->asid;
      version->writer_task     = taskdescr;
      version->writer_tasklist = tasklist;
      _kaapi_metadata_info_set_writer( mdi, thread->asid );
#pragma message ("warning: delete all copies")
    }

    /* change the data in the task by the handle */
    access.data = version->handle;
    kaapi_format_set_access_param(fmt, i, task->sp, &access);

    all_version[i] = version;
    
    /* store the format to avoid lookup */
    taskdescr->fmt = fmt;
  }
  
  /* if wc ==0, push the task into the ready list */
  if (taskdescr->wc == 0)
    kaapi_tasklist_pushback_ready(tasklist, taskdescr);

  return 0;
}
