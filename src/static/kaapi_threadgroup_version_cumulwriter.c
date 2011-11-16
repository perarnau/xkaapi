/*
 ** xkaapi
 ** 
 **
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
#include "kaapi_staticsched.h"


/* Cumulative write is a mix of reader / writer task creation
   - the first cumulative write fix the site for the final version
   - a cumlative write on a new asid will add:
      - a reception on the master site
   - at the end of cumul (after processing all cw accesses), then
   for each copies add a bcast task to send local accumulation to
   master site.
   
   Each local accumulation will signal the bcast task that has
   a waiting counter equals to the number of local accumulations per
   asid.
   
   The code must be executed properly by the gid owner of the cw
   as well as the gid of the final version (for reduction) : he should
   know where are the copies to generate the correct communication.
   And to have correct generation of tag for communication: all threads
   should execute the code (except for allocation / pushing into successor list etc...)
*/
int kaapi_threadgroup_version_newwriter_cumulwrite( 
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver, 
    int                   tid, 
    kaapi_access_mode_t   mode,
    kaapi_taskdescr_t*    task, 
    const kaapi_format_t* fmt,
    int                   ith,
    kaapi_access_t*       access
)
{
  void* data =0;
  kaapi_memory_view_t view;
  kaapi_taskdescr_t* dv_task =0;
  int retval;
  int is_first_access_asid = 0;

  kaapi_assert_debug( (-1 <= tid) && (tid < thgrp->group_size) );
  kaapi_assert_debug( KAAPI_ACCESS_IS_CUMULWRITE(mode) );

  /* asid for the target thread */
  kaapi_address_space_id_t asid = kaapi_threadgroup_tid2asid(thgrp, tid);

  kaapi_globalid_t gid_writer = kaapi_threadgroup_tid2gid( thgrp, tid );

  /* a priori: access is not ready */
  retval = 0;
 
  /* it is the first cw access ? */
  if ( !KAAPI_ACCESS_IS_CUMULWRITE(ver->writer_mode) )
  {
  
    /* delete copies */
    kaapi_data_version_list_append( &ver->todel, &ver->copies );
    is_first_access_asid = 1;

    /* find data version with same asid: 1/ in copies (WAR) 2/ in writer (WAW) */
    kaapi_data_version_t* dv = kaapi_version_findcopiesrmv_asid_in( ver, asid );
    if ((dv ==0) && (ver->writer.asid == asid))
    {
      dv = &ver->writer;
    }

    if (dv != 0)
    {
      data    = dv->addr;
      view    = dv->view;
      dv_task = dv->task;

      /* recycle the data version if != ver->writer */
      if (dv != &ver->writer) 
        kaapi_threadgroup_deallocate_dataversion(thgrp, dv);

      if (dv_task ==0) 
      {
        /* first access: mute the origin and the task is set as ready, keep data */
        retval = 1;
      }
      else if (dv_task != task)
      {
        if (gid_writer == thgrp->localgid)
        {
          /* avoid WAW (if dv=writer) or WAR (dv=a copy) */
          kaapi_taskdescr_push_successor( thgrp->threadctxts[tid]->sfp->tasklist, dv_task, task );
        }
      }
      ver->writer.asid   = asid;
      ver->writer_thread = tid;
      kaapi_assert_debug(ver->writer.asid == asid);
      ver->writer.task   = task;
      ver->writer.ith    = ith;
      ver->writer.view   = kaapi_format_get_view_param(fmt, ith, task->task->sp);
      ver->writer.reducor= 0;
    } 
    else /* else case: no find data version in the same asid */
    {
      /* mute the writer */
      ver->writer.asid   = asid;
      ver->writer_thread = tid;
      ver->writer.ith    = ith;
      ver->writer.view   = kaapi_format_get_view_param(fmt, ith, task->task->sp);
      ver->writer.reducor= 0;
//      is_first_access_asid = 1;

      retval = 1;

      if (ver->writer.task ==0) 
      {
        /* the task is set ready and keep original data */
        data = ver->writer.addr;
        ver->writer.task   = task;
      } 
      else 
      {
        ver->writer.task   = task;
        if (gid_writer == thgrp->localgid)
        {
          /* allocate the data for the final version */
          data  = (void*) kaapi_memory_allocate_view( 
                  asid,
                  &ver->writer.view,
                  KAAPI_MEM_SHARABLE );
          ver->writer.addr = data;
        }
      }
    }
    
    if (gid_writer == thgrp->localgid)
    {
      kaapi_assert_debug( data != 0 );
      kaapi_access_t a;
      a.data = data;
      kaapi_format_set_cwaccess_param(fmt, ith, task->task->sp, &a, is_first_access_asid );
      kaapi_format_set_view_param(    fmt, ith, task->task->sp, &ver->writer.view );
    }

    ver->writer_mode   = mode;
    ver->writer_thread = tid;

    /* mark copies as deleted and return
    */
    kaapi_data_version_list_append( &ver->todel, &ver->copies );
    return retval;
  }
  
  /* now the other case: this is not the first cw access 
     - insert the new access in the list of copies (when next mode will be != cw) 
     then call finalize_cwaccesses function.
  */

  /* find the data info in the map attached to the version and remove it from list if found, else return 0 */
  kaapi_data_version_t* dv = kaapi_version_findcopiesrmv_asid_in( ver, asid );
  if ((dv ==0) && (ver->writer.asid == asid))
    dv = &ver->writer;
    
  if (dv !=0) 
  {
    data = dv->addr;

    /* assumer views remains equal */
    /* dv->view == kaapi_format_get_view_param(fmt, ith, task->task->sp); */

    dv_task     = dv->task;
    dv->asid    = asid;
    dv->task    = task;
    dv->ith     = ith;
    dv->reducor = kaapi_format_get_reducor(fmt, ith, task->task->sp);
    
    if (gid_writer == thgrp->localgid)
    {
      /* push the task as a successor of the task attached to the data version */
      kaapi_taskdescr_push_successor( thgrp->threadctxts[tid]->sfp->tasklist, dv_task, task );
    }
    retval = 0;
  }
  else 
  {
    is_first_access_asid = 1;
    /* this is a new access on asid : find data into todel ? */
    dv = kaapi_version_findtodelrmv_asid_in( ver, asid );
    data =0;
    if (dv !=0) 
    {
      view = kaapi_format_get_view_param(fmt, ith, task->task->sp);
      if (kaapi_memory_view_size(&dv->view) == kaapi_memory_view_size(&view))
      {
        data = dv->addr;
        dv_task = dv->task;

        /* avoid WAR here: may be solved at runtime */
        retval = (dv_task == 0);

        if ((dv_task !=0) && (gid_writer == thgrp->localgid))
        { /* avoid WAW or WAR */
          kaapi_taskdescr_push_successor( thgrp->threadctxts[tid]->sfp->tasklist, dv_task, task );
        }
      }
      else {
        dv->view = view;
      } 
      dv->asid    = asid;
      dv->task    = task;
      dv->ith     = ith;
      dv->reducor = kaapi_format_get_reducor(fmt, ith, task->task->sp);
    } 
    else {
      /* new data version access */
      dv = kaapi_threadgroup_allocate_dataversion(thgrp);
      dv->asid = asid;
      dv->task = task;
      dv->ith  = ith;
      dv->view = kaapi_format_get_view_param(fmt, ith, task->task->sp);
      dv->reducor = kaapi_format_get_reducor(fmt, ith, task->task->sp);
      dv->next = 0;
    }
    
    if (data ==0)
    {
      /* no copy on asid: access is ready. need to allocate data */
      data = (void*)kaapi_memory_allocate_view( 
            kaapi_threadgroup_tid2asid(thgrp, tid), 
            &dv->view,
            KAAPI_MEM_LOCAL );
      dv->addr = data;
      retval = 1;
    }
  }

  /* set the access for the task with correct data */
  if (gid_writer == thgrp->localgid)
  {
    kaapi_assert_debug( data != 0 );
    kaapi_access_t a;
    a.data = data;
    kaapi_format_set_cwaccess_param(fmt, ith, task->task->sp, &a, is_first_access_asid );
    kaapi_format_set_view_param(  fmt, ith, task->task->sp, &dv->view );
  }

  /* link it into list of copies */
  if (dv != &ver->writer)
    kaapi_data_version_list_add(&ver->copies, dv );

  /* return 1: the access is ready ! */
  return retval;
}



/**
*/
int kaapi_threadgroup_version_finalize_cw(
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver
)
{
  /* at least one cw, so not in writer_thread */
  kaapi_assert_debug( ver->writer_thread != -1 ); 

  /* - reference site is ver->writer 
     - other distributed cumulative are on each asid in ver->copies
    
     For each recv add a task (with RW access mode)
  */
  
  kaapi_globalid_t gid_writer = kaapi_threadgroup_tid2gid( thgrp, ver->writer_thread );
  kaapi_taskdescr_t* td_finalizer = 
    kaapi_tasklist_allocate_td( thgrp->threadctxts[ver->writer_thread]->sfp->tasklist, 0 );

  kaapi_data_version_t* curr = ver->copies.front;
  while (curr !=0)
  {
    /* new tag for communciation */
    kaapi_comtag_t tag = ++thgrp->tag_count;
    kaapi_globalid_t gid_sender = kaapi_memory_address_space_getgid(curr->asid);
    int tid_receiver = kaapi_memory_address_space_getgid(ver->writer.asid);
    if (gid_writer == thgrp->localgid)
    {
      kaapi_taskdescr_t* td_reduce = 
          kaapi_tasklist_allocate_td(thgrp->threadctxts[ver->writer_thread]->sfp->tasklist, 0);
      td_reduce->reduce_fnc = curr->reducor;
      td_reduce->context    = ver->writer.addr;
      kaapi_comrecv_t* wc = (kaapi_comrecv_t*)kaapi_allocator_allocate(&thgrp->allocator, sizeof(kaapi_comrecv_t));
      wc->tag        = tag;
      wc->from       = gid_sender;
      wc->tid        = ver->writer_thread;
      wc->tasklist   = thgrp->threadctxts[ver->writer_thread]->sfp->tasklist;
      wc->list.front = 0;
      wc->list.back  = 0;
      wc->view       = curr->view;
      wc->data       = (void*) kaapi_memory_allocate_view( 
              kaapi_threadgroup_tid2asid(thgrp, tid_receiver), 
              &wc->view,
              KAAPI_MEM_SHARABLE );

      td_reduce->value      = wc->data;
      
      /* push the reducer as a successor of the last cw writer task */
      kaapi_taskdescr_push_successor( 
        thgrp->threadctxts[ver->writer_thread]->sfp->tasklist,
        ver->writer.task,
        td_reduce
      );

      /* increment counter for external synchronisation : should be done into function push ? */
      KAAPI_ATOMIC_INCR(&td_reduce->counter);
    
      /* push the task reduce into the activation link of the comrecv data structure */
      kaapi_activationlist_pushback( thgrp, &wc->list, td_reduce );

      /* push the finalizer as a successor of the reduction task */
      kaapi_taskdescr_push_successor( 
        thgrp->threadctxts[ver->writer_thread]->sfp->tasklist, 
        td_reduce, 
        td_finalizer 
      );

      /* register it into global table */
      kaapi_threadgroup_comrecv_register( thgrp, ver->writer_thread, tag, wc );
    }

    if (gid_sender == thgrp->localgid)
    {
      /* add bcast info on the last task that do cw */
      kaapi_taskbcast_arg_t* bcast;
      curr->task->bcast = bcast = 
        (kaapi_taskbcast_arg_t*)kaapi_tasklist_allocate( thgrp->threadctxts[gid_sender]->sfp->tasklist, 
                                                         sizeof(kaapi_taskbcast_arg_t));
      bcast->front.vertag          = tag;
      bcast->front.ith             = curr->ith;  /* no parameter */
      bcast->front.data            = curr->addr;
      bcast->front.view            = curr->view;
      bcast->front.next            = 0;
      bcast->front.front.tag       = tag;
      bcast->front.front.asid      = ver->writer.asid;
      bcast->front.front.rsignal   = 0;
      bcast->front.front.raddr     = 0;
      kaapi_memory_view_clear( &bcast->front.front.rview );
      bcast->front.front.next      = 0;
      bcast->front.back            = &bcast->front.front;
      bcast->back                  = &bcast->front;

      kaapi_threadgroup_comsend_register( thgrp, kaapi_memory_address_space_getuser(curr->asid), tag, &bcast->front );  
    }

    curr = curr->next;
  }

  /* update writer to be the finalizer task */
  ver->writer_mode = KAAPI_ACCESS_MODE_RW;
  ver->writer.task = td_finalizer;
  
  /* delete copies */
  kaapi_data_version_list_append( &ver->todel, &ver->copies );
  
  /* reset tag to regenerate communication with new tag */
  ver->tag = 0;
  return 0;
}

