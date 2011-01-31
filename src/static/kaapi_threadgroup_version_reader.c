/*
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
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


static inline void kaapi_threadgroup_add_recvtask(
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver, 
    int                   tid, 
    kaapi_address_space_id_t asid,
    kaapi_taskdescr_t*    task, 
    const kaapi_format_t* fmt,
    int                   ith,
    kaapi_access_t*       access 
)
{
  kaapi_taskbcast_arg_t* bcast   = 0;
  kaapi_comsend_t*       comd    = 0;
  kaapi_comsend_raddr_t* comasid = 0;
  
  /* allocate the tag if is == 0 */
  if (ver->tag == 0) ver->tag = ++thgrp->tag_count;

  kaapi_globalid_t gid_writer = kaapi_threadgroup_tid2gid( thgrp, ver->writer_thread );
#if !defined(KAAPI_ADDRSPACE_ISOADDRESS)
  kaapi_globalid_t gid_reader = kaapi_threadgroup_tid2gid( thgrp, tid );
#endif

  /* look if the writer task has been defined: this code should be executed on the owner of the writer task */
  kaapi_tasklist_t* tasklist = thgrp->threadctxts[ver->writer_thread]->tasklist;

  if (gid_writer == thgrp->localgid)
  {
    if (ver->writer.task ==0) 
    {
      /* no writer task == data allocated on an address space and read in an other one 
         - allocate a new task descriptor without any associated task that will send data to the reader
      */
      kaapi_taskdescr_t* writer_task = kaapi_tasklist_allocate_td( tasklist, 0 );
      bcast = (kaapi_taskbcast_arg_t*)kaapi_tasklist_allocate( tasklist, sizeof(kaapi_taskbcast_arg_t));
      bcast->front.vertag          = ver->tag;
      bcast->front.ith             = -1;  /* no parameter */
      bcast->front.data            = access->data;
      bcast->front.view            = kaapi_format_get_view_param(fmt, ith, task->task->sp);
      bcast->front.next            = 0;

#if 0
printf("Add (1) sendaddr to asid:");
kaapi_memory_address_space_fprintf(stdout, asid);
printf("\n"); 
fflush(stdout);
#endif

      bcast->front.front.tag       = ver->tag;
      bcast->front.front.asid      = asid;
      bcast->front.front.rsignal   = 0;
      bcast->front.front.raddr     = 0;
      kaapi_memory_view_clear( &bcast->front.front.rview );
      bcast->front.front.next      = 0;
      bcast->front.back            = &bcast->front.front;

      bcast->back                  = &bcast->front;
      
      writer_task->bcast           = bcast;
      ver->writer.task             = writer_task;
      
      /* register the kaapi_comsend_t structure into the global table of tid writer_thread  
         In order to exchange address between memory space.
      */
      kaapi_threadgroup_comsend_register( thgrp, ver->writer_thread, ver->tag, &bcast->front );  

      /* push the bcast task into the ready list of task */
      kaapi_tasklist_pushback_ready(tasklist, writer_task);

      comd = &bcast->front;
      comasid = &comd->front;
    }
    else {
      /* the writer task already exists on w_asid address space:
         - register a remote reader into the bcast arg list 
      */
      bcast = ver->writer.task->bcast;
      if (bcast ==0) 
      {
        bcast = (kaapi_taskbcast_arg_t*)kaapi_tasklist_allocate( tasklist, sizeof(kaapi_taskbcast_arg_t));
        bcast->front.vertag        = ver->tag;
        bcast->front.ith           = -1;  /* no parameter */
        bcast->front.data          = ver->writer.addr;
        bcast->front.view          = (ver->writer.ith == -1 ? 
                                        kaapi_format_get_view_param(fmt, ith, task->task->sp) 
                                      : ver->writer.view);
        bcast->front.next          = 0;

#if 0
printf("Add (2) sendaddr to asid:");
kaapi_memory_address_space_fprintf(stdout, asid);
printf("\n"); 
fflush(stdout);
#endif
        bcast->front.front.tag     = ver->tag;
        bcast->front.front.asid    = asid;
        bcast->front.front.rsignal = 0;
        bcast->front.front.raddr   = 0;
        kaapi_memory_view_clear( &bcast->front.front.rview );
        bcast->front.front.next    = 0;
        bcast->front.back          = &bcast->front.front;

        bcast->back                = &bcast->front;
        
        ver->writer.task->bcast    = bcast;
        
        /* register the kaapi_comsend_t structure into the global table of tid writer_thread  
           In order to exchange address between memory space.
        */
        kaapi_threadgroup_comsend_register( thgrp, ver->writer_thread, ver->tag, &bcast->front );  

        comd = &bcast->front;
        comasid = &comd->front;
      }
      else 
      {
        /* find tag ver->tag into writer_task bcast information */
        comd = kaapi_sendcomlist_find_tag( bcast, ver->tag );
        if (comd ==0) 
        { /* no found, push a new comonedata info */
          comd = (kaapi_comsend_t*)kaapi_tasklist_allocate( tasklist, sizeof(kaapi_comsend_t));
          comd->vertag            = ver->tag;
          comd->ith               = -1;  /* here: ith of the parameter of the writer task */
          comd->data              = ver->writer.addr;
          comd->view              = (ver->writer.ith == -1 ? 
                                          kaapi_format_get_view_param(fmt, ith, task->task->sp) 
                                        : ver->writer.view);
          comd->next              = 0;

#if 0
printf("Add (3) sendaddr to asid:");
kaapi_memory_address_space_fprintf(stdout, asid);
printf("\n"); 
fflush(stdout);
#endif
          comd->front.tag         = ver->tag;
          comd->front.asid        = asid;
          comd->front.rsignal     = 0;
          comd->front.raddr       = 0;
          kaapi_memory_view_clear( &comd->front.rview );
          comd->front.next        = 0;
          comd->back              = &comd->front;
          
          bcast->back->next       = comd;
          bcast->back             = comd;

          /* register the kaapi_comsend_t structure into the global table of tid writer_thread  
             In order to exchange address between memory space.
          */
          kaapi_threadgroup_comsend_register( thgrp, ver->writer_thread, ver->tag, comd );  
          
          comasid = &comd->front;
        }
        else  /* else comd != 0 */
        {
          /* comd if found, look at com to asid */
          comasid = kaapi_sendcomlist_find_asid( comd, asid );
          if (comasid == 0)
          {
#if 0
printf("Add (4) sendaddr to asid:");
kaapi_memory_address_space_fprintf(stdout, asid);
printf("\n"); 
fflush(stdout);
#endif

            comasid = (kaapi_comsend_raddr_t*)kaapi_allocator_allocate( &thgrp->allocator, sizeof(kaapi_comsend_raddr_t));
            comasid->tag     = ver->tag;
            comasid->asid    = asid;
            comasid->rsignal = 0;
            comasid->raddr   = 0;
            kaapi_memory_view_clear( &comasid->rview );
            
            /* push it at the end: always back exist */
            comasid->next      = 0;
            comd->back->next   = comasid;
            comd->back         = comasid;
          }
          else {
            /* only register recv task */
#if 0
printf("Sendaddr to asid:");
kaapi_memory_address_space_fprintf(stdout, asid);
printf(" already exist\n"); 
fflush(stdout);
#endif
          }
        }
      }
    }
    
    
    /* also link globally for matching with recv if recv is not local */
    if ((comasid !=0) && (kaapi_memory_address_space_getgid(asid) != thgrp->localgid))
    {
      kaapi_comaddrlink_t* cal = (kaapi_comaddrlink_t*)kaapi_allocator_allocate( &thgrp->allocator, sizeof(kaapi_comaddrlink_t));
      cal->tag = comd->vertag;
      cal->send = comasid;
      cal->next = thgrp->all_sendaddr;
      thgrp->all_sendaddr = cal;
#if 0
      printf("%i:[kaapi_threadgroup_add_recvtask] register send com to %i tag:%i, tid_reader:%i, comsend:%p\n", 
          thgrp->localgid,
          (int)kaapi_memory_address_space_getgid(cal->send->asid),
          (int)cal->tag,
          (int)kaapi_memory_address_space_getuser(cal->send->asid), //ver->writer_thread,
          (void*)cal
      );
      fflush(stdout);
#endif
    }

  } /* if local writer */
  
#if defined(KAAPI_ADDRSPACE_ISOADDRESS)
  /* in that configuration all threads maintain remote address allocation
     event if they do not participate in the communication
  */
  if (1)
#else
  if (gid_reader == thgrp->localgid)
#endif
  {
    /* ok here is the receive side of the communication 
       - allocate the kaapi_comrecv_t 
       - register it into global table for the thread tid.
       - this is a persistant data structure
    */
    kaapi_comrecv_t* wc = kaapi_recvcomlist_find_tag( thgrp->lists_recv[tid], ver->tag );
    kaapi_access_t a; /* to store data access and allocate */
    if (wc ==0)
    {
#if defined(KAAPI_ADDRSPACE_ISOADDRESS)
      wc = (kaapi_comrecv_t*)kaapi_memory_allocate( 
                      kaapi_threadgroup_tid2asid(thgrp, tid), 
                      sizeof(kaapi_comrecv_t),
                      KAAPI_MEM_SHARABLE );
#else
      wc = (kaapi_comrecv_t*)kaapi_allocator_allocate(&thgrp->allocator, sizeof(kaapi_comrecv_t));
#endif
      wc->tag        = ver->tag;
      wc->from       = kaapi_memory_address_space_getgid(ver->writer.asid);
      wc->tid        = tid;
      wc->reduce_fnc = 0;
      wc->result     = 0; // debug only
      wc->tasklist   = thgrp->threadctxts[tid]->tasklist;
      wc->list.front = 0;
      wc->list.back  = 0;
      wc->view       = kaapi_format_get_view_param(fmt, ith, task->task->sp);
      
      /* this instruction should executed by every body 
         Only the owner of thread tid allocate data.
         Other threads only compute address.
      */
      wc->data       = a.data = (void*) kaapi_memory_allocate_view( 
              kaapi_threadgroup_tid2asid(thgrp, tid), 
              &wc->view,
              KAAPI_MEM_SHARABLE );
      wc->next       = 0;

#if 0
      printf("%i:[kaapi_threadgroup_add_recvtask] allocate new recv synchro tag:%i, tid:%i rsignal:%p, data@:%p, size: %lu\n", 
          thgrp->localgid,
          (int)wc->tag,
          (int)wc->tid,
          (void*)wc,
          (void*)wc->data,
          kaapi_memory_view_size(&wc->view)
      );
      fflush(stdout);
#endif

    }
    else {
      a.data = wc->data;
    }

    /* push the task into the activation link of the comrecv data structure */
    kaapi_activationlist_pushback( thgrp, &wc->list, task );
    kaapi_format_set_access_param(fmt, ith, task->task->sp, &a);

    /* one more external synchronisation: add bcast */
    KAAPI_ATOMIC_INCR(&task->counter);
    
    /* - allocate a new reader entry
    */
    kaapi_data_version_t* dv_reader = kaapi_threadgroup_allocate_dataversion( thgrp );
    dv_reader->asid = asid; 
    dv_reader->task = task; 
    dv_reader->ith  = ith; 
    dv_reader->addr = a.data; 
    dv_reader->view = kaapi_format_get_view_param(fmt, ith, task->task->sp);
    
    /* link it into list of copies */
    kaapi_data_version_list_add(&ver->copies, dv_reader );

    /* register it into global table */
    kaapi_threadgroup_comrecv_register( thgrp, tid, ver->tag, wc );
  }
}



/*
*/
static inline int kaapi_version_add_reader( 
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver, 
    int                   tid, 
    kaapi_data_version_t* over,
    kaapi_taskdescr_t*    task, 
    const kaapi_format_t* fmt,
    int                   ith,
    kaapi_access_t*       access
)
{
  kaapi_assert( (ver->tag != 0) || (&ver->writer == over) 
      || kaapi_memory_address_space_isequal(ver->writer.asid, over->asid) );
  kaapi_assert( over != 0);

  int retval = 0;
  
  if (&ver->writer == over) 
  {
    kaapi_assert_debug(kaapi_memory_address_space_isequal(ver->writer.asid, over->asid));

    /* the returned kaapi_data_version_t is the writer:
       - allocate a new reader entry
    */
    kaapi_data_version_t* dv_reader = kaapi_threadgroup_allocate_dataversion( thgrp );
    dv_reader->asid = over->asid; 
    dv_reader->task = task; 
    dv_reader->ith  = ith; 
    dv_reader->addr = over->addr; 
    dv_reader->view = (over->ith == -1 ? 
                            kaapi_format_get_view_param(fmt, ith, task->task->sp) 
                          : over->view);

    /* link it into list of copies */
    kaapi_data_version_list_add(&ver->copies, dv_reader );
    
    /* set the return value */
    if (ver->writer.task !=0)
      retval = 0;
    else
      retval = 1;
  }
  else {
    retval = 0;
    /* mute the task field of over to points to the last task */
    over->task = task;
    over->ith  = ith;
  }

  
  kaapi_access_t a; /* to store data access and allocate */
  a.data = over->addr;
  kaapi_format_set_access_param(fmt, ith, task->task->sp, &a);

  if (kaapi_memory_address_space_getgid(over->asid) == thgrp->localgid)
  {
    /* means writer == access not ready (writer -> read) */
    if ((ver->writer.task != 0) && (kaapi_memory_address_space_isequal(over->asid, ver->writer.asid)))
    {
      kaapi_assert_debug( (ver->writer_thread >= -1) && (ver->writer_thread < thgrp->group_size) );
      kaapi_tasklist_t* tasklist = thgrp->threadctxts[ver->writer_thread]->tasklist;
      kaapi_taskdescr_push_successor( tasklist, ver->writer.task, task );
    }
    else {
      kaapi_comrecv_t* wc = kaapi_recvcomlist_find_tag( thgrp->lists_recv[tid], ver->tag );
      kaapi_assert_debug(wc !=0);
      /* one more external synchronisation: add bcast */
      KAAPI_ATOMIC_INCR(&task->counter);
      /* push the task into the activation link of the comrecv data structure */
      kaapi_activationlist_pushback( thgrp, &wc->list, task );
    }
  }

  /* */
  return retval;
}


/* Add a new reader and update the list of tasks for all the impacted threads.
*/
int kaapi_threadgroup_version_newreader( 
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
  kaapi_assert_debug( (-1 <= tid) && (tid < thgrp->group_size) );
  
  if ((ver->writer_mode != KAAPI_ACCESS_MODE_VOID) && KAAPI_ACCESS_IS_CUMULWRITE(ver->writer_mode))
    kaapi_threadgroup_version_finalize_cw( thgrp, ver );

  /* asid for the target thread */
  kaapi_address_space_id_t asid = kaapi_threadgroup_tid2asid(thgrp, tid);
  
  /* find the data info in the map attached to the version */
  kaapi_data_version_t* tov = kaapi_version_findasid_in( ver, asid );
  
  if (tov ==0) 
  { /* no data info means no copy of the data into asid.
       - add the task description activated by the writer task on reception of data with the tag
       - it is the first reader on this asid, thus generates the tag
       - the recv data address is flagged in order to be pre-allocated before execution
    */
    kaapi_threadgroup_add_recvtask( thgrp, ver, tid, asid, task, fmt, ith, access );
    
    /* the access is not ready */
    return 0;
  }

  /* the version is already located into the same address space than those of the thread:
     - add the reader, if tov is the writer and return 0 (not ready access)
     - return value is 1 iff the access is ready
  */
  return kaapi_version_add_reader( thgrp, ver, tid, tov, task, fmt, ith, access );  
}
