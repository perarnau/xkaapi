/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
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

/*
*/
kaapi_version_t* kaapi_version_createreplicat( 
    kaapi_frame_tasklist_t* tl,
    kaapi_version_t*        master_version
)
{
#if 0
  /* assert: this is the master copy */
  kaapi_assert_debug( master_version->master->version == master_version );

  /* assert: this not the local version */
//  kaapi_assert_debug( master_version->master->tl != tl );
  
  kaapi_link_version_t* version_link;
  kaapi_version_t*      local_version;
  
  local_version = (kaapi_version_t*)kaapi_tasklist_allocate( tl, sizeof(kaapi_version_t) );
  version_link  = (kaapi_link_version_t*)kaapi_tasklist_allocate( tl, sizeof(kaapi_link_version_t) );

  local_version->last_mode     = KAAPI_ACCESS_MODE_VOID;
  local_version->master        = master_version->master;
  local_version->handle        = (kaapi_data_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_data_t) );
  local_version->handle->ptr   = kaapi_make_pointer(KAAPI_EMPTY_ADDRESS_SPACE_ID, 0);  
  local_version->handle->view  = master_version->handle->view;

  version_link->tl             = tl;
  version_link->version        = local_version;
  version_link->next           = master_version->master->next;
  master_version->master->next = version_link->next;

#if defined(KAAPI_DEBUG)
  local_version->handle        = 0;
  local_version->writer_task   = 0;
#endif

  return local_version;
#endif //
  return 0;
}


/*
*/
int kaapi_thread_insert_synchro(
    kaapi_frame_tasklist_t* tl,
    kaapi_version_t*        version, 
    kaapi_access_mode_t     m
)
{
#if 0
  kaapi_version_t* version_master = version->master->version;
  kaapi_assert_debug( version_master != version );
  
  /* if m is a write mode, only add a initial access
  */
  if (KAAPI_ACCESS_IS_WRITE(m))
  {
    kaapi_taskdescr_t* td_alloc;
    kaapi_move_arg_t*  argalloc 
        = (kaapi_move_arg_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_move_arg_t) );
    argalloc->src_data.ptr   = kaapi_make_pointer(KAAPI_EMPTY_ADDRESS_SPACE_ID, 0);
    argalloc->src_data.view  = version_master->handle->view;
    argalloc->dest           = version->handle;
    td_alloc                 = kaapi_tasklist_allocate_td_withbody( tl, 0, kaapi_taskalloc_body, argalloc );
    version->writer_task     = td_alloc;
    kaapi_tasklist_pushback_ready( tl, td_alloc);
  }
  else {
    /* else it is a read access:
       - look if the master_version/writer has a bcast task, else add one
       - add a recv task
    */
    kaapi_tasklist_t*       tl_master = version->master->tl;
    kaapi_taskdescr_t*      td_writer = version_master->writer_task;
    kaapi_activationlist_t* ltd_bcast = td_writer->u.acl.bcast;
    kaapi_taskdescr_t*      td_bcast  = 0;
    kaapi_taskdescr_t*      td_recv;
    
    kaapi_recv_arg_t*  arg_recv 
        = (kaapi_recv_arg_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_recv_arg_t) );
    /* #warning "init arg recv" */
    td_recv         = kaapi_tasklist_allocate_td_withbody( tl_master, 0, kaapi_taskrecv_body, arg_recv );

    /* find td bcast into ltd_bcast */
    if (ltd_bcast ==0)
    {
      ltd_bcast = kaapi_tasklist_allocate( tl_master, sizeof(kaapi_activationlist_t) );
      kaapi_activationlist_clear( ltd_bcast );
    }
    else 
    { /* list all ltdbcast to find a tdbcast that will send the argument of version_master */
      kaapi_activationlink_t* curr = ltd_bcast->front;
      while (curr !=0)
      {
	/* #warning "Test is bad !!!" */
        if (curr->queue == tl)
        {
          td_bcast = curr->td;
          break;
        }
        curr = curr->next;
      }
    }
    
    /* add new td_bcast if does not exist */ 
    if (td_bcast == 0)
    {
      kaapi_bcast_arg_t*  arg 
          = (kaapi_bcast_arg_t*)kaapi_tasklist_allocate(tl_master, sizeof(kaapi_bcast_arg_t) );
      /* #warning "init arg bcast" */

      kaapi_activationlink_t* link;
      
      td_bcast         = kaapi_tasklist_allocate_td_withbody( tl_master, 0, kaapi_taskbcast_body, arg );
      link             = kaapi_tasklist_allocate_al( tl_master );
      link->td         = td_bcast;
      link->queue      = tl_master;
      link->next       = 0;
      kaapi_activationlist_pushback( td_writer->u.acl.bcast, link );
    }
  }
#endif
  return 0;
}


/*
*/
int kaapi_version_invalidatereplicat( 
    kaapi_version_t*        version
)
{
#if 0
  /* invalidate */
  kaapi_link_version_t* head = version->master;
  while (head !=0)
  {
    if (head->version != version)
    {
    }
    /* garbage cell kaapi_link_version which has been allocated into head->tl */
    
    /* pass to the next */
    head = head->next;
  }
#endif
  return 0;
}
