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


/* Add a new writer and update the list of tasks for all the impacted threads.
*/
int kaapi_threadgroup_version_newwriter( 
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
  void*  data;
  kaapi_memory_view_t view;
  int retval;
  kaapi_assert_debug( (-1 <= tid) && (tid < thgrp->group_size) );
  kaapi_assert_debug( !KAAPI_ACCESS_IS_CUMULWRITE(mode) );

  /* retval = 1: assume access is ready */
  retval = 1;
  
  /* asid for the target thread */
  kaapi_address_space_t asid = thgrp->tid2asid[tid];

  /* find the data info in the map attached to the version and remove it from list if found, else return 0 */
  kaapi_data_version_t* dv = kaapi_version_findcopiesrmv_asid_in( ver, asid );
  if (dv !=0) 
  {
    data = dv->addr;
    view = dv->view;
    dv->addr = 0;
    /* recycle the data version */
    kaapi_threadgroup_deallocate_dataversion(thgrp, dv);
    
    /* avoid WAR here: may be solved at runtime */
    retval = 0;
    if (dv->task != task)
    {
      kaapi_tasklist_t* tasklist = thgrp->threadctxts[tid]->tasklist;
      kaapi_taskdescr_push_successor( tasklist, dv->task, task );
    }
  }
  else 
  {
    dv = kaapi_version_findtodelrmv_asid_in( ver, asid );
    view = kaapi_format_get_view_param(fmt, ith, task->task->sp);
    if ((dv !=0) 
     && (kaapi_memory_view_size(&dv->view) == kaapi_memory_view_size(&view)) )
    {
      data = dv->addr;
      dv->addr = 0;
      /* recycle the data version */
      kaapi_threadgroup_deallocate_dataversion(thgrp, dv);
      
    /* avoid WAR here: may be solved at runtime */
      retval = 0;

      if (dv->task !=0)
      {
        kaapi_tasklist_t* tasklist = thgrp->threadctxts[tid]->tasklist;
        kaapi_taskdescr_push_successor( tasklist, dv->task, task );
      }
    }
    else 
      data = malloc(kaapi_memory_view_size(&view))  ;
  }
  
  /* data already exist: multiple cases must be considered.
     1- if it exists readers (>=1), then copies are invalidated 
        - Else if only one reader exists into the asid then the WAR dependency is solved at runtime if the
        task is stolen.
      2- if the accesses are not reader but writers (currently ==1 writer because we do not consider cw)
        - currently all other accesses are not in the same address space because we only support one thread per address
        space, so the WAW dependencies is solved.
        The copies are marked 'deleted'.
      3- All readers are deleted and copies, except in the same asid, are deleted.

  */
  /* mark copies as deleted except in asid for thread tid 
  */
  kaapi_data_version_list_append( &ver->todel, &ver->copies );

  /* update the writer of the version */
  ver->writer.asid   = asid;
  ver->writer.task   = task;
  ver->writer.ith    = ith;

  ver->writer.addr   = data;
  ver->writer.view   = view;
  ver->writer.next   = 0;
  ver->writer_thread = tid;

  kaapi_access_t a;          /* to store data access and allocate */
  a.data = ver->writer.addr;
  kaapi_format_set_access_param(fmt, ith, task->task->sp, &a);

  /* reset the tag for the version */
  ver->tag = 0;

  /* return 1: the access is ready ! */
  return retval;
}