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
int kaapi_version_add_initialaccess( 
    kaapi_version_t*           version, 
    kaapi_frame_tasklist_t*    tl,
    kaapi_access_mode_t        m,
    void*                      data, 
    const kaapi_memory_view_t* view 
)
{
  kaapi_assert_debug(version->last_mode == KAAPI_ACCESS_MODE_VOID)

  version->handle = (kaapi_data_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_data_t) );
  version->handle->ptr  = kaapi_make_pointer(KAAPI_EMPTY_ADDRESS_SPACE_ID, 0);  
  version->handle->view = *view;
  
  if (KAAPI_ACCESS_IS_READ(m))
  {
    /* push a move task to insert the data into the task list */
    kaapi_taskdescr_t* td_move;
    kaapi_move_arg_t*  argmove 
        = (kaapi_move_arg_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_move_arg_t) );
    kaapi_task_t* task_move = kaapi_tasklist_allocate_task( tl, kaapi_taskmove_body, argmove );
    argmove->src_data.ptr    = kaapi_make_pointer(KAAPI_EMPTY_ADDRESS_SPACE_ID, data);
    argmove->src_data.view   = *view;
    argmove->dest            = version->handle;
    td_move                  = kaapi_tasklist_allocate_td( tl, task_move, 0);
//    td_move                  = kaapi_tasklist_allocate_td_withbody( tl, 0, kaapi_taskmove_body, argmove);
    version->writer_task     = td_move;

    kaapi_frame_tasklist_pushback_ready( tl, td_move);
  }
  else if (KAAPI_ACCESS_IS_WRITE(m))
  {
    kaapi_taskdescr_t* td_alloc;
    kaapi_move_arg_t*  argalloc 
        = (kaapi_move_arg_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_move_arg_t) );
    kaapi_task_t* task_alloc = kaapi_tasklist_allocate_task( tl, kaapi_taskalloc_body, argalloc );
    argalloc->src_data.ptr   = kaapi_make_pointer(KAAPI_EMPTY_ADDRESS_SPACE_ID, data);
    argalloc->src_data.view  = *view;
    argalloc->dest           = version->handle;
    td_alloc                 = kaapi_tasklist_allocate_td( tl, task_alloc, 0);
//    td_alloc                 = kaapi_tasklist_allocate_td_withbody( tl, 0, kaapi_taskalloc_body, argalloc );
    version->writer_task     = td_alloc;
    kaapi_frame_tasklist_pushback_ready( tl, td_alloc);
  } else {
    /* not yet implemented || KAAPI_ACCESS_IS_CUMULWRITE(m)) */
    kaapi_assert(0);
  }

  version->last_mode = KAAPI_ACCESS_MODE_W;

  return 0;
}
