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
kaapi_data_t* kaapi_thread_computeready_access( 
    kaapi_tasklist_t*   tl, 
    kaapi_version_t*    version, 
    kaapi_taskdescr_t*  task,
    kaapi_access_mode_t m 
)
{
  kaapi_assert_debug( version->last_mode != KAAPI_ACCESS_MODE_VOID);
  kaapi_assert_debug( version->writer_task != 0);
  kaapi_assert_debug( version->writer_tasklist != 0);
  kaapi_assert_debug( version->handle != 0);

  if (KAAPI_ACCESS_IS_READWRITE(m))
  {
    kaapi_tasklist_push_successor( version->writer_tasklist, version->writer_task, task );
    version->writer_task = task;
    version->last_mode   = m;
    return version->handle;
  } 
  else if (KAAPI_ACCESS_IS_READ(m)) 
  {
    kaapi_tasklist_push_successor( version->writer_tasklist, version->writer_task, task );
    return version->handle;
  }
  else if (KAAPI_ACCESS_IS_WRITE(m)) /* w or rw */
  {
    /* look if it is a WAR or WAW dependencies (do not consider initial access) */
    if (kaapi_task_getbody(&version->writer_task->task) == kaapi_taskalloc_body)
    {
      kaapi_tasklist_push_successor( version->writer_tasklist, version->writer_task, task );
      version->writer_task = task;
      version->last_mode   = m;
      return version->handle;
    }
    
    /* else we have a true WAR or WAW dependencies: currently we serialize standard execution 
       The idea would be to:
        - either rename variable at low cost (i.e: fast memory allocation / liberation)
        - either rename variable when stealing the task
    */
    kaapi_tasklist_push_successor( version->writer_tasklist, version->writer_task, task );
    version->writer_task = task;
    version->last_mode   = m;
    return version->handle;
  }
#if 0 // assert is on kaapi_version_add_initialaccess
  else if (KAAPI_ACCESS_IS_CUMULWRITE(m))
  {
    /* cw has successor the finalizer (!=0) in writer_task and as last_task 
       the previous writer (not a cw) if not null !
    */
    return 0;
  }
#endif
  return 0;
}
