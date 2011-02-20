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

/**  
*/
int kaapi_thread_computeready_date( 
    const kaapi_version_t* version, 
    kaapi_taskdescr_t*     task,
    kaapi_access_mode_t    m 
)
{
  if (version->last_mode == KAAPI_ACCESS_MODE_VOID)
  {
    task->date = _kaapi_max( task->date, 1);
  }
  else if (KAAPI_ACCESS_IS_CONCURRENT(m, version->last_mode))
  {
    if (KAAPI_ACCESS_IS_READ(m))
    { /* its a r */
      if (version->writer_task !=0)
      {
        task->date = _kaapi_max( task->date, 1+version->writer_task->date);
      }
      else {
        task->date = _kaapi_max( task->date, 1);
      }
    }
  }
  else if (KAAPI_ACCESS_IS_READWRITE(m)) /* rw */
  {
    if (version->last_task !=0) /* whatever is the previous task, do link */
    {
      task->date = _kaapi_max(task->date, 1+version->last_task->date);
    }
    else 
    {
      kaapi_assert_debug( version->last_mode == KAAPI_ACCESS_MODE_VOID);
      task->date = _kaapi_max( task->date, 1);
    }
  }
  else if (KAAPI_ACCESS_IS_READ(m)) /* r (rw already test) */
  { /* means previous is a w or rw or cw */
    task->date = _kaapi_max( task->date, 1+version->last_task->date);
  }
  else if (KAAPI_ACCESS_IS_WRITE(m)) /* cw or w */
  {
  }
  return 0;
}
