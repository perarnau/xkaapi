/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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

int kaapi_sched_splittask
(
  const kaapi_format_t*         task_fmt,
  kaapi_task_t*                 task,
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_adaptivetask_splitter_t splitter; 

  /* only steal into an correctly initialized steal context */
  uintptr_t orig_state = kaapi_task_getstate(task);

  /* Only split if task is INIT or EXEC.
     The victim may synchronize with the end of thief 
     on the lock of the current thread. 
  */  
  if ( (orig_state == KAAPI_TASK_STATE_INIT) 
    || (orig_state == KAAPI_TASK_STATE_EXEC) )
  {
    splitter = kaapi_format_get_splitter( task_fmt, task->sp );
    if ((splitter !=0) && kaapi_task_is_splittable( task ))
    {
      int err = splitter( task, task->sp, lrequests, lrrange );
      if (err == ECHILD)
      { /* mark the task as terminated: it is of the responsability of the splitter to ensure that */
        kaapi_task_markterm(task); 
        return 0;
      }
      return 0;
    }
    return EPERM;
  }
  return 0;
}
