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
int kaapi_sched_computereadylist( void )
{
  int err;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;
  err= kaapi_thread_computereadylist( thread );
  return err;
}


/** task is the top task not yet pushed.
    This function is called is after all task has been pushed into a specific frame.
 */
int kaapi_thread_computereadylist( kaapi_thread_context_t* thread )
{
  kaapi_frame_t*          frame;
  kaapi_task_t*           task_top;
  kaapi_task_t*           task_bottom;
  
  /* assume no task list or task list is empty */
  frame    = thread->sfp;
  
  /* iteration over all tasks of the current top frame thread->sfp */
  task_top    = frame->pc;
  task_bottom = frame->sp;
  while (task_top > task_bottom)
  {
    kaapi_thread_computedep_task( thread, frame, task_top );
    --task_top;
  } /* end while task */

#if defined(KAAPI_USE_PERFCOUNTER)
  printf("[tasklist] #task:%lu\n", cnt_tasks);
#endif

  return 0;
}
