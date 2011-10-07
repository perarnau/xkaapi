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
  kaapi_tasklist_t* tasklist;
  int err;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;
  if (kaapi_frame_isempty(thread->stack.sfp)) return ENOENT;
  tasklist = (kaapi_tasklist_t*)malloc(sizeof(kaapi_tasklist_t));
  kaapi_tasklist_init( tasklist, thread );
  err= kaapi_thread_computereadylist( thread, tasklist  );
  kaapi_thread_tasklistready_push_init( &tasklist->rtl, &tasklist->readylist );
  kaapi_thread_tasklist_commit_ready( tasklist );
  /* NO */ /*keep the first task to execute outside the workqueue */
  tasklist->context.chkpt = 0;
  thread->stack.sfp->tasklist = tasklist;
  return err;
}


/**
*/
int kaapi_sched_clearreadylist( void )
{
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;

  kaapi_tasklist_t* tasklist = thread->stack.sfp->tasklist;

  if (tasklist != 0)
  {
    kaapi_sched_lock(&thread->stack.proc->lock);
    thread->stack.sfp->tasklist = 0;
    kaapi_sched_unlock(&thread->stack.proc->lock);
    kaapi_tasklist_destroy(tasklist);
    free( tasklist );
  }

  /* HERE: hack to do loop over SetStaticSched because memory state
     is leaved in inconsistant state.
  */
  kaapi_memory_destroy();
  kaapi_memory_init();

  return 0;
}


/** task is the top task not yet pushed.
    This function is called is after all task has been pushed into a specific frame.
 */
int kaapi_thread_computereadylist( kaapi_thread_context_t* thread, kaapi_tasklist_t* tasklist )
{
  kaapi_frame_t*          frame;
  kaapi_task_t*           task_top;
  kaapi_task_t*           task_bottom;
  
  /* assume no task list or task list is empty */
  frame    = thread->stack.sfp;
  
  /* initialize hashmap for version */
  if (thread->kversion_hm ==0)
    thread->kversion_hm = (kaapi_big_hashmap_t*)malloc( sizeof(kaapi_big_hashmap_t) );
  kaapi_big_hashmap_init( thread->kversion_hm, 0 );  
  
  /* iteration over all tasks of the current top frame thread->sfp */
  task_top    = frame->pc;
  task_bottom = frame->sp;
  while (task_top > task_bottom)
  {
    kaapi_thread_computedep_task( thread, tasklist, task_top );
    --task_top;
  } /* end while task */

  /* */
//  kaapi_thread_tasklist_print( stdout, tasklist );
  
  kaapi_big_hashmap_destroy( thread->kversion_hm );
  free( thread->kversion_hm );
  thread->kversion_hm = 0;
  return 0;
}
