/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 **
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
#include <inttypes.h>

/**
*/
int kaapi_sched_computereadylist( void )
{
  kaapi_frame_tasklist_t* frame_tasklist;
  int err;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  if (thread ==0) return EINVAL;
  if (kaapi_frame_isempty(thread->stack.sfp)) return ENOENT;
  frame_tasklist = (kaapi_frame_tasklist_t*)malloc(sizeof(kaapi_frame_tasklist_t));
  kaapi_frame_tasklist_init( frame_tasklist, thread );
  
  err= kaapi_thread_computereadylist( thread, frame_tasklist  );
  kaapi_readytasklist_push_from_activationlist( &frame_tasklist->tasklist.rtl, frame_tasklist->readylist.front );

  thread->stack.sfp->tasklist = &frame_tasklist->tasklist;
  return err;
}



/** task is the top task not yet pushed.
    This function is called is after all task has been pushed into a specific frame.
 */
double _kaapi_time_tasklist;
int kaapi_thread_computereadylist( kaapi_thread_context_t* thread, kaapi_frame_tasklist_t* frame_tasklist )
{
  kaapi_frame_t*          frame;
  kaapi_task_t*           task_top;
  kaapi_task_t*           task_bottom;

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_counter_t    t1,t0;
#endif
  
#if defined(KAAPI_USE_PERFCOUNTER)
  t0 = kaapi_get_elapsedns();
#endif

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
    kaapi_thread_computedep_task( thread, frame_tasklist, task_top );
    --task_top;
  } /* end while task */

  /* Here compute the apriori minimal date of execution */
  if (kaapi_default_param.ctpriority)
  {
    
    KAAPI_DEBUG_INST(double t0 =) kaapi_get_elapsedtime();
    kaapi_tasklist_critical_path( frame_tasklist );  
    KAAPI_DEBUG_INST(double t1 =) kaapi_get_elapsedtime();
//    kaapi_frame_tasklist_print( stdout, frame_tasklist );
    KAAPI_DEBUG_INST(printf("Criticalpath = %" PRIu64 "\n Time criticalpath:%f\n", frame_tasklist->tasklist.t_infinity, t1-t0);)
  }
  
  kaapi_big_hashmap_destroy( thread->kversion_hm );
  free( thread->kversion_hm );
  thread->kversion_hm = 0;

//printf("Tinf: %lu\n", tasklist->t_infinity );

#if defined(KAAPI_USE_PERFCOUNTER)
  t1 = kaapi_get_elapsedns();
  _kaapi_time_tasklist = 1e-9 * (double)(t1-t0);
  KAAPI_PERF_REG_SYS(thread->stack.proc, KAAPI_PERF_ID_TASKLISTCALC) += t1-t0;
#endif

  return 0;
}
