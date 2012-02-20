/*
** kaapi_task_finalize.c
** xkaapi
** 
**
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


/**
*/
int kaapi_task_end_adaptive( void* arg )
{
  kaapi_task_t* task_merge;
  kaapi_task_t* task_adapt;
  kaapi_taskbegendadaptive_arg_t* adap_arg;
  kaapi_taskmerge_arg_t* merge_arg;
  kaapi_thread_context_t* const self_thread = kaapi_self_thread_context();
  kaapi_thread_t* const thread = kaapi_threadcontext2thread(self_thread);

  task_adapt = (kaapi_task_t*)arg;
  adap_arg = kaapi_task_getargst(task_adapt, kaapi_taskbegendadaptive_arg_t);

/* TODO: clear task stealable & splittable attribute first ? 
*/
//  kaapi_task_set_unstealable( task_adapt );
//  kaapi_task_unset_splittable( task_adapt ); 
  
  /* create the merge task : avoid to push the task_adapt in order
     to avoid its visibility before creation of the merge task.
     - the merge task has the SC structure as parameter, not the task_adapt
  */
  merge_arg = (kaapi_taskmerge_arg_t*)kaapi_thread_pushdata
    (thread, sizeof(kaapi_taskmerge_arg_t));
  kaapi_assert_debug(merge_arg != 0);

  kaapi_access_init(&merge_arg->shared_sc, adap_arg->shared_sc.data);

  task_merge = kaapi_thread_toptask(thread);
  kaapi_task_init_with_flag(
      task_merge, 
      kaapi_taskadaptmerge_body, 
      merge_arg, 
      KAAPI_TASK_UNSTEALABLE /* default is also un-splittable */
  );

  /* memory barrier done by kaapi_thread_pushtask */
  kaapi_thread_pushtask(thread);

  /* force execution of all previously pushed task of the frame */
  kaapi_sched_sync_(self_thread);
  kaapi_thread_pop_frame();

  /* force execution of all previously pushed task of the frame */
  kaapi_task_markterm(task_adapt); 

#if defined(KAAPI_DEBUG)
  kaapi_stealcontext_t* sc = (kaapi_stealcontext_t*)merge_arg->shared_sc.data;
  kaapi_assert_debug( KAAPI_ATOMIC_READ( &sc->thieves.count ) == 0 );
  sc->state = 0;
#endif

  return 0;
}
