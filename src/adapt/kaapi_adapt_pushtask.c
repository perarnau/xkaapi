/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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
int kaapi_thread_pushtask_adaptive(
  kaapi_thread_t* thread, 
  kaapi_adaptivetask_splitter_t splitter
)
{
  kaapi_taskadaptive_arg_t* arg;
  kaapi_task_t* task_adapt;
  kaapi_taskmerge_arg_t* merge_arg;
  kaapi_task_t* task_merge;

  /* this is the new adaptative task */
  task_adapt = kaapi_thread_toptask(thread);
  kaapi_stealcontext_t* sc 
      = kaapi_thread_pushdata_align(thread, sizeof(kaapi_stealcontext_t),sizeof(void*));

  arg = (kaapi_taskadaptive_arg_t*)kaapi_thread_pushdata(
    thread, sizeof(kaapi_taskadaptive_arg_t)
  );
  kaapi_assert_debug((sc != 0) && (arg !=0));

  merge_arg = (kaapi_taskmerge_arg_t*)kaapi_thread_pushdata
    (thread, sizeof(kaapi_taskmerge_arg_t));
  kaapi_assert_debug(merge_arg != 0);

  sc->msc        = sc; /* self pointer to detect master */
  sc->ktr        = 0;
  if ( kaapi_task_is_withpreemption(task_adapt))
  {
    kaapi_assert_debug( !(task_adapt->u.s.flag & KAAPI_TASK_S_NOPREEMPTION) );
    sc->flag = KAAPI_SC_PREEMPTION;
    /* if preemption, thief list used ... */
    KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
    sc->thieves.list.head = 0;
    sc->thieves.list.tail = 0;
  }
  else /* no preemption */
  {
    kaapi_assert_debug( !(task_adapt->u.s.flag & KAAPI_TASK_S_PREEMPTION) );
    sc->flag = KAAPI_SC_NOPREEMPTION;
    /* ... otherwise thief count */
    KAAPI_ATOMIC_WRITE(&sc->thieves.count, 0);
  }

  /* initialize the taskadapt_body args */
  kaapi_access_init(&arg->shared_sc, sc);
  arg->user_body     = task_adapt->body;
  arg->user_sp       = task_adapt->sp;
  arg->user_splitter = splitter;
  
  /* keep the same flag as the pushed task and add splittable attribut.
     Replace the body to the adaptive body
  */
  task_adapt->body = (kaapi_task_body_t)kaapi_taskadapt_body;
  task_adapt->sp   = arg;
  
  kaapi_access_init( &merge_arg->shared_sc, sc );

  task_merge = kaapi_thread_nexttask(thread, task_adapt);
  kaapi_task_init_with_flag(
      task_merge, 
      kaapi_taskadaptmerge_body, 
      merge_arg, 
      KAAPI_TASK_UNSTEALABLE /* default is also un-splittable */
  );

  /* memory barrier done by kaapi_thread_pushtask */
  return kaapi_thread_push_packedtasks(thread, 2);
}
