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
  kaapi_taskmerge_arg_t* const merge_arg = (kaapi_taskmerge_arg_t*)arg;
  kaapi_thread_context_t* const self_thread = kaapi_self_thread_context();
  kaapi_thread_t* const thread = kaapi_threadcontext2thread(self_thread);

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

  /* made visible old frame if not empty */
  if (!kaapi_frame_isempty(&merge_arg->saved_frame))
  {
    kaapi_thread_restore_frame(thread, &merge_arg->saved_frame);
  }
  /* Synchronize with thief to avoid ABA probem if current thread
     re-push tasks at the same address of the poped frame
  */
  kaapi_synchronize_steal(self_thread);

  return 0;
}
