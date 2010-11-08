/*
** kaapi_task_preemptpoint.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
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
int kaapi_preemptpoint_before_reducer_call( 
    kaapi_stealcontext_t* stc,
    void* arg_for_victim, 
    void* result_data, 
    int result_size
)
{
  kaapi_taskadaptive_result_t* const ktr = stc->header.ktr;

  kaapi_assert_debug( ktr != 0 );

  /* disable and wait no more thief on stc */
  kaapi_steal_disable_sync( stc );

  /* recopy data iff its non null */
  if (result_data !=0)
  {
    if (result_size < ktr->size_data) 
      ktr->size_data = result_size;

    if (ktr->size_data >0)
      memcpy(ktr->data, result_data, ktr->size_data);
  }
  /* push data to the victim and list of thief */
  ktr->arg_from_thief = arg_for_victim;

  /* no lock needed since no more steal possible */
  ktr->rhead = stc->thieves.list.head;
  stc->thieves.list.head = 0;

  ktr->rtail = stc->thieves.list.tail;
  stc->thieves.list.tail = 0;
  
  return 0;
}


/**
*/
int kaapi_preemptpoint_after_reducer_call( 
    kaapi_stealcontext_t* stc,
    int reducer_retval __attribute__((unused))
)
{
  kaapi_taskadaptive_result_t* const ktr = stc->header.ktr;
  uintptr_t state;

  kaapi_assert_debug( ktr != 0 );

  /* serialize previous line with next line */
  kaapi_writemem_barrier();

  /* signal termination */
  state = kaapi_task_orstate(&ktr->state, KAAPI_MASK_BODY_TERM);
  if (state & KAAPI_MASK_BODY_PREEMPT)
  {
    /* @see comment in kaapi_task_adap_body */
    while (*ktr->preempt == 0)
      kaapi_slowdown_cpu();
  }

  /* adapt_body needs to know about preemption */
  stc->header.ktr = 0;

  return 1;
}
