/*
** kaapi_task_preempt_next.c
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

int kaapi_preempt_nextthief_helper( 
  kaapi_stealcontext_t*        stc, 
  kaapi_taskadaptive_result_t* ktr, 
  void*                        arg_to_thief
)
{
  kaapi_taskadaptive_t* ta;
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint64_t t1;
  kaapi_uint64_t t0;
#endif
  if (ktr ==0) return 0;

#if defined(KAAPI_USE_PERFCOUNTER)
  t0 = kaapi_get_elapsedns();
#endif
  ta = (kaapi_taskadaptive_t*)stc;

  /* pass arg to the thief */
  ktr->arg_from_victim = arg_to_thief;  
  kaapi_writemem_barrier();

  ktr->req_preempt = 1;
  kaapi_mem_barrier();
    
  /* busy wait thief receive preemption */
  while (!ktr->thief_term) 
   ;/* pthread_yield(); */

  /*ok: here thief has been preempted */
  kaapi_readmem_barrier();
  
  while (!KAAPI_ATOMIC_CAS(&ta->lock, 0, 1)) 
    ;

  if (ktr->rhead != NULL)
  {
    /* rtail non null too */
    ktr->rhead->prev = ktr->prev;
    ktr->rtail->next = ktr->next;

    if (ktr->prev != NULL)
      ktr->prev->next = ktr->rhead;
    else
      ta->head = ktr->rhead;

    if (ktr->next != NULL)
      ktr->next->prev = ktr->rtail;
    else
      ta->tail = ktr->rtail;
  }
  else /* no thieves, unlink */
  {
    if (ktr->prev != NULL)
      ktr->prev->next = ktr->next;
    else
      ta->head = ktr->next;

    if (ktr->next != NULL)
      ktr->next->prev = ktr->prev;
    else
      ta->tail = ktr->prev;
  }

  /* mostly for debug: */
  ktr->next = 0;
  ktr->prev = 0;

  KAAPI_ATOMIC_WRITE(&ta->lock, 0);
#if defined(KAAPI_USE_PERFCOUNTER)
  t1 = kaapi_get_elapsedns();
  stc->ctxtthread->proc->t_preempt += (double)(t1-t0)*1e-9;
  printf("Delay preempt:%15f, Total=%15f\n", (double)(t1-t0)*1e-9, stc->ctxtthread->proc->t_preempt );
#endif
  
  return 1;
}
