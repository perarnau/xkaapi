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
#include <stddef.h> /* offsetof */

int kaapi_preempt_nextthief_helper( 
  kaapi_stealcontext_t*        stc, 
  kaapi_taskadaptive_result_t* ktr, 
  void*                        arg_to_thief 
)
{
  kaapi_taskadaptive_t* ta;
  if (ktr ==0) return 0;

  /* container_of */
  ta = (kaapi_taskadaptive_t*)((char *)stc - offsetof(kaapi_taskadaptive_t, sc));

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
  
  /* TODO: Here it should be:
     - replace ktr in the list of ta by the thives of ktr 
     but it required to link ktr to the stealcontext used by the thief....
  */
  while (!KAAPI_ATOMIC_CAS(&ta->lock, 0, 1)) 
    ;
    if (ktr->next ==0) /* it's on tail */
      ta->tail = ktr->prev;
    else  
      ktr->next->prev = ktr->prev;
    if (ktr->prev ==0) /* it's on head */
      ta->head = ktr->next;
    else 
      ktr->prev->next = ktr->next;
    /* mostly for debug: */
    ktr->next = 0;
    ktr->prev = 0;

  KAAPI_ATOMIC_WRITE(&ta->lock, 0);
  
  return 1;
}
