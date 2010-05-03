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


static inline void steal_sync(kaapi_stealcontext_t* stc)
{
  while (KAAPI_ATOMIC_READ(&stc->is_there_thief))
    kaapi_slowdown_cpu();
}

kaapi_taskadaptive_result_t* kaapi_get_thief_head( kaapi_stealcontext_t* stc )
{
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)stc;

  if (ta->head == NULL)
    steal_sync(stc);

  /* should be an atomic read -> 64 alignment boundary of IA32/IA64 */
  return ta->head;  
}

kaapi_taskadaptive_result_t* kaapi_get_nextthief_head( kaapi_stealcontext_t* stc, kaapi_taskadaptive_result_t* curr )
{
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)stc;
  kaapi_taskadaptive_result_t* ncurr;
  
  while (!KAAPI_ATOMIC_CAS(&ta->lock, 0, 1)) 
    kaapi_slowdown_cpu();
  ncurr = curr->next;
  if (ncurr ==0) ncurr = ta->head; /* try restarting from head */
  KAAPI_ATOMIC_WRITE(&ta->lock, 0);
  
  /* should be an atomic read -> 64 alignment boundary of IA32/IA64 */
  return ncurr;
}
