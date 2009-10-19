/*
** xkaapi
** 
** Created on Tue Mar 31 15:21:08 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
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
#include "kaapi_stealapi.h"

/**
*/
int kaapi_steal_processor( kaapi_steal_processor_t* kpss )
{
  int i;
  
  if (KAAPI_QUEUE_EMPTY( kpss )) return 0;

  int count = KAAPI_ATOMIC_READ(&kpss->_list_request._count);
  if (count ==0) return 0;

  int count0 = count;
  kaapi_steal_request_t** requests = kpss->_list_request._request;

  CPU_ZERO( &kpss->_list_request._cpuset );
  for (i=0; i<KAAPI_MAXSTACK_STEAL; ++i)
  {
    if (requests[i] !=0)
      CPU_SET( i, &kpss->_list_request._cpuset );
  }

  kaapi_steal_context_t* current_sc = kaapi_steal_context_top( kpss );
  while ((current_sc !=0) && (count != 0))
  {
    if (current_sc->_splitter !=0)
    {
      count -= (*current_sc->_splitter)(kpss, current_sc, count, requests );
    }
    current_sc = current_sc->_next;
  }
  
  /* reply failed to remaining requests */
  for (i=0; i<KAAPI_MAXSTACK_STEAL; ++i)
  {
    if (requests[i] !=0) kaapi_thief_reply_request( current_sc, requests, i, 0 );
  }
  KAAPI_ATOMIC_WRITE(&kpss->_list_request._count, 0);  
  return count0-count;
}

