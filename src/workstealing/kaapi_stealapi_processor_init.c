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
#include <unistd.h>


int kaapi_steal_processor_init( kaapi_steal_processor_t* kpss, int index, int sz, void* staddr ) 
{ 
  KAAPI_ATOMIC_WRITE(&(kpss->_list_request._count), 0);
  { int i; for (i=0; i<KAAPI_MAXSTACK_STEAL; ++i)
    {  kpss->_list_request._request[i] = 0; }
  }
  kpss->_processor_id = 0;
  KAAPI_STEAL_PROCESSOR_SETINDEX(kpss, index);
  KAAPI_QUEUE_CLEAR( kpss ); 
  kpss->_stackaddr = kpss->_sp = (char*)(staddr);
  kpss->_stackend  = sz+(char*)(staddr);
  kpss->_state = KAAPI_PROCESSOR_S_CREATED;
  xkaapi_assert(0 == kaapi_mutex_init(&kpss->_lock, 0) );
  kaapi_writemem_barrier();

  kaapi_all_stealprocessor[ index ] = kpss;

  /* initialize my request */  
  kaapi_thief_request_init( &kpss->_request, kpss);

  return 0;
}


/**
*/
int kaapi_steal_processor_terminate( kaapi_steal_processor_t* kpss ) 
{
  /* here should reply FAIL to all posted request (if any ) */
  KAAPI_ATOMIC_WRITE(&(kpss->_list_request._count), 0);
  { int i; for (i=0; i<KAAPI_MAXSTACK_STEAL; ++i)
    {  kpss->_list_request._request[i] = 0; }
  }
  kpss->_state = KAAPI_PROCESSOR_S_TERMINATED;
  kaapi_writemem_barrier();
  kaapi_all_stealprocessor[ KAAPI_STEAL_PROCESSOR_GETINDEX(kpss) ] = 0;  
  return 0;
}
