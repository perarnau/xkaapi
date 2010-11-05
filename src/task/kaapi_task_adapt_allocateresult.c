/*
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

/*
*/
kaapi_taskadaptive_result_t* kaapi_allocate_thief_result(
  kaapi_request_t* kreq, int size, void* data
)
{
  kaapi_taskadaptive_result_t* result;
  void* addr_tofree;
  size_t size_alloc;
  
  /* allocate space for futur result of size size
     kaapi_taskadaptive_result_t has correct alignment
  */
  size_alloc = sizeof(kaapi_taskadaptive_result_t);
  if ((size >0) && (data ==0)) size_alloc += size;
  result = (kaapi_taskadaptive_result_t*)kaapi_malloc_align
    ( KAAPI_CACHE_LINE, size_alloc, &addr_tofree );
  if (result== 0) return 0;
  
  result->size_data = size;
  if ((size >0) && (data ==0)) {
    result->flag = KAAPI_RESULT_DATARTS;
    result->data = (void*)((uintptr_t)result + sizeof(*result));
  }
  else {
    result->flag = KAAPI_RESULT_DATAUSR;
    result->data = data;
  }

  result->arg_from_victim = 0;
  result->rhead           = 0;
  result->rtail           = 0;
  result->prev            = 0;
  result->next            = 0;
  result->addr_tofree	  = addr_tofree;
  result->status	  = &kreq->reply->status;
  result->preempt	  = &kreq->reply->preempt;
  result->state.u.state	  = 0;

  return result;
}
