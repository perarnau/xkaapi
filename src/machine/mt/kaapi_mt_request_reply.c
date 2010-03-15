/*
** xkaapi
** 
** Created on Tue Mar 31 15:21:00 2009
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
#include "kaapi_impl.h"
#include <stddef.h> 

#include <stddef.h>

/* compute the cache aligned size for kaapi_taskadaptive_result_t
 */
static inline size_t compute_struct_size(size_t data_size)
{
  size_t total_size = offsetof(kaapi_taskadaptive_result_t, data) + data_size;

  if (total_size & (KAAPI_CACHE_LINE - 1))
    {
      total_size += KAAPI_CACHE_LINE;
      total_size &= ~(KAAPI_CACHE_LINE - 1);
    }

  return total_size;
}

/** Implementation note:
    - only the thief_stack + signal to the thief has to be port on the machine.
    - the creation of the task to signal back the end of computation must be keept.
    I do not want to split this function in two parts (machine dependent and machine independent)
    in order to avoid 2 function calls, BUT for maintenance an inline version in kaapi_impl.h
    would be peferable.
    
    For instance if no memory is shared between both -> communication of the memory stack.
    with translation of :
      - function body
      - function splitter
      - arguments in case of an adaptive task
      - all stack pointer of parameter that should be considered as offset.
    The format of the task should give all necessary information about types used in the
    data stack.
*/
int _kaapi_request_reply
( 
  kaapi_request_t*        request, 
  kaapi_thread_context_t* retval, 
  int                     isok
)
{
  kaapi_processor_t*      kproc = request->proc;
  kaapi_assert_debug( kproc != 0 );
  kaapi_assert_debug( request != 0 );
//  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->hlrequests.count) > 0 );
  
  request->flag   = 0;
  request->status = KAAPI_REQUEST_S_EMPTY;
  KAAPI_ATOMIC_DECR( &kproc->hlrequests.count );
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->hlrequests.count) >= 0 );

#if 0
  fprintf(stdout,"%i kproc reply request to:proc=%p, @req=%p\n", kaapi_get_current_kid(), (void*)kproc, (void*)request );
  fflush(stdout);
#endif
  if (isok)
  {
    request->reply->data = retval;
    kaapi_writemem_barrier();
    request->reply->status = KAAPI_REQUEST_S_SUCCESS;
  }
  else 
  {
    kaapi_writemem_barrier();
    request->reply->status = KAAPI_REQUEST_S_FAIL;
  }
  return 0;
}
