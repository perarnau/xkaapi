/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#if  !defined(_WIN32)
#include <sys/mman.h>
#endif

/**
*/
int kaapi_context_free(
    kaapi_processor_t* kproc, 
    kaapi_thread_context_t* ctxt 
)
{
  if (ctxt ==0) return 0;

  /* wait end of thieves on the processor */
  kaapi_synchronize_steal(kproc);

  if (kaapi_lfree_push(kproc, ctxt)) 
    return 0;

  /* this is the only vital ressource to destroy properly */
  kaapi_stack_destroy(&ctxt->stack);

//to delete stack frame pointer, but now its part of the thread data structure
//  if (ctxt->alloc_ptr !=0) free(ctxt->alloc_ptr);
#if defined (_WIN32)
  VirtualFree(ctxt, ctxt->size,MEM_RELEASE);
#elif defined(KAAPI_USE_NUMA)
    numa_free(ctxt, ctxt->size );
#else
  munmap( ctxt, ctxt->size );
#endif
  return 0;
}
