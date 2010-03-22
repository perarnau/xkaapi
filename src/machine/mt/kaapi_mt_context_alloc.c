/*
** kaapi_mt_context_alloc.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>

/** 
*/
kaapi_thread_context_t* kaapi_context_alloc( kaapi_processor_t* kproc )
{
  kaapi_thread_context_t* ctxt;
  kaapi_stack_t* stack;
  kaapi_uint32_t size_data;
  size_t k_stacksize;
  size_t pagesize, count_pages;

  /* already allocated ? */
  if (!KAAPI_STACK_EMPTY(&kproc->lfree)) 
  {
    ctxt = KAAPI_STACK_TOP(&kproc->lfree);
    KAAPI_STACK_POP(&kproc->lfree);
    kaapi_thread_clear( ctxt );
    return ctxt;
  }

  /* round to the nearest closest value */
  size_data = ((kaapi_default_param.stacksize + KAAPI_MAX_DATA_ALIGNMENT -1) / KAAPI_MAX_DATA_ALIGNMENT) * KAAPI_MAX_DATA_ALIGNMENT;
  
  /* allocate a thread context + stack */
  pagesize = getpagesize();
  count_pages = (size_data + sizeof(kaapi_thread_context_t) + pagesize -1 ) / pagesize;
  k_stacksize = count_pages*pagesize;
  ctxt = (kaapi_thread_context_t*) mmap( 0, k_stacksize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, (off_t)0 );
  if (ctxt == (kaapi_thread_context_t*)-1) {
    int err __attribute__((unused)) = errno;
    return 0;
  }
  memset(ctxt, 0, k_stacksize );

  stack = kaapi_threadcontext2stack(ctxt);
  if (kaapi_stack_init( stack, size_data, stack+1 ) !=0)
  {
    munmap( ctxt, ctxt->size );
    return 0;
  }

  /* should be aligned on a multiple of 64bit due to atomic read / write of pc in each kaapi_frame_t */
  ctxt->stackframe = malloc(sizeof(kaapi_frame_t)*KAAPI_MAX_RECCALL);
  kaapi_assert_m( (((kaapi_uintptr_t)ctxt->stackframe) & 0x3F), 0, "StackFrame pointer not aligned to 64 bit boundary");
  if (ctxt->stackframe ==0) {
    munmap( ctxt, ctxt->size );
    return 0;
  }

  kaapi_thread_clear(ctxt);
#if (KAAPI_USE_STEALFRAME_METHOD == KAAPI_STEALTHE_METHOD)
  ctxt->thieffp = 0;
#endif
#if (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALTHE_METHOD)
  ctxt->thiefpc = 0;
#endif
  return ctxt;
}
