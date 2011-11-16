/*
** kaapi_mt_context_alloc.c
** xkaapi
** 
**
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
#if defined(KAAPI_USE_NUMA)
#include <numa.h>
#endif
#if !defined (_WIN32)
#include <sys/mman.h>
#endif
#include <string.h>

/** kaapi_context_alloc
    Initialize the Kaapi thread context data structure.
    The stack of the thread is organized in two parts : the first, from address 0 to sp_data,
    contains the stack of data; the second contains the stack of tasks from address task down to sp.
    Pc points on the next task to execute.

       -------------------  <- thread
       | thread internal |
       |                 | 
   |   -------------------  <- thread->data
   |   |  data[]         |
   |   |                 |
   |   |                 |
   |   |                 |
  \|/  -------------------  <- thread->sfp->sp_data
       |                 |
       |  free zone      |
       |                 |
       -------------------  <- thread->sfp->sp
  /|\  |                 |  
   |   |                 |  
   |   |                 |  <- thread->sfp->pc
   |   |                 |
   |   -------------------  <- thread->task
  
  The stack is full when sfp->sp_data == sfp->sp.
*/


/** 
*/
kaapi_thread_context_t* kaapi_context_alloc( kaapi_processor_t* kproc, size_t stacksize )
{
  kaapi_thread_context_t* ctxt;
  size_t size_data;
  size_t k_stacksize;
  size_t k_allocatestack;
  size_t pagesize, count_pages;
#if defined(KAAPI_USE_NUMA)
  int err;
#endif

  /* already allocated ? */
  if (!kaapi_lfree_isempty(kproc)) 
  {
    ctxt = kaapi_lfree_pop(kproc);
    kaapi_thread_clear(ctxt);
    return ctxt;
  }

  /* round to the nearest closest value */
  if (stacksize == (size_t)-1)
    stacksize = kaapi_default_param.stacksize;
  size_data = ((stacksize + KAAPI_MAX_DATA_ALIGNMENT -1) 
        / KAAPI_MAX_DATA_ALIGNMENT) *KAAPI_MAX_DATA_ALIGNMENT;
  
  /* allocate a thread context + stack */
#if defined (_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  pagesize = si.dwPageSize;
#else
  pagesize = getpagesize();
#endif
  /* count number of pages to allocate:
     - sizeof kaapi_context_t
     - size stackframe
     - size stack
  */
  count_pages = size_data + sizeof(kaapi_thread_context_t);
  count_pages = (count_pages + pagesize -1 ) / pagesize;
  k_allocatestack = count_pages*pagesize;

#  if defined (_WIN32)
  ctxt = (kaapi_thread_context_t*) VirtualAlloc(0, k_allocatestack, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#  elif defined(KAAPI_USE_NUMA)
  ctxt = numa_alloc_local( k_allocatestack );
#  else
  ctxt = (kaapi_thread_context_t*) mmap( 0, k_allocatestack, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, (off_t)0 );
#  endif
  if (ctxt == (kaapi_thread_context_t*)-1) 
    return 0;

#if defined(KAAPI_DEBUG) && !defined (_WIN32) /*VirtualAlloc initializes memory to zero*/
  memset(ctxt, 0, sizeof(kaapi_thread_context_t) ); 
#endif

  /* test page size alignment (4KBytes page) */
  kaapi_assert_debug( __kaapi_isaligned(ctxt, 0x1000) ==1 );

  ctxt->size = (uint32_t)k_allocatestack;
  ctxt->stack_size = (uint32_t) stacksize;


  /* should be aligned on a multiple of 64bit due to atomic read / write of pc in each kaapi_frame_t */
  ctxt->stack.stackframe = ctxt->stackframe;
//  kaapi_malloc_align(64, sizeof(kaapi_frame_t)*KAAPI_MAX_RECCALL, &ctxt->alloc_ptr);
  kaapi_assert_m( __kaapi_isaligned( &ctxt->stack.stackframe, 8), "StackFrame pointer not aligned to 64 bits boundary");

  /* force alignment of ctxt->task to be aligned on 64 bytes boundary */
  uintptr_t bottom_task = (uintptr_t)ctxt->data;
  k_stacksize = k_allocatestack - (bottom_task - (uintptr_t)ctxt);
  ctxt->stack.data = (void*)bottom_task;
  ctxt->stack.task = (kaapi_task_t*)((bottom_task + k_stacksize - sizeof(kaapi_task_t) - 0x3FUL) & ~0x3FUL);
  kaapi_assert_m( (((uintptr_t)ctxt->stack.task) & 0x3FUL)== 0, "Stack of task not aligned to 64 bytes boundary");

#if defined(KAAPI_DEBUG)
  for (int i=0; i<KAAPI_MAX_RECCALL; ++i)
    kaapi_frame_clear( &ctxt->stack.stackframe[i] );
#endif

  kaapi_thread_clear(ctxt);
  return ctxt;
}
