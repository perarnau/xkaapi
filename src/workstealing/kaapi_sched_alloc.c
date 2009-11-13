/*
** xkaapi
** 
** Created on Tue Mar 31 15:17:57 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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
#include <sys/types.h>
#include <sys/mman.h>

/** 0 if always for the main thread
*/
static kaapi_atomic_t count_tid[4] = {
  { 0 },  /* 00 -> process scope */
  { 0 },  /* 01 -> processor scope */
  { 0 },  /* unused */
  { 0 }   /* 11 -> system scope */
};

static kaapi_uint32_t BIT_NAMESPACE[] = {
  0U,
  1U << 30,
  2U << 30,
  3U << 30  
};

/**
*/
kaapi_thread_id_t kaapi_allocate_tid(int type)
{
  kaapi_uint32_t tid = KAAPI_ATOMIC_INCR( &count_tid[type] ) || BIT_NAMESPACE[type];
  return tid;
}

/**
*/
kaapi_thread_processor_t** kaapi_allocate_processors( int kproc, cpu_set_t cpuset)
{
  int i;
  
  kaapi_thread_processor_t** kap = malloc( sizeof(kaapi_thread_processor_t*) * kproc );

  for (i=0; i<kproc; ++i)
  {
    kaapi_thread_processor_t* proc = kaapi_allocate_processor();
    kap[i] = proc;
  }
  return kap;
}

/**
*/
kaapi_thread_processor_t* kaapi_allocate_processor()
{
  kaapi_thread_descr_t* proc = kaapi_allocate_thread_descriptor( 
        KAAPI_PROCESSOR_SCOPE, 
        1, 
        default_param.stacksize, 
        default_param.stacksize );

  return &proc->th.k;
}


/** deallocate the processors
*/
void kaapi_deallocate_processor(kaapi_thread_processor_t** procs, int kproc)
{
  int i;
  for (i=0; i<kproc; ++i)
  {
    char* td = (char*)procs[i];
    if (td !=0)
    {
      td -= offsetof(kaapi_thread_descr_t, th.k);
      kaapi_deallocate_thread_descriptor( (kaapi_thread_descr_t*)td );
    }
  }
}


/** allocate a thread descriptor on a given processor
    May be extend kprocessor to fill some byte between struct kaapi_thread_descr_t and a alignment constraint to put
    data specific array ?
*/
kaapi_thread_descr_t* kaapi_allocate_thread_descriptor( 
  int scope, int detachstate, size_t c_stacksize, size_t k_stacksize,
  void*(*start_routine)(void *), void * arg  
)
{
  size_t pagesize;
  size_t count_pages;
  kaapi_thread_descr_t* td;
  size_t k_sizetask;
  size_t k_sizedata;
  char* buffer;

  kaapi_assert_debug( (scope == KAAPI_PROCESS_SCOPE) || (scope -= KAAPI_SYSTEM_SCOPE) || (scope == KAAPI_PROCESSOR_SCOPE) );

  if (scope == KAAPI_PROCESSOR_SCOPE)
    detachstate = 1;
    
  if (k_stacksize < 256) return 0;
  k_sizetask = k_stacksize / 4;
  k_sizedata = k_stacksize - k_sizetask;
              
  pagesize = getpagesize();
  count_pages = (c_stacksize + k_stacksize + sizeof(kaapi_thread_descr_t) + pagesize -1 ) / pagesize;
  td = (kaapi_thread_descr_t*)mmap( 0, count_pages*pagesize, PROT_READ|PROT_WRITE, MAP_ANON, -1, 0 );
  kaapi_assert(td !=(kaapi_thread_descr_t*)-1); 

  td->state          = KAAPI_THREAD_S_ALLOCATED;
  td->scope          = scope;
  td->pagesize       = count_pages;
  td->affinity       = (kaapi_uint16_t)-1;
  if (detachstate !=0)
    td->futur          = 0;
  else {
    td->futur = calloc(1, sizeof(kaapi_thread_futur_t));
    kaapi_assert( td->futur !=0 );
  }

  buffer = (char*)td;
  buffer += sizeof(kaapi_thread_descr_t)+c_stacksize;
  
  switch (scope )
  {
    case KAAPI_PROCESS_SCOPE:
      td->th.p.proc = 0;
    break;

    case KAAPI_PROCESSOR_SCOPE:
      td->th.k.ctxt.tid = kaapi_allocate_tid(KAAPI_TYPE_PROCESSOR_SCOPE);
      td->th.k.ctxt.flags = KAAPI_CONTEXT_SAVE_CSTACK|KAAPI_CONTEXT_SAVE_KSTACK;
      td->th.k.ctxt.dataspecific = 0;
      td->th.k.ctxt.cstacksize = c_stacksize;
      td->th.k.ctxt.cstackaddr = td+1;       /* page align ? */
      kaapi_assert( 0 == kaapi_stack_init( &td->th.k.ctxt.kstack, k_sizetask, buffer, k_sizedata, buffer + k_sizetask ) );

      KAAPI_FIFO_CLEAR( &td->th.k.ready_threads);
      KAAPI_FIFO_CLEAR( &td->th.k.suspended_threads);

    case KAAPI_SYSTEM_SCOPE:
      td->th.s.tid = kaapi_allocate_tid(KAAPI_TYPE_SYSTEM_SCOPE);
      kaapi_assert( 0 == kaapi_stack_init( &td->th.s.kstack, k_sizetask, buffer, k_sizedata, buffer + k_sizetask ) );
      td->th.s.dataspecific   = 0;
      td->th.s.arg_entrypoint = 0;
      td->th.s.entrypoint     = 0;
      kaapi_assert( 0 == pthread_cond_init( &td->th.s.cond,0) );
    break;
  }
  
  return td;
}


/** deallocate a thread descriptor on a given processor
*/
void kaapi_deallocate_thread_descriptor( struct kaapi_thread_descr_t* thread )
{
  munmap(thread, thread->pagesize * getpagesize());
}

