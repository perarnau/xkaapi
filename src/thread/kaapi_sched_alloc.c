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

/**
*/
kaapi_processor_t** kaapi_allocate_processors( int kproc, cpu_set_t cpuset)
{
  int i;
  
  kaapi_processor_t** kap = malloc( sizeof(kaapi_processor_t*) * kproc );

  for (i=0; i<kproc; ++i)
  {
    kaapi_processor_t* proc = malloc( getpagesize() );
    proc->_the_steal_processor = (kaapi_steal_processor_t*)(((char*)proc)+sizeof(kaapi_processor_t));
    kaapi_steal_processor_init(proc->_the_steal_processor, KAAPI_ATOMIC_INCR(&kaapi_index_stacksteal)-1, 
        getpagesize()-(sizeof(kaapi_steal_processor_t) + sizeof(kaapi_processor_t)), 
        sizeof(kaapi_steal_processor_t)+ ((char*)proc->_the_steal_processor));

    /* initialize the steal context of threads */
    kaapi_steal_thread_context_init( &proc->_sc_thread );
    
    /* push it on top of the stack of the kaapi_steal_processor_t object */
    kaapi_steal_context_push( proc->_the_steal_processor, &proc->_sc_thread._sc, &kaapi_sched_steal_sc_thread );

    /* push it on top of the stack of the kaapi_steal_processor_t object */
    kaapi_steal_context_push( proc->_the_steal_processor, &proc->_sc_inside, &kaapi_sched_steal_sc_inside );
    
    kap[i] = proc;
  }
  return kap;
}

/**
*/
kaapi_processor_t* kaapi_allocate_processor()
{
  kaapi_processor_t* proc = malloc( getpagesize() );
  proc->_the_steal_processor = (kaapi_steal_processor_t*)(((char*)proc)+sizeof(kaapi_processor_t));
  kaapi_steal_processor_init(proc->_the_steal_processor, KAAPI_ATOMIC_INCR(&kaapi_index_stacksteal)-1, 
      getpagesize()-(sizeof(kaapi_steal_processor_t) + sizeof(kaapi_processor_t)), 
      sizeof(kaapi_steal_processor_t)+ ((char*)proc->_the_steal_processor));

  /* initialize the steal context of threads */
  kaapi_steal_thread_context_init( &proc->_sc_thread );
  
  /* push it on top of the stack of the kaapi_steal_processor_t object */
  kaapi_steal_context_push( proc->_the_steal_processor, &proc->_sc_thread._sc, &kaapi_sched_steal_sc_thread );

  /* push it on top of the stack of the kaapi_steal_processor_t object */
  kaapi_steal_context_push( proc->_the_steal_processor, &proc->_sc_inside, &kaapi_sched_steal_sc_inside );

  return proc;
}


/** deallocate the processors
*/
void kaapi_deallocate_processor(kaapi_processor_t** procs, int kproc)
{
  int i;
  for (i=0; i<kproc; ++i)
  {
    free(procs[i]);
  }
}


/** allocate a thread descriptor on a given processor
*/
struct kaapi_thread_descr_t* allocate_thread_descriptor( int scope, int detachstate )
{
  kaapi_thread_descr_t* td;
  td = (kaapi_thread_descr_t*)malloc( sizeof(struct kaapi_thread_descr_t) );

  td->_state = KAAPI_THREAD_ALLOCATED;
  td->_run_entrypoint = 0;
  td->_td             = td;
  td->_stackaddr      = 0;
  td->_stacksize      = 0;
  td->_key_table      = 0;
  td->_next           = 0;
  
  /* init its conditions, mutex */
  if (scope != KAAPI_PROCESS_SCOPE)
    xkaapi_assert ( 0 == pthread_cond_init(&td->_cond, 0) );

  if (detachstate ==0)
  {
    xkaapi_assert ( 0 == pthread_mutex_init(&td->_mutex_join, 0) );
    xkaapi_assert ( 0 == pthread_cond_init(&td->_cond_join, 0) );
  }
  return td;
}


/** deallocate a thread descriptor on a given processor
*/
void deallocate_thread_descriptor( struct kaapi_processor_t* proc, struct kaapi_thread_descr_t* thread )
{
#if !defined(KAAPI_USE_SCHED_AFFINITY)
  free(thread);
#else
#endif  
}

/*
*/
void kaapi_workqueue_alloc_bloc(void* ptr )
{
  kaapi_workqueue_head_t* head = (kaapi_workqueue_head_t*)ptr;
  
  kaapi_workqueue_bloc_t* bloc = malloc( 4096 ); /* TO DEFINE AS CONSTANT DURING CONFIGURATION */
  bloc->_top = bloc->_bottom = 0;
  bloc->_nextbloc = 0;
  bloc->_prevbloc = head->_bottom_bloc;
  if (head->_bottom_bloc != 0)
    head->_bottom_bloc->_nextbloc = bloc;
  
  head->_bottom_bloc = bloc;
  kaapi_writemem_barrier();
  if (head->_first_bloc == 0) 
    head->_first_bloc = head->_top_bloc = bloc;

  return;
}

