/*
** kaapi_thread_create.c
** xkaapi
** 
** Created on Tue Mar 31 15:16:47 2009
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

#if defined(KAAPI_USE_SETJMP)

#if defined(KAAPI_USE_APPLE) || defined(KAAPI_USE_IPHONEOS) /* DARWING */

#if defined(KAAPI_USE_ARCH_PPC)
/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#define KAAPI_SAVE_SPACE (16)
#define xkaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[0]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE) & ~0x1f;\
  ((unsigned long*)jb)[21] = (unsigned long)f;

#define get_sp() \
({ \
  register unsigned long sp asm("r1"); \
  sp; \
})

#elif defined(KAAPI_USE_ARCH_ARM)
/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#define KAAPI_SAVE_SPACE (16)
#define xkaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[7]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE) & ~0x1f;\
  ((unsigned long*)jb)[8] = ((unsigned long)f);

#define get_sp() \
({ \
  register unsigned long sp asm("r13"); \
  sp; \
})


 /* */
#elif defined(KAAPI_USE_ARCH_X86)

/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#define KAAPI_SAVE_SPACE (16+4)/*l ajout de 4+4*nb_arg une fois 
                                 l allignement sur 16 est effectue 
                                 pour l emplacement de l adresse de retour*/
#define xkaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[8]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE);\
  ((unsigned long*)jb)[9]  = ((unsigned long*)jb)[8];\
  ((unsigned long*)jb)[10] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[14] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[15] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[17] = (unsigned long)0x37;\
  ((unsigned long*)jb)[12] = (unsigned long)f;

#define get_sp() \
({ \
  register unsigned long sp asm("esp"); \
  sp; \
})

#endif /* PPC and ARM and IA32  */
#endif /* KAAPI_USE_APPLE) || defined(KAAPI_USE_IPHONEOS */
#endif /* KAAPI_USE_SETJMP */


/*
*/
static void* kaapi_start_process_handler(void *arg);


/*
*/
static void kaapi_dataspecific_destructor (kaapi_t td);


/*
*/
int kaapi_create(kaapi_t *__restrict thread, const kaapi_attr_t *__restrict attr, void*(*start_routine)(void *), void *__restrict arg)
{
  int err;
  
  if (thread ==0) return EINVAL;
  if (start_routine ==0) return EINVAL;

  if (attr ==0)
    attr = &kaapi_default_attr;

  if ((attr->_stackaddr != 0) && (attr->_stacksize ==0)) return EINVAL;
  
  kaapi_processor_t* running_proc = (kaapi_processor_t*)pthread_getspecific( kaapi_current_processor_key );
  kaapi_thread_descr_t* td = allocate_thread_descriptor(attr->_scope, attr->_detachstate);
  

  /* initialize the fields */
  td->_state          = KAAPI_THREAD_CREATED;
  td->_detachstate    = attr->_detachstate;
  td->_scope          = attr->_scope;
  td->_cpuset         = attr->_cpuset;
  td->_stacksize      = attr->_stacksize;
  td->_stackaddr      = attr->_stackaddr;
  td->_run_entrypoint = start_routine;
  td->_arg_entrypoint = arg;

  if (td->_stackaddr == 0)
  {
    if (td->_stacksize ==0) td->_stacksize = default_param.stacksize;
    /* WARNING : FAST ALLOC */
    td->_stackaddr = malloc( td->_stacksize ); 
  }
  
  if ((td->_scope == KAAPI_SYSTEM_SCOPE)|| (td->_scope == KAAPI_PROCESSOR_SCOPE))
  {
    /* set attr to the posix thread */
    pthread_attr_t posix_attr;
    xkaapi_assert ( 0 == pthread_attr_init( &posix_attr ));
    if (td->_stackaddr !=0) 
    {
#if defined(KAAPI_USE_APPLE)
      /* stack grows down, so offset the base pointer to the last position.
         It is a bogus implementation on darwin ?
         It seems that darwnin only accept page aligned address.
      */
      td->_stackaddr = (void*)((((unsigned long)td->_stackaddr) + getpagesize()-1UL)& ~((unsigned long)getpagesize()-1UL)); 
      td->_stacksize = td->_stacksize & ~((unsigned long)getpagesize()-1UL); 
      err = pthread_attr_setstackaddr( &posix_attr, ((char*)td->_stackaddr)+td->_stacksize );
      if (err !=0) return err;
      err = pthread_attr_setstacksize( &posix_attr, td->_stacksize );
      if (err !=0) return err;
#else      
      err = pthread_attr_setstack( &posix_attr, td->_stackaddr, td->_stacksize );
      if (err !=0) return err;
#endif
    }

    /* detach state */
    if (td->_detachstate !=0) 
      xkaapi_assert( 0 == pthread_attr_setdetachstate(&posix_attr, td->_detachstate) );
      
    /* set cpu set for affinity, only for kernel thread */
#if defined(KAAPI_USE_SCHED_AFFINITY)
    if (attr != &kaapi_default_attr)
    {
      int i;
      CPU_ZERO( &td->_cpuset );
      for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
      {
        if (CPU_ISSET( i, &attr->_cpuset))
          CPU_SET( kaapi_kproc2cpu[i % kaapi_countcpu], &td->_cpuset);
      }
      xkaapi_assert( 0 == pthread_attr_setaffinity_np( &posix_attr, sizeof(cpu_set_t), &td->_cpuset));
    }
#endif    

    /* create the pthread */
    err = pthread_create( &td->_pthid, &posix_attr, &kaapi_start_system_handler, td );
    if (err ==0) *thread = td;
    return err;
  }

  /* case of user level thread  */
#if defined(KAAPI_USE_SCHED_AFFINITY)
  /* process scope: use the cpuset : default is on all processors */
  td->_cpuset = attr->_cpuset;
#endif
  
  /* create a context for a user thread */
#if defined(KAAPI_USE_UCONTEXT)
  err = getcontext(&td->_ctxt);
  xkaapi_assert( err == 0);
  td->_ctxt.uc_link = 0;
  td->_ctxt.uc_stack.ss_sp    = td->_stackaddr;
  td->_ctxt.uc_stack.ss_size  = td->_stacksize;
  td->_ctxt.uc_stack.ss_flags = 0;
  makecontext( &td->_ctxt, (void (*)())&kaapi_start_process_handler, 1, td );

#elif defined(KAAPI_USE_SETJMP)
#  if defined(KAAPI_USE_ARCH_X86)
  _setjmp(td->_ctxt);
  td->_stackaddr=(int *)((unsigned long)td->_stackaddr-((unsigned long)td->_stackaddr % (unsigned long)16)+16);
  td->_stacksize=td->_stacksize-((unsigned long)td->_stacksize % (unsigned long)16);
  xkaapi_update_context(td->_ctxt, &kaapi_start_process_handler, ((char*)td->_stackaddr), td->_stacksize );
#  elif defined(KAAPI_USE_ARCH_ARM)
  _setjmp(_self);
  td->_stackaddr=(int *)((unsigned long)td->_stackaddr-((unsigned long)td->_stackaddr % (unsigned long)16)+16);
  td->_stacksize=td->_stacksize-((unsigned long)td->_stacksize % (unsigned long)16);
  xkaapi_update_context(_self, &kaapi_start_process_handler, ((char*)td->_stackaddr), td->_stacksize );
#  endif
#else 
#  error "not implemented"  
#endif  
  xkaapi_assert(running_proc != 0) 
  /* no lock because, the running proc is doing that operation + no concurrency */
  KAAPI_WORKQUEUE_READY_PUSH(&running_proc->_sc_thread._ready_list, td );

  *thread = td;
  
  return 0;
}


/**
*/
void* kaapi_start_system_handler(void *arg)
{
  kaapi_thread_descr_t* td = (kaapi_thread_descr_t*)arg;
  
  /* set the current running thread as td */
  xkaapi_assert( 0 == pthread_setspecific( kaapi_current_thread_key, td ) );
  
  td->_state = KAAPI_THREAD_RUNNING;

  td->_return_value = (*td->_run_entrypoint)(td->_arg_entrypoint);

  if (td->_detachstate ==0)
    xkaapi_assert ( 0 == pthread_mutex_lock( &td->_mutex_join ) );
  td->_state = KAAPI_THREAD_TERMINATED;
  if (td->_detachstate ==0)
    xkaapi_assert ( 0 == pthread_cond_broadcast( &td->_cond_join ) );
  
  kaapi_dataspecific_destructor (td);
  
  if (td->_detachstate ==0)
    xkaapi_assert ( 0 == pthread_mutex_unlock( &td->_mutex_join ) );

  /* set the current running thread as 0 */
  xkaapi_assert( 0 == pthread_setspecific( kaapi_current_thread_key, 0 ) );
  
  return td->_return_value;
}


/**
*/
static void* kaapi_start_process_handler(void *arg)
{
#if defined(KAAPI_USE_SETJMP)
#if defined(KAAPI_USE_APPLE) && defined(KAAPI_USE_ARCH_PPC)
  /* should be parameter, but ... ? */
  kaapi_thread_descr_t* td = __extension__ ({ register kaapi_thread_descr_t* arg0 __asm("r3"); arg0; });
#elif  defined(KAAPI_USE_APPLE) && defined(KAAPI_USE_ARCH_X86) 
  /* should be parameter, but ... ? */
  kaapi_thread_descr_t* td = __extension__ ({ register kaapi_thread_descr_t* arg0 __asm("eax"); arg0; });
  
  /*some problems with this. But without it doesn't do anything, so it is
   * better*/
#elif defined(KAAPI_USE_IPHONEOS) 

#if defined(KAAPI_USE_ARCH_ARM) 
  kaapi_thread_descr_t* td = __extension__ ({ register kaapi_thread_descr_t* arg0 __asm("r0"); arg0; });
  td = should_th;
#elif defined(KAAPI_USE_ARCH_X86) 
  kaapi_thread_descr_t* td = __extension__ ({ register kaapi_thread_descr_t* arg0 __asm("eax"); arg0; });
#else
#warning "error"
#endif

#endif

#else
  kaapi_thread_descr_t* td = (kaapi_thread_descr_t*)arg;
#endif
  
  xkaapi_assert_debug( td->_proc == (kaapi_processor_t*)pthread_getspecific( kaapi_current_processor_key ) );
  td->_state = KAAPI_THREAD_RUNNING;
  
  /* set the current running thread as the running thread of the processor */
  td->_proc->_sc_thread._active_thread = td;
  /*xkaapi_assert( 0 == pthread_setspecific( current_thread_key, td ) ); */

redo_compute:  
  td->_return_value = (*td->_run_entrypoint)(td->_arg_entrypoint);

  if (td->_detachstate ==0)
    xkaapi_assert( 0 == pthread_mutex_lock( &td->_mutex_join ) );
  td->_state = KAAPI_THREAD_TERMINATED;
  if (td->_detachstate ==0)
    xkaapi_assert( 0 == pthread_cond_broadcast( &td->_cond_join ));
  
  kaapi_dataspecific_destructor (td);
  
  if (td->_detachstate ==0)
    xkaapi_assert( 0 == pthread_mutex_unlock( &td->_mutex_join ) );
  if (kaapi_sched_terminate_or_redo( td->_proc, td )) goto redo_compute;

  xkaapi_assert( 0 );
  return NULL;  
}


/**
*/
static void kaapi_dataspecific_destructor (kaapi_t td)
{
  if (td->_key_table != NULL)
  {
    int i, redo=1, iter=0;
    
    do
    {
      if (redo == 0) break;
      
      redo = 0;
      iter++;
      
      for (i = 0; i < KAAPI_KEYS_MAX; i++)
      {
        if ((kaapi_global_keys[i].next == -1) && (kaapi_global_keys[i].dest != NULL) && (td->_key_table[i] != NULL))
        {
          void (*destor)(void*) = kaapi_global_keys[i].dest;
          void *arg = td->_key_table[i];
          td->_key_table[i] = NULL;
          destor (arg);
          if (td->_key_table[i] != NULL) redo = 1;
        }
      }      
    } while (iter != KAAPI_DESTRUCTOR_ITERATIONS);
  }
}
