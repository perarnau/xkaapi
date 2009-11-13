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


/*
*/
static void* kaapi_start_system_handler(void *arg);

/*
*/
static void* kaapi_start_processor_handler(void *arg);

/*
*/
static void kaapi_body_exec_processthread( kaapi_task_t* task, kaapi_stack_t* stack );

/*
*/
static void kaapi_dataspecific_destructor (kaapi_key_table_t dataspecific);


/*
*/
int kaapi_create_process(
    kaapi_t *__restrict thread, const kaapi_attr_t *__restrict attr, void*(*start_routine)(void *), void *__restrict arg
)
{
  kaapi_task_t* task_run;
  kaapi_thread_descr_t* kproc = kaapi_sched_get_processor();
  kaapi_assert_debug( kproc !=0 );
  
  /* only detach attribut for user thread (process scope) are used
     The others are ignored
  */
  task_run           = kaapi_stack_top( &kproc->th.k.ctxt.kstack);
  task_run->sf.state = KAAPI_TASK_INIT;
  task_run->body     = &kaapi_body_exec_processthread;
  task_run->param.rdata[0] = (kaapi_func_t)start_routine;
  task_run->param.pdata[1] = arg;
  kaapi_stack_push( &kproc->th.k.ctxt.kstack);
  return 0;
}



/*
*/
int kaapi_create_system_or_processor(
  kaapi_t *__restrict thread, const kaapi_attr_t *__restrict attr, void*(*start_routine)(void *), void *__restrict arg
)
{
  int err;
  kaapi_thread_descr_t* td;
  pthread_attr_t posix_attr;
  
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug( start_routine !=0 );
  kaapi_assert_debug( attr !=0 );
  kaapi_assert_debug((attr->_scope == KAAPI_SYSTEM_SCOPE)|| (attr->_scope == KAAPI_PROCESSOR_SCOPE));


  if ((attr->_stackaddr != 0) && (attr->_stacksize ==0)) return EINVAL;
  
  /* should also initialize td using thread attribut */
  td = kaapi_allocate_thread_descriptor( attr->_scope, 
                                         attr->_detachstate, 
                                         (attr->_stackaddr != 0 ? 0 : attr->_stacksize), 
                                         attr->_stacksize
                                        );

  /* initialize the fields */
  td->state          = KAAPI_THREAD_S_CREATED;

  if (td->scope == KAAPI_SYSTEM_SCOPE)
  {
    td->th.s.entrypoint     = start_routine;
    td->th.s.arg_entrypoint = arg;
  }

  /* set attr to the posix thread */
  kaapi_assert ( 0 == pthread_attr_init( &posix_attr ));

  if (attr->_stackaddr !=0) 
  {
#if defined(KAAPI_USE_APPLE)
    /* stack grows down, so offset the base pointer to the last position.
       It is a bogus implementation on darwin ?
       It seems that darwnin only accept page aligned address.
    */
    void* stackaddr = (void*)((((unsigned long)attr->_stackaddr) + getpagesize()-1UL)& ~((unsigned long)getpagesize()-1UL)); 
    size_t stacksize = attr->_stacksize & ~((unsigned long)getpagesize()-1UL); 
    err = pthread_attr_setstackaddr( &posix_attr, ((char*)stackaddr)+stacksize );
    if (err !=0) return err;
    err = pthread_attr_setstacksize( &posix_attr, stacksize );
    if (err !=0) return err;
#else      
    err = pthread_attr_setstack( &posix_attr, attr->_stackaddr, attr->_stacksize );
    if (err !=0) return err;
#endif
  }

  /* detach state always set for processor */
  kaapi_assert_debug( (td->scope != KAAPI_PROCESSOR_SCOPE) || (attr->_detachstate !=0));
  
  if (attr->_detachstate !=0) 
    kaapi_assert( 0 == pthread_attr_setdetachstate(&posix_attr, PTHREAD_CREATE_DETACHED) );
    
  /* create the pthread */
  if (td->scope == KAAPI_SYSTEM_SCOPE)
    err = pthread_create( &td->th.s.pthid, &posix_attr, &kaapi_start_system_handler, td );
  else {
    pthread_t pthid;
    err = pthread_create( &pthid, &posix_attr, &kaapi_start_processor_handler, td );
  }

  if ((err ==0) && (thread !=0))
  {
    thread->tid   = (td->scope == KAAPI_SYSTEM_SCOPE ? td->th.s.tid : td->th.k.ctxt.tid);
    thread->futur = td->futur; 
  }
  return err;
}


/**
*/
int kaapi_create(
    kaapi_t *__restrict thread, const kaapi_attr_t *__restrict attr, void*(*start_routine)(void *), void *__restrict arg
)
{
  if (thread ==0) return EINVAL;
  if (start_routine ==0) return EINVAL;
  if (attr ==0) attr = &kaapi_default_attr;

  if ((attr->_scope == KAAPI_SYSTEM_SCOPE)|| (attr->_scope == KAAPI_PROCESSOR_SCOPE))
    return kaapi_create_system_or_processor(thread, attr, start_routine, arg);

  else if (attr->_scope == KAAPI_PROCESS_SCOPE)
    return kaapi_create_process(thread, attr, start_routine, arg);

  return EINVAL; /* bad scope !!! */
}



/**
*/
void* kaapi_start_system_handler(void *arg)
{
  void* retval;
  kaapi_thread_futur_t* futur;
  kaapi_thread_descr_t* td = (kaapi_thread_descr_t*)arg;
  
  /* set the current running thread as td */
  kaapi_assert( 0 == pthread_setspecific( kaapi_current_thread_key, td ) );
  
  td->state = KAAPI_THREAD_S_RUNNING;

  retval = (*td->th.s.entrypoint)(td->th.s.arg_entrypoint);

  td->state = KAAPI_THREAD_S_TERMINATED;
  
  kaapi_dataspecific_destructor (td->th.s.dataspecific);
  
  /* */
  futur = td->futur;
  if (futur !=0) 
  {
    futur->result = retval;
    td->futur = 0;
    kaapi_writemem_barrier();
    futur->state |= KAAPI_FUTUR_S_TERMINATED;
    kaapi_cond_broadcast( &futur->condition );
  }
  
  /* set the current running thread as 0 */
  kaapi_assert( 0 == pthread_setspecific( kaapi_current_thread_key, 0 ) );
  
  return 0;
}



/**
*/
void* kaapi_start_processor_handler(void *arg)
{
  void* retval;
  kaapi_thread_descr_t* td = (kaapi_thread_descr_t*)arg;
  
  /* set the current running thread as td */
  kaapi_assert( 0 == pthread_setspecific( kaapi_current_thread_key, td ) );
  
  td->state = KAAPI_THREAD_S_RUNNING;

  kaapi_sched_idle();

  td->state = KAAPI_THREAD_S_TERMINATED;
  
  kaapi_dataspecific_destructor (td->th.k.ctxt.dataspecific);
  
  /* set the current running thread as 0 */
  kaapi_assert( 0 == pthread_setspecific( kaapi_current_thread_key, 0 ) );
  
  return 0;
}


/*
*/
void kaapi_body_exec_processthread( kaapi_task_t* task, kaapi_stack_t* stack )
{
  void* (*startup_rountine)(void*) = (void* (*)(void*))task->param.rdata[0];
  (*startup_rountine)(task->param.pdata[1]);
}


/**
*/
static void kaapi_dataspecific_destructor (kaapi_key_table_t dataspecific)
{
  int i, redo=1, iter=0;
  if (dataspecific ==0) return;

  do
  {
    if (redo == 0) break;
    
    redo = 0;
    iter++;
    
    for (i = 0; i < KAAPI_KEYS_MAX; i++)
    {
      if ((kaapi_global_keys[i].next == -1) && (kaapi_global_keys[i].dest != NULL) && (dataspecific[i] != NULL))
      {
        void (*destor)(void*) = kaapi_global_keys[i].dest;
        void *arg = dataspecific[i];
        dataspecific[i] = NULL;
        destor (arg);
        if (dataspecific[i] != NULL) redo = 1;
      }
    }      
  } while (iter != KAAPI_DESTRUCTOR_ITERATIONS);

}
