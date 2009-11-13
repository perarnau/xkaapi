/*
** kaapi_mt_setconcurrency
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

static void* kaapi_sched_run_processor( void* arg );

/**
*/
static kaapi_atomic_t barrier_init = {0};

/**
*/
static kaapi_atomic_t barrier_init2 = {0};

/** Create and initialize and start concurrency kernel thread to execute user threads
    TODO: faire une implementation dynamique
      - si appel dans un thread kaapi (kaapi_get_current_processor() !=0)
      stopper tous les threads -1
      - sinon stopper tous les threads kaapi.
      - si ajout -> simple
      - si retrait -> a) simple: stopper le thread, lui mettre tout son travail dans le thread main.
                      b) + compliquer: prendre le travail des threads à stopper + signaler (kill kaapi) à se terminer
                      (ou amortir une creation ultérieur des threads en les mettant en attente de se terminer apres un timeout)
      - proteger l'appel d'appel concurrents.
*/
int kaapi_setconcurrency( int concurrency )
{
  static int isinit = 0;
  pthread_t tid;
    
  if (concurrency <1) return EINVAL;
  if (concurrency > default_param.syscpucount) return EINVAL;

  if (isinit) return EINVAL;
  isinit = 1;
  
  /* */
  kaapi_all_kprocessors = calloc( concurrency, sizeof(kaapi_processor_t*) );
  if (kaapi_all_kprocessors ==0) return ENOMEM;

  /* default processor number */
  kaapi_count_kprocessors = concurrency;

  kaapi_barrier_td_init( &barrier_init, 0);
  kaapi_barrier_td_init( &barrier_init2, 1);
      
  /* TODO: allocate each kaapi_processor_t of the selected numa node if it exist */
  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
    if (i>0)
    {
      kaapi_barrier_td_setactive(&barrier_init, 1);
      if (EAGAIN == pthread_create(tid, 0, &kaapi_sched_run_processor, (void*)i))
      {
        kaapi_count_kprocessors = i;
        kaapi_barrier_td_setactive(&barrier_init, 0);
        return EAGAIN;
      }
    }
    else {
      kaapi_all_kprocessors[i] = calloc( 1, sizeof(kaapi_processor_t) );
      kaapi_all_kprocessors[i]->kid = i;
      kaapi_stack_init( &kaapi_all_kprocessors[i]->stack, 0, 0, 0, 0 );

      /* TODO: allocate the hierarchy information -> hrequest data structure */
      kaapi_all_kprocessors[i]->hlevel   = 0;
      kaapi_all_kprocessors[i]->hlkid    = 0;
      kaapi_all_kprocessors[i]->hrequest = 0;
      if (kaapi_all_kprocessors[i] ==0) 
      {
        free(kaapi_all_kprocessors);
        kaapi_all_kprocessors = 0;
        return ENOMEM;
      }

      /* register the processor */
      kaapi_barrier_td_setactive(&kaapi_term_barrier, 1);
    }
  }

  /* wait end of the initialization */
  kaapi_barrier_td_waitterminated( &barrier_init );
  
  /* broadcast to all threads that they have been started */
  kaapi_barrier_td_setactive(&barrier_init2, 0);
  
  kaapi_barrier_td_destroy( &barrier_init );    
  return 0;
}


/**
*/
void* kaapi_sched_run_processor( void* arg )
{
  kaapi_processor_t* kproc =0;
  int i;
  int kid = (int)arg;
  
  /* force reschedule of the posix thread, we that the thread will be mapped on the correct processor ? */
  sched_yield();
  
  kproc = kaapi_all_kprocessors[kid] = calloc( 1, sizeof(kaapi_processor_t) );
  if (kproc ==0) {
    kaapi_barrier_td_setactive(&barrier_init, 0);
    return 0;
  }
  kaapi_assert_debug( 0 == pthread_setspecific( kaapi_current_processor_key, kproc ) );

  kproc->kid = i;
  kaapi_stack_init( &kproc->stack, 0, 0, 0, 0 );

  KAAPI_STACK_CLEAR( &kproc->lsuspend );

  /* kprocessor correctly initialize */
  kaapi_barrier_td_setactive(&kaapi_term_barrier, 1);

  /* quit initialization process */
  kaapi_barrier_td_setactive(&barrier_init, 0);

  /* wait end of the first steal initialization */
  kaapi_barrier_td_waitterminated( &barrier_init2 );

  nkproc = KAAPI_ATOMIC_READ( &kaapi_term_barrier );
  
  /* TODO: allocate the hierarchy information and initialize the hrequest data structure */
  kproc->hlevel   = 1;
  kproc->hlkid    = calloc( kproc->hlevel, sizeof(kaapi_uint16_t) );
  kproc->hrequest = calloc( kproc->hlevel, sizeof(kaapi_list_request) );
  for (i=0; i<kproc->hlevel; ++i)
  {
    kproc->hlkid[i] = kproc->kid; /* only one level !!!! */
    kaapi_listrequest_init( &kproc->hrequest[i] );
  }  
      
  /* may wait startup here ? */
  
  kaapi_sched_idle( kproc );
  return 0;
}