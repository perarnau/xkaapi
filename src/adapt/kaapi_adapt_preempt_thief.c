/*
 ** kaapi_task_preempt_next.c
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

#if 0
/*
 */
int kaapi_remove_finishedthief( 
                               kaapi_stealcontext_t*        sc,
                               kaapi_taskadaptive_result_t* ktr
                               )
{
  kaapi_task_lock_adaptive_steal(sc);
  
  if (ktr->rhead != 0)
  {
    /* rtail non null too */
    ktr->rhead->prev = ktr->prev;
    ktr->rtail->next = ktr->next;
    
    if (ktr->prev != 0)
      ktr->prev->next = ktr->rhead;
    else
      sc->thieves.list.head = ktr->rhead;
    
    if (ktr->next != 0)
      ktr->next->prev = ktr->rtail;
    else
      sc->thieves.list.tail = ktr->rtail;
  }
  else /* no thieves, unlink */
  {
    if (ktr->prev != 0)
      ktr->prev->next = ktr->next;
    else
      sc->thieves.list.head = ktr->next;
    
    if (ktr->next != 0)
      ktr->next->prev = ktr->prev;
    else
      sc->thieves.list.tail = ktr->prev;
  }
  
  /* mostly for debug: */
  ktr->next = 0;
  ktr->prev = 0;
  
  kaapi_task_unlock_adaptive_steal(sc);
  
  return 0;
}




/*
 */
int kaapi_preempt_thief_helper
( 
 kaapi_stealcontext_t*        sc,
 kaapi_taskadaptive_result_t* ktr, 
 void*                        arg_to_thief
 )
{
#if defined(KAAPI_USE_PERFCOUNTER)
  uint64_t t0;
  uint64_t t1;
#endif
  
  kaapi_task_body_t body;
  int retval;
  
  kaapi_assert_debug(ktr != 0);
  
#if defined(KAAPI_USE_PERFCOUNTER)
  t0 = kaapi_get_elapsedns();
#endif
  
  /* pass arg to the thief */
  ktr->arg_from_victim = arg_to_thief;  
  
  /* next write should ne be reorder with previous */
  kaapi_writemem_barrier();
  
#warning TODO HERE
#if 0
  /* preempt the task if not already terminated */
  body = kaapi_task_getbody(&ktr->state);
  if (body != kaapi_term_body)
  {
    retval = kaapi_task_casbody(&ktr->state, body, kaapi_preempt_body);
    if (!retval) 
    {
      kaapi_writemem_barrier();
      body = kaapi_task_getbody(&ktr->state);
      kaapi_assert( body == kaapi_term_body );
    }
    else {
      *ktr->preempt = 1;
      kaapi_mem_barrier();
      
      /* wait until task is terminated */
      while (1)
      {
        if (kaapi_task_getbody(&ktr->state) == kaapi_term_body)
          break;
        kaapi_slowdown_cpu();
      }
    }
  }
#endif

  /* remove thief and replace the thieves of ktr into the list */
  kaapi_remove_finishedthief(sc, ktr);
  
#if defined(KAAPI_USE_PERFCOUNTER)
  t1 = kaapi_get_elapsedns();
#if 0 /* TODO */
  sc->ctxtthread->proc->t_preempt += (double)(t1-t0)*1e-9;
#endif
#endif
  
  return 1;
}



/*
 */
int kaapi_preemptasync_thief
( 
 kaapi_stealcontext_t*               sc __attribute__((unused)), 
 struct kaapi_taskadaptive_result_t* ktr, 
 void*                               arg_to_thief 
 )
{
  kaapi_task_body_t body;
  int retval;

  if (ktr ==0) return 0;
  
  /* pass arg to the thief */
  ktr->arg_from_victim = arg_to_thief;  
  kaapi_writemem_barrier();

#warning TODO HERE
#if 0
  /* preempt the task if not already terminated */
  body = kaapi_task_getbody(&ktr->state);
  if (body == kaapi_term_body) return 0;
  retval = kaapi_task_casbody(&ktr->state, body, kaapi_preempt_body);
#endif
  if (retval)
  {
    *ktr->preempt = 1;
    return 0;
  }
  return ECHILD;
}


/*
 */
int kaapi_preemptasync_waitthief
( 
 kaapi_stealcontext_t*               sc,
 struct kaapi_taskadaptive_result_t* ktr 
 )
{
  if (ktr ==0) return 0;

#warning TODO HERE
  
  /* wait until task is terminated */
  while (kaapi_task_getstate(&ktr->state) != KAAPI_TASK_STATE_TERM)
  {
    kaapi_slowdown_cpu();
  }
  
  /* remove thief and replace the thieves of ktr into the list */
  kaapi_remove_finishedthief(sc, ktr);
  
  return 0;
}

#endif // #if 0
