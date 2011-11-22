/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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


int kaapi_request_pushtask(kaapi_request_t* request)
{
  kaapi_assert_debug( request !=0 );
  /* this a reply to a steal request: do not allow to steal first again the task */
  kaapi_task_set_unstealable(request->frame.sp);
  --request->frame.sp;
  /* do not use a barrier here: it exists on reply 
     to the request, see kaapi_request_committask */
  return 0;
}


/* Here:
    - the user task is already pushed its task into the frame pointed by the request
    - acts as kaapi_thread_pushtask_adaptive in order to envelop the (adaptative) top task
    in a kaapi_taskstealadapt_body body.
    - the kaapi_taskstealadapt_body has the following properties:
      - it has the same format as the embedded task with an extra format parameter
      - on remote side, the body executes the task (as kaapi_taskadapt_body)
      - it updates (using XX_body) the thief result with the list of thieves
    - link victim steal context and thief steal context using the direction encoded in headtail_flag
 */
int kaapi_request_pushtask_adaptive(
  kaapi_request_t*              request, 
  kaapi_task_t*                 victim_task, 
  kaapi_adaptivetask_splitter_t user_splitter,
  int                           headtail_flag
)
{
  kaapi_taskadaptive_arg_t* victim_adapt_arg;
  kaapi_taskadaptive_arg_t* adapt_arg;
  kaapi_stealcontext_t* sc;
  kaapi_task_t* toptask;
  
  kaapi_assert_debug( kaapi_task_is_splittable(victim_task) );
  kaapi_assert_debug( (victim_task->body == (kaapi_task_body_t)kaapi_taskadapt_body )
                  ||  (victim_task->body == (kaapi_task_body_t)kaapi_taskbegendadapt_body) );
  
  toptask = kaapi_thread_toptask(&request->frame);
  /* this a reply to an adaptive task: set the task as unstealable */
  kaapi_task_set_unstealable(toptask);
  kaapi_task_set_splittable(toptask);
  kaapi_thread_pushtask_adaptive(&request->frame, user_splitter);

  /* here toptask was replaced in pushtask_adaptive by a kaapi_taskadapt_body, 
     but flags remain equals.
     Link together the adaptive victim_task's steal context to the newly created task stealcontext
  */
  victim_adapt_arg = kaapi_task_getargst(victim_task, kaapi_taskadaptive_arg_t);
  adapt_arg = kaapi_task_getargst(toptask, kaapi_taskadaptive_arg_t);
  sc = (kaapi_stealcontext_t*)adapt_arg->shared_sc.data;
  sc->msc  = ((kaapi_stealcontext_t*)victim_adapt_arg->shared_sc.data)->msc;
  KAAPI_DEBUG_INST( adapt_arg->version = victim_adapt_arg->version; )

  if (kaapi_task_is_withpreemption(victim_task) && !kaapi_task_is_withpreemption(toptask))
  {
    toptask->u.s.flag = KAAPI_TASK_S_PREEMPTION;
    sc->flag = KAAPI_SC_PREEMPTION;
  }

  /* list insert is atomic with respect to the lock on the adaptive task */
  if (kaapi_task_is_withpreemption(toptask))
  {
    /* this allocation may be amortized if we reserve some memory in the steal context
       for thief + bloc allocation
    */
    kaapi_thiefadaptcontext_t* ktr 
      = (kaapi_thiefadaptcontext_t*)malloc(sizeof(kaapi_thiefadaptcontext_t));
    if (ktr ==0)
      return ENOMEM;
    sc->ktr = ktr;
    kaapi_atomic_initlock(&ktr->lock);
    ktr->thief_task = toptask; /* task to signal */
#if defined(KAAPI_DEBUG) /* will be set by the thief on terminaison / preemption */
    ktr->thief_of_the_thief_tail = 0;
    ktr->thief_of_the_thief_head = 0;
#endif
  
    /* insert in head or tail: one thief do allocation at any time, due 
       to kproc lock owner ship
    */
    if (sc->thieves.list.head == 0)
    {
      sc->thieves.list.tail = ktr;
      sc->thieves.list.head = ktr;
    }
    else if (headtail_flag == KAAPI_REQUEST_REPLY_HEAD)
    { 
      ktr->next = sc->thieves.list.head;
      sc->thieves.list.head->prev = ktr;
      sc->thieves.list.head = ktr;
    } 
    else 
    {
      ktr->prev = sc->thieves.list.tail;
      sc->thieves.list.tail->next = ktr;
      sc->thieves.list.tail = ktr;
    }
  }
  else
  {
    sc->ktr = 0;
    KAAPI_ATOMIC_INCR(&sc->msc->thieves.count);
  }
  
  return 0;
}
