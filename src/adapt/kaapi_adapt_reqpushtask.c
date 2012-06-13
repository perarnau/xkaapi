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


int kaapi_request_pushtask(kaapi_request_t* request, kaapi_task_t* victim_task )
{
  kaapi_task_t* toptask;
  kaapi_task_t* signaltask;
  kaapi_stealcontext_t* victim_sc;
  
  kaapi_assert_debug( request !=0 );

  /* this a reply to a steal request: do not allow to steal first again the task */
  kaapi_task_set_unstealable(request->frame.sp);
  
  if (victim_task !=0)
  {
    if (kaapi_task_is_withpreemption(victim_task))
    {
      /* push a non adaptive task, but with preemption = used general pushtask_adaptive */
      return kaapi_request_pushtask_adaptive(
                  request, 
                  victim_task, 
                  0, 
                  KAAPI_REQUEST_REPLY_TAIL);
    }

    toptask = kaapi_request_toptask(request);
    signaltask = kaapi_thread_nexttask(&request->frame, toptask);
    victim_sc = (kaapi_stealcontext_t*)kaapi_task_getargst(
          victim_task, kaapi_taskadaptive_arg_t)->shared_sc.data;

    kaapi_task_init_with_flag(
        signaltask, 
        kaapi_tasksignaladapt_body, 
        victim_sc->msc,
        KAAPI_TASK_UNSTEALABLE /* default is also un-splittable */
    );
    KAAPI_ATOMIC_INCR(&victim_sc->msc->thieves.count);

    /* push two tasks at a time */
    request->frame.sp -= 2;
  }
  else 
    /* else push only the top task */
    --request->frame.sp;
  
  /* do not use a barrier here: it exists on reply 
     to the request, see kaapi_request_committask */
  return 0;
}
