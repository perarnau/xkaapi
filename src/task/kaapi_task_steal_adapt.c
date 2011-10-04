/*
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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

void kaapi_task_steal_adapt
(
  kaapi_thread_context_t*       thread, 
  kaapi_task_t*                 task,
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange,
  void                        (*callback_empty)(kaapi_task_t*)
)
{
  kaapi_task_splitter_t splitter; 
  void*                 argsplitter;
  kaapi_task_body_t     task_body = kaapi_task_getbody(task);
#warning TODO HERE
  /* its an adaptive task !!! */
  if (task_body == kaapi_adapt_body || task_body == kaapi_hws_adapt_body)
  {
    /* only steal into an correctly initialized steal context */
    kaapi_stealcontext_t* const sc = kaapi_task_getargst(task, kaapi_stealcontext_t);
    if (sc->header.flag & KAAPI_SC_INIT) 
    {
      splitter = sc->splitter;
      argsplitter = sc->argsplitter;
      
      if (splitter !=0)
      {
        uintptr_t orig_state = kaapi_task_getstate(task);
        /* do not steal if terminated */
        if (    (orig_state == KAAPI_TASK_STATE_INIT) && (orig_state == KAAPI_TASK_STATE_EXEC)
             && likely(kaapi_task_casstate(task, orig_state, KAAPI_TASK_STATE_STEAL))
           )
        {
          splitter = sc->splitter;
          if (splitter !=0)
          {
            /* steal may sucess: possible race, reread the splitter */
            argsplitter = sc->argsplitter;
            const kaapi_ws_error_t err = kaapi_task_splitter_adapt(
                  thread, 
                  task, 
                  splitter, 
                  argsplitter, 
                  lrequests, 
                  lrrange 
            );
            if ((err == KAAPI_WS_ERROR_EMPTY) && (callback_empty !=0))
              callback_empty(task);
          }
          
          /* reset initial state of the task */
          kaapi_task_setstate(task, orig_state);
        }
      }
    } /* end if init */
  }
}
