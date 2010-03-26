/*
** kaapi_task_preemptpoint.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
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

/**
*/
int kaapi_preemptpoint_before_reducer_call( 
    struct kaapi_taskadaptive_result_t* ktr, 
    kaapi_stealcontext_t* stc,
    void* arg_for_victim, int size 
)
{
  /* push data to the victim and list of thief */
  if (arg_for_victim !=0)
  {
    if (size > ktr->size_data) size = ktr->size_data;
    memcpy(ktr->data, arg_for_victim, size );
  }

  if (stc !=0)
  {
    kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)stc;

    kaapi_steal_begincritical( &ta->sc );
    /* avoid to steal to this context */
    ta->sc.splitter = 0;
    ta->sc.argsplitter = 0;    
    kaapi_steal_endcritical( &ta->sc );
    
    /* here no more steal thief can call the splitter on stc */
    ktr->rhead = ta->head; ta->head = 0;
    ktr->rtail = ta->tail; ta->tail = 0;
    
  }
  
  /* delete the preemption flag */
  ktr->req_preempt = 0;

  return 0;
}

/**
*/
int kaapi_preemptpoint_after_reducer_call( 
    kaapi_taskadaptive_result_t* ktr, 
    kaapi_stealcontext_t* stc,
    int reducer_retval 
)
{

  kaapi_writemem_barrier();   /* serialize previous line with next line */
  ktr->thief_term = 1;

  return 1;
}
