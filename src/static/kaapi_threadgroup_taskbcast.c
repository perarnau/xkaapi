/*
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
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
#include "kaapi_staticsched.h"


/*
*/
void kaapi_taskbcast_body( void* sp, kaapi_thread_t* thread )
{
  /* thread->pc is the executing task */
  kaapi_task_t*          self    = thread->pc;
#if 0
  kaapi_taskbcast_arg_t* wc_list = (kaapi_taskbcast_arg_t*)self->pad;
  kaapi_task_t* task;
  short i;
#endif

  if (sp != self->pad)
  { /* encapsulation of the taskbcast on top of an existing task */
    (*self->ebody)(sp,thread);
  }
  
  /* write memory barrier to ensure that other threads will view the data produced */
  kaapi_mem_barrier();
#if 0

  /* signal all readers */  
  while(wc_list != 0) 
  {
    for (i=0; i<wc_list->size; ++i)
    {
      counter = wc_list->entry[i].waiting_counter;
      task = wc_list->entry[i].waiting_task;
      
      if (KAAPI_ATOMIC_DECR(counter) ==0)
        kaapi_task_setbody(task, task->ebody);
    }
    wc_list = wc_list->next;
  }
#endif
}
