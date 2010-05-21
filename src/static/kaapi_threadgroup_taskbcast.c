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


/*
*/
void kaapi_taskbcast_body( void* sp, kaapi_thread_t* thread )
{
  /* thread[-1]->pc is the executing task (pc of the upper frame) */
  /* kaapi_task_t*          self = thread[-1].pc;*/
  kaapi_taskbcast_arg_t* arg  = sp;
  kaapi_com_t* comlist;
  kaapi_processor_t* kproc = 0;
  int i;

  printf("In TaskBcast body\n");
  if (arg->common.original_body != 0)
  { /* encapsulation of the taskbcast on top of an existing task */
    (*arg->common.original_body)(arg->common.original_sp,thread);
  }
  
  /* write memory barrier to ensure that other threads will view the data produced */
  kaapi_mem_barrier();

  /* signal all readers */
  kproc = kaapi_get_current_processor();
  comlist = &arg->head;
  while(comlist != 0) 
  {
    for (i=0; i<comlist->size; ++i)
    {
      kaapi_task_t* task = comlist->entry[i].task;
      kaapi_assert( (task->ebody == kaapi_taskrecv_body) || (task->ebody == kaapi_taskbcast_body) );
      kaapi_taskrecv_arg_t* argrecv = (kaapi_taskrecv_arg_t*)task->sp;
      
      void* newsp;
      kaapi_task_body_t newbody;
      if (task->ebody == kaapi_taskrecv_body)
      {
        newbody = argrecv->original_body;
        newsp   = argrecv->original_sp;
      }
      else 
      {
        newbody = task->ebody;
        newsp   = task->sp;
      }
      
      if (kaapi_threadgroup_decrcounter(argrecv) ==0)
      {
        /* task becomes ready */        
        task->sp = newsp;
        kaapi_task_setextrabody(task, newbody);

        /* see code in kaapi_taskwrite_body */
        if (task->pad != 0) {
          kaapi_wc_structure_t* wcs = (kaapi_wc_structure_t*)task->pad;
          kaapi_stack_t* stack = kaapi_threadcontext2stack(wcs->wccell->thread);
          if (!stack->sticky) 
          {
            /* remove it from suspended queue */
            kaapi_thread_context_t* kthread = kaapi_wsqueuectxt_steal_cell( wcs->wclist, wcs->wccell );
            if (kthread !=0)
            {
              kaapi_sched_lock( kproc );
              kaapi_task_setbody(task, newbody );
              kaapi_sched_pushready( kproc, kthread );
              /* bcast will activate a suspended thread */
              printf("Bcast wakeup non stick stack @:%p, can be moved...\n", (void*)stack);
              fflush(stdout );
              kaapi_sched_unlock( kproc );
            }
            else 
              kaapi_task_setbody(task, newbody);
          }
          else {
            /* may activate the task */
            kaapi_task_setbody(task, newbody);
            /* else stack may not be .. */
            printf("Bcast wakeup stick stack @:%p, cannot be moved...\n", (void*)stack);
          }
        }
        else {
          /* thread is not suspended... */        
          /* may activate the task */
          kaapi_task_setbody(task, newbody);
        }
      }
    }
    comlist = comlist->next;
  }

  printf("Out TaskBcast body\n");
}
