/*
** xkaapi
** 
**
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

/** Activate and push ready tasks of an activation link.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
int kaapi_tasklist_pushready_td( 
    kaapi_tasklist_t*       tasklist, 
    kaapi_taskdescr_t*      td,
#if !defined(TASKLIST_ONEGLOBAL_MASTER)  
    kaapi_taskdescr_t**     tdref,
#endif
    int                     priority
)
{
#if 0 /* desactivate this portion of code if you do not want push on remote queue */
  kaapi_ws_queue_t* queue = 0;
  int nodeid;
  kaapi_bitmap_value32_t ocr = td->ocr;
  uintptr_t addr = 0;
  kaapi_task_t* task;
#if defined(KAAPI_TASKLIST_POINTER_TASK)
  task = td->task;
#else
  task = &td->task;
#endif
#endif


#if 0 /* desactivate this portion of code if you do not want push on remote queue */
  int nodeid;

  /* if non local -> push on remote queue ? 
     - first find a memory address where to get numa information
     - then if addr !=0, get the associated queue
  */
  if (kaapi_default_param.emitsteal == kaapi_sched_flat_emitsteal) 
  {
    /* here to not push into remote queue because flat emitsteal is inable to steal inside them */
  }
  else if (kaapi_bitmap_value_empty_32(&ocr) || (td->fmt ==0))
  {
    if ((task->body == kaapi_taskmove_body) || (task->body == kaapi_taskalloc_body))
    {
      kaapi_move_arg_t* arg = (kaapi_move_arg_t*)task->sp;
      addr = arg->src_data.ptr.ptr;
    }
  }
  else 
  {
    int ith = kaapi_bitmap_value_first1_and_zero_32( &ocr )-1;
  /* get binding. TEST: only logical binding */
    kaapi_access_t a __attribute__((unused))= kaapi_format_get_access_param(td->fmt, ith, task->sp);
    kaapi_data_t* gd = (kaapi_data_t*)a.data;
    addr = gd->ptr.ptr;
  }

  if (addr !=0)
  {
    nodeid = kaapi_numa_get_page_node( (void*)addr );
    if (nodeid != -1)
      queue = hws_levels[KAAPI_HWS_LEVELID_NUMA].blocks[ nodeid ].queue;
  }
#endif
  
  if (1) //(queue == 0)
  {
    /* I was not able to identify a queue for the task: push locally */
    kaapi_readylist_pushone_td( 
        &tasklist->rtl, 
        td, 
        (tasklist->t_infinity !=0) ?
          (tasklist->t_infinity - td->u.acl.date) * (KAAPI_TASKLIST_NUM_PRIORITY-1) / tasklist->t_infinity
        :  td->priority 
    );
    return 0;
  }

#if 0
  /* push remote queue:
     - same code as kaapi_task_splitter_readylist except state ALLOCATED */
#if defined(TASKLIST_ONEGLOBAL_MASTER)  
  if (tasklist->master == 0)
    td->tasksteal_arg.master_tasklist     = tasklist;
  else
    td->tasksteal_arg.master_tasklist     = tasklist->master;
  td->tasksteal_arg.td                    = td;
#else
  td->tasksteal_arg.master_tasklist       = tasklist;
  td->tasksteal_arg.td_beg                = &td;
  td->tasksteal_arg.td_beg                = tdref;
  td->tasksteal_arg.td_end                = tdref+1;
#endif

  kaapi_task_init(
      &td->tasksteal, 
      kaapi_taskstealready_body, 
      &td->tasksteal_arg
  );

#if defined(TASKLIST_ONEGLOBAL_MASTER)  
  if (tasklist->master == 0)
    KAAPI_ATOMIC_INCR(&tasklist->count_thief);
  else
    KAAPI_ATOMIC_INCR(&tasklist->master->count_thief);
#else
  KAAPI_ATOMIC_INCR(&tasklist->count_thief);
#endif

  /* push the task in the remote queue */
  //kaapi_thread_pushtask_atlevel(task, KAAPI_HWS_LEVELID_NUMA);
  kaapi_thread_pushtask_atlevel_with_nodeid( &td->tasksteal, KAAPI_HWS_LEVELID_NUMA, nodeid );

#endif

  return 1;
}
