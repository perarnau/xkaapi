/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
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

/** 
*/
int kaapi_thread_computeready_access( 
    kaapi_tasklist_t*   tl, 
    kaapi_version_t*    version, 
    kaapi_taskdescr_t*  task,
    kaapi_access_mode_t m 
)
{
  if (version->last_mode == KAAPI_ACCESS_MODE_VOID)
  {
    if (KAAPI_ACCESS_IS_READ(m))
    {
      /* push a bcast task into list of bcast of the task 'task' */
      kaapi_taskdescr_t* td_move;
      kaapi_task_t* task_move;
      kaapi_move_arg_t* argmove 
          = (kaapi_move_arg_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_move_arg_t) );
      argmove->src_data  = version->orig.addr;
      argmove->src_view  = version->orig.view;
      argmove->dest      = version->handle;
      task_move = kaapi_tasklist_push_task( tl, kaapi_taskmove_body, argmove);
      td_move =  kaapi_tasklist_allocate_td( tl, task_move );
      version->writer_task= td_move;
      version->is_ready = 0;
      kaapi_tasklist_push_successor( tl, td_move, task );
      kaapi_tasklist_pushback_ready( tl, td_move);
    }
    else if (KAAPI_ACCESS_IS_WRITE(m))
      version->writer_task = task;
  }
  else if (KAAPI_ACCESS_IS_CONCURRENT(m, version->last_mode))
  {
    if (KAAPI_ACCESS_IS_CUMULWRITE(m))
    { /* concurrent => last_mode is also a CW, writer is the futur writer */
      /* cw access are link together */
      if (version->last_task != 0) 
      {
        kaapi_tasklist_push_successor( tl, version->last_task, task );
      }
#warning "TODO: indep CW, mais ordre à respecter"
      version->writer_task = task;
    }
    else 
    { /* its a r */
      if (version->writer_task !=0)
      {
        kaapi_assert_debug(!version->is_ready);
        kaapi_tasklist_push_successor( tl, version->writer_task, task );
      }
    }

  }
  else if (KAAPI_ACCESS_IS_READWRITE(m)) /* rw */
  {
    if (version->last_task !=0) /* whatever is the previous task, do link */
    {
      kaapi_tasklist_push_successor( tl, version->last_task, task );
      version->is_ready    = 0;
    }
    else 
    {
      kaapi_assert_debug( version->last_mode == KAAPI_ACCESS_MODE_VOID);
    }
    version->writer_task = task;
  }
  else if (KAAPI_ACCESS_IS_WRITE(m)) /* cw or w */
  {
    /* WAR or WAW */
    kaapi_assert_debug( version->last_task !=0 ); //* else mode == VOID !!! */
    if (version->last_task->date < task->date)
    { /* not a true WAR or WAW */
      printf("Task: td:%p -> task:%p has a false WA{RW} dependency with td:%p -> task:%p\n",
          (void*)task, (void*)task->task,
          (void*)version->last_task, (void*)version->last_task->task
      );
    }
    else 
    { /* potential WAR or WAW : allocate a new data and insert it into hash map */
      printf("Task: td:%p -> task:%p has WA{RW} dependency with td:%p -> task:%p\n",
          (void*)task, (void*)task->task,
          (void*)version->last_task, (void*)version->last_task->task
      );
      kaapi_assert(0);
    }
    version->writer_task = task;
    version->is_ready = 0;
  }
  else if (KAAPI_ACCESS_IS_READ(m)) /* r (rw already test) */
  { /* means previous is a w or rw or cw */
    kaapi_tasklist_push_successor( tl, version->last_task, task );
    version->is_ready = 0;
  }
  
  version->last_mode = m;
  version->last_task = task;
  
  return 0;
}
