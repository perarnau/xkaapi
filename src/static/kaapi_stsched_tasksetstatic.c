/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
void kaapi_staticschedtask_body( void* sp, kaapi_thread_t* uthread )
{
  int save_state;
  kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)sp;
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  kaapi_assert( thread->sfp == (kaapi_frame_t*)uthread );
  
  /* unset steal capability and wait no more thief 
     lock the kproc: it ensure that no more thief has reference on it 
  */
  kaapi_sched_lock(&thread->proc->lock);
  save_state = thread->unstealable;
  thread->unstealable = 1;
  kaapi_sched_unlock(&thread->proc->lock);

  /* the embedded task cannot be steal because it was not visible to thieves */
  arg->sub_body( arg->sub_sp, uthread );
  
  /* currently: that all, do not compute other things */
  double t0 = kaapi_get_elapsedtime();
  kaapi_sched_computereadylist();
  double t1 = kaapi_get_elapsedtime();
  
  printf("Scheduling in %e (s)\n",t1-t0);
  thread->unstealable = save_state;

//  kaapi_thread_print( stdout, thread ); 
  FILE* filedot = fopen("/tmp/graph.dot", "w");
  kaapi_frame_print_dot( filedot, thread->sfp );
  fclose(filedot);
  
  /* exec the spawned subtasks */
  kaapi_thread_execframe_readylist( thread );
}


