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
#include "kaapi_staticsched.h"


/*
*/
int kaapi_task_bindfifo(kaapi_task_t* task, int ith, kaapi_access_mode_t m, kaapi_fifo_t* fifo_param)
{
  if ( !KAAPI_ACCESS_IS_FIFO(m)) return 0;

  kaapi_assert_debug( KAAPI_ACCESS_IS_ONLYWRITE(KAAPI_ACCESS_GET_MODE(m)) || KAAPI_ACCESS_IS_READ(KAAPI_ACCESS_GET_MODE(m)));
  if (KAAPI_ACCESS_IS_READ(KAAPI_ACCESS_GET_MODE(m)))
  {
    kaapi_assert( fifo_param->task_reader ==0);
    fifo_param->task_reader  = task;
    fifo_param->param_reader = ith;
  }
  else if (KAAPI_ACCESS_IS_ONLYWRITE(KAAPI_ACCESS_GET_MODE(m)))
  {
    kaapi_assert( fifo_param->task_writer ==0);
    fifo_param->task_writer  = task;
    fifo_param->param_writer = ith;
  }
  return 0;
}


/*
*/
int kaapi_thread_pushfifotask(kaapi_thread_t* thread)
{
  int i, countparam;
  const kaapi_format_t* task_fmt;
  kaapi_task_t* task = kaapi_thread_toptask(thread);
  task_fmt = kaapi_format_resolvebybody( task->body );

  if (task_fmt ==0) return 0;

  countparam = task_fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = task_fmt->mode_params[i];
    if ( KAAPI_ACCESS_IS_FIFO(m)) 
    {
      kaapi_fifo_t* fifo_param = (kaapi_fifo_t*)(task_fmt->off_params[i] + (char*)task->sp);
      kaapi_task_bindfifo(task, i, m, fifo_param);
    }
  }
  
  return 0;
}

