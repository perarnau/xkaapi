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

/** \ingroup ADAPTIVE
*/
struct kaapi_thief_iterator_t* kaapi_thiefiterator_tail( kaapi_task_t* task )
{
  kaapi_taskadaptive_arg_t* adapt_arg = kaapi_task_getargst(task, kaapi_taskadaptive_arg_t);
  kaapi_assert_debug( kaapi_task_is_splittable(task) );
  kaapi_assert_debug( task->body == (kaapi_task_body_t)kaapi_taskadapt_body );
  kaapi_stealcontext_t* sc = (kaapi_stealcontext_t*)adapt_arg->shared_sc.data;
  /* tail is always a valid pointer (if read was atomic) */
  return (struct kaapi_thief_iterator_t*)sc->thieves.list.tail;
}

/** \ingroup ADAPTIVE
*/
struct kaapi_thief_iterator_t* kaapi_thiefiterator_prev( struct kaapi_thief_iterator_t* pos )
{
  kaapi_thiefadaptcontext_t* arg = (kaapi_thiefadaptcontext_t*)pos;
  /* prev is always a valid pointer (if read was atomic) */
  return (struct kaapi_thief_iterator_t*)arg->prev;
  return 0; 
}

