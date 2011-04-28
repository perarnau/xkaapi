/*
** kaapi_thread_clear.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
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
#include <strings.h>
#include <stddef.h>

/**
*/
int kaapi_thread_clear( kaapi_thread_context_t* thread )
{
  kaapi_assert_debug( thread != 0);

  thread->sfp        = thread->stackframe;
  thread->esfp       = thread->stackframe;
  thread->sfp->sp    = thread->sfp->pc  = thread->task; /* empty frame */
  thread->sfp->sp_data = (char*)&thread->data; /* empty frame */
  thread->sfp->tasklist= 0;

  thread->the_thgrp  = 0;
  thread->unstealable= 0;
  thread->partid     = -10; /* out of bound value */

  thread->_next      = 0;
  thread->_prev      = 0;
  thread->asid       = 0;
  thread->affinity[0]= ~0UL;
  thread->affinity[1]= ~0UL;

  thread->wcs        = 0;

  /* zero all bytes from static_reply until end of sc_data */
  bzero(&thread->static_reply, (ptrdiff_t)(&thread->sc_data+1)-(ptrdiff_t)&thread->static_reply );

#if !defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  thread->thgrp      = 0;
#endif
  return 0;
}


/*
*/
void kaapi_thread_set_unstealable(unsigned int fu)
{
  kaapi_thread_context_t* const thread = kaapi_self_thread_context();
  kaapi_sched_lock(&thread->proc->lock);
  thread->unstealable = fu;
  kaapi_sched_lock(&thread->proc->lock);
}
