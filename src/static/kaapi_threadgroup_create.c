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
** threadctxts.
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
int kaapi_threadgroup_create(kaapi_threadgroup_t* pthgrp, int size )
{
  int i, j;
  int error = 0;
  kaapi_threadgroup_t thgrp = 0;
  kaapi_processor_t* proc = 0;  
  
  if (pthgrp ==0) return EINVAL;
  thgrp = (kaapi_threadgroup_t)malloc(sizeof(kaapi_threadgrouprep_t));
  if (thgrp ==0)
  {
    error = ENOMEM;
    goto return_error_0;
  }
  
  /* */
  proc = _kaapi_get_current_processor();

  thgrp->group_size  = size;
  thgrp->startflag   = 0;
  thgrp->mainthread  = 0;
  thgrp->mainctxt    = proc->thread;
  KAAPI_ATOMIC_WRITE(&thgrp->countend, 0);
  thgrp->waittask    = 0;
  thgrp->threadctxts = malloc( size* sizeof(kaapi_thread_context_t*) );
  if (thgrp->threadctxts ==0) 
  {
    error = ENOMEM;
    goto return_error_1;
  }
  thgrp->threads    = malloc( size* sizeof(kaapi_thread_t*) );
  if (thgrp->threads ==0) 
  {
    error = ENOMEM;
    goto return_error_15;
  }
  
  /* here may be dispatch allocation of all processors */
  for (i=0; i<size; ++i)
  {
    thgrp->threadctxts[i] = kaapi_context_alloc( proc );
    if (thgrp->threadctxts[i] ==0) 
    {
      size = i;
      error = ENOMEM;
      goto return_error_2;
    }
    /* init the thread from the thread context */
    thgrp->threads[i] = kaapi_threadcontext2thread(thgrp->threadctxts[i]);
  }
  
  error =pthread_mutex_init(&thgrp->mutex, 0);
  if (error !=0) goto return_error_2;

  error =pthread_cond_init(&thgrp->cond, 0);
  if (error !=0) goto return_error_3;

  /* ok */
  thgrp->step     = -1;
  thgrp->state    = KAAPI_THREAD_GROUP_CREATE_S;
  *pthgrp = thgrp;
  return 0;

return_error_3:
  pthread_mutex_destroy(&thgrp->mutex);

return_error_2:
  for (j=0; j<size; ++j)
    kaapi_context_free(thgrp->threadctxts[j]);

  free( thgrp->threads );
  
return_error_15:
  free( thgrp->threadctxts );
  thgrp->group_size = 0;
  thgrp->threadctxts = 0;
  
return_error_1:
  free( thgrp );

return_error_0:
  return error;  
}
