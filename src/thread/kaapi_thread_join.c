/*
** kaapi_thread_join.c
** xkaapi
** 
** Created on Tue Mar 31 15:16:38 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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


int kaapi_join(kaapi_t thread, void **value_ptr)
{
  if (thread == kaapi_self ()) return EDEADLK;
  if (thread->_detachstate ==1) return EINVAL;
  
  int err = pthread_mutex_lock( &thread->_mutex_join );
  if (err !=0) return err;
  while (thread->_state != KAAPI_THREAD_TERMINATED)
  {
    err = pthread_cond_wait( &thread->_cond_join, &thread->_mutex_join );
    if (err !=0) {
      xkaapi_assert ( 0 == pthread_mutex_unlock( &thread->_mutex_join ));
      return err;
    }
  }
  *value_ptr = thread->_return_value;
  err = pthread_mutex_unlock( &thread->_mutex_join );
  if (err !=0) return err;
  return 0;
}
