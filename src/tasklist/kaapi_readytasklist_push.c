/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** joao.lima@imag.fr
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

static inline int kaapi_onereadytasklist_push(
  kaapi_onereadytasklist_t* ortl, 
  kaapi_taskdescr_t* td 
)
{
  td->next = td->prev = NULL;
  kaapi_atomic_lock( &ortl->lock );
  td->next = ortl->head;
  if( ortl->head != NULL )
    ortl->head->prev = td;
  else
    ortl->tail = td;
  ortl->head = td;
  ortl->size++;
  kaapi_atomic_unlock( &ortl->lock );
  return 0;
}

int kaapi_readylist_push( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority )
{
  kaapi_onereadytasklist_t* ortl;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MIN_PRIORITY) && (priority <= KAAPI_TASKLIST_MAX_PRIORITY) );
  
  ortl = &rtl->prl[priority];
  kaapi_onereadytasklist_push( ortl, td );
  KAAPI_ATOMIC_INCR( &rtl->cnt_tasks );
  return priority;
}
