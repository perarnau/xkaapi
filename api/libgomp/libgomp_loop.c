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
#include "libgomp.h"
#include <stdio.h>

/* Implementation note.
    This functions are called by all threads.
    Only the master thread call kaapic_foreach_workend in order
    to destroy global team information...
   TG.
*/
void GOMP_loop_end (void)
{
//  printf("%s:: \n", __FUNCTION__);
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  
  /* implicit barrier at the end ? It will deadlock if parallel region task is steal ...*/
  gomp_barrier_wait(&ctxt->teaminfo->barrier);
  
  if (ctxt->threadid == 0)
    kaapic_foreach_workend( self_thread, ctxt->workshare.lwork);
  else
    kaapic_foreach_local_workend( self_thread, ctxt->workshare.lwork );
}

void GOMP_loop_end_nowait (void)
{
//  printf("%s:: \n", __FUNCTION__);
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  if (ctxt->threadid == 0)
    kaapic_foreach_workend( self_thread, ctxt->workshare.lwork);
  else
    kaapic_foreach_local_workend( self_thread, ctxt->workshare.lwork );
}
