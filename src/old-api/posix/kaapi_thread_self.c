/*
** kaapi_thread_self.c
** xkaapi
** 
** Created on Tue Mar 31 15:16:30 2009
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

/* Note that kaapi_self returns either a system scope thread or a processor_scope thread.
   The process scope thread is never return : it is the current running thread on the k-processor.
   This process scope thread is a lazy thread and does not exist most of the time.
*/
kaapi_thread_descr_t* kaapi_self_internal(void)
{
  kaapi_thread_descr_t* td = (kaapi_thread_descr_t*)pthread_getspecific(kaapi_current_thread_key);
  kaapi_assert_debug( td !=0 );

  return td;
}


/* Public interface
*/
kaapi_t kaapi_self(void)
{
  kaapi_t retval;
  kaapi_thread_descr_t* td = kaapi_self_internal();
  kaapi_assert_debug( td !=0 );
  kaapi_assert_debug( td->scope != KAAPI_PROCESS_SCOPE );

  retval.futur = td->futur;
  if (td->scope == KAAPI_SYSTEM_SCOPE) retval.tid = td->th.s.tid;
  else retval.tid = td->th.k.ctxt.tid;
  
  return retval;
}
