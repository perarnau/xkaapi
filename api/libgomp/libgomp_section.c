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
#include "kaapi_impl.h"

static struct current_selection {
  kaapi_thread_context_t* master;
  int                     maxvalue;
  kaapi_atomic_t          iter;
} selector = { 0, 0, {0} };

unsigned GOMP_sections_start (unsigned maxsec)
{
  kaapi_thread_context_t* thread = kaapi_self_thread_context();
  GOMP_critical_start();
  if (selector.master == 0)
  {
    selector.master   = thread;
    selector.maxvalue = maxsec;
    KAAPI_ATOMIC_WRITE(&selector.iter, 0);
  }
  GOMP_critical_end();

  return GOMP_sections_next();
}

unsigned GOMP_sections_next (void)
{
  unsigned retval = KAAPI_ATOMIC_INCR(&selector.iter);
  if (retval == selector.maxvalue)
  { /* reset to 0 */
    GOMP_critical_start();
    selector.master = 0;
    selector.maxvalue = 0;
    KAAPI_ATOMIC_WRITE(&selector.iter, 0);
    GOMP_critical_start();
  }
  return (retval > selector.maxvalue) ? 0 : retval;
}

void GOMP_parallel_sections_start (
    void (*fn) (void *), 
    void *data,
    unsigned num_threads, 
    unsigned count
)
{
  printf("%s:: \n", __FUNCTION__);
}

void GOMP_sections_end (void)
{
  printf("%s:: \n", __FUNCTION__);
}

void GOMP_sections_end_nowait (void)
{
  printf("%s:: \n", __FUNCTION__);
}


