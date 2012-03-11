/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
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


void
komp_barrier_init (struct komp_barrier *barrier, unsigned int num)
{
  barrier->nthreads = num;
  KAAPI_ATOMIC_WRITE (&barrier->cycle, 0);
  memset (barrier->count, 0, BAR_CYCLES * CACHE_LINE_SIZE);
}

void
komp_barrier_destroy (struct komp_barrier *barrier)
{
  barrier->nthreads = -1;
  KAAPI_ATOMIC_WRITE (&barrier->cycle, -1);
  memset (barrier->count, -1, BAR_CYCLES * CACHE_LINE_SIZE);
}

void
komp_barrier_wait (struct komp_barrier *barrier)
{
  int current_cycle = KAAPI_ATOMIC_READ (&barrier->cycle);
  int next_cycle = (current_cycle + 1) % BAR_CYCLES;
  int nthreads = barrier->nthreads;

  /* _barrier_ call generated from a _single_ construct: Only the
     thread performing the single body (creating OpenMP tasks) is
     waiting for completion of created tasks. */
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->inside_single)
    {
      if (ctxt->icv.threadid == 0)
	{
	  ctxt->inside_single = 0;
	  kaapic_sync ();
	}
    }
  else
    {
      int nb_arrived = KAAPI_ATOMIC_INCR ((kaapi_atomic_t *)&barrier->count[current_cycle * CACHE_LINE_SIZE]);
      
      if (nb_arrived == nthreads)
	{
	  int cycle_to_clean = (next_cycle + 1) % BAR_CYCLES;
	  
	  KAAPI_ATOMIC_WRITE_BARRIER (&barrier->cycle, next_cycle);
	  KAAPI_ATOMIC_WRITE ((kaapi_atomic_t *)&barrier->count[cycle_to_clean * CACHE_LINE_SIZE], 0);
	}
      else
	{
	  while (KAAPI_ATOMIC_READ (&barrier->cycle) != next_cycle)
	    {
	      kaapi_slowdown_cpu ();
	    }
	}
    }
}

void GOMP_barrier (void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->teaminfo ==0) /* not in parallel region */
    return;
  komp_barrier_wait (&ctxt->teaminfo->barrier);
  
  /* barrier should reset single ? */  
}
