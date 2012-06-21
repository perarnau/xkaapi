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

static void
komp_ordered_sync (kompctxt_t *ctxt, int it_min, int it_max)
{
  while (!((ctxt->teaminfo->current_ordered_index >= it_min) && (ctxt->teaminfo->current_ordered_index < it_max)))
    kaapi_slowdown_cpu ();
}

void GOMP_ordered_start (void)
{
  kompctxt_t *ctxt = komp_get_ctxt ();
  int it_min = ctxt->workshare->cur_start;
  int it_max = ctxt->workshare->cur_end;

  komp_ordered_sync (ctxt, it_min, it_max);
}

void GOMP_ordered_end (void)
{
  kompctxt_t *ctxt = komp_get_ctxt ();
  __sync_fetch_and_add (&ctxt->teaminfo->current_ordered_index, 1);
  kaapi_writemem_barrier ();
}
