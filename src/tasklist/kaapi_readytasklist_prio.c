/*
 ** xkaapi
 **
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
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

/*
 * max_prio - maximum priority of a given kproc
 * min_prio - minimum priority of a given kproc
 * inc_prio - increment
 * Used to iterate from max_prio to min_prio inclusive
 */
void
kaapi_readylist_get_priority_range(
                                   int* const min_prio,
                                   int* const max_prio,
                                   int* const inc_prio
                                   )
{
#if defined(KAAPI_USE_CUDA)
  if( kaapi_processor_get_type(kaapi_get_current_processor()) == KAAPI_PROC_TYPE_CUDA )
  {
    *min_prio = KAAPI_TASKLIST_MAX_PRIORITY+1;
    *max_prio = KAAPI_TASKLIST_MIN_PRIORITY;
    *inc_prio = 1;
    return;
  }
#endif
  *min_prio = KAAPI_TASKLIST_MIN_PRIORITY-1;
  *max_prio = KAAPI_TASKLIST_MAX_PRIORITY;
  *inc_prio = -1;
}
