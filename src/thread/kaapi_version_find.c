/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
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

/*
*/
kaapi_version_t* kaapi_version_findinsert( 
    kaapi_thread_context_t* thread,
    kaapi_tasklist_t*       tl,
    const void*             addr 
)
{
  kaapi_hashentries_t* entry;
  kaapi_version_t* version;
  
  entry   = kaapi_big_hashmap_findinsert( &thread->kversion_hm, addr );
  version = entry->u.version;
  if (version !=0)
    return version;

  version = entry->u.version
      = (kaapi_version_t*)kaapi_tasklist_allocate( tl, sizeof(kaapi_version_t) );

  version->last_mode       = KAAPI_ACCESS_MODE_VOID;

#if defined(KAAPI_DEBUG)
  version->handle          = 0;
  version->writer_task     = 0;
  version->writer_tasklist = 0;
#endif
  
  return entry->u.version;
}
