/*
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
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
#include "kaapi_impl.h"
#include "kaapi_staticsched.h"


/* New version.
   This method is called when no entry has been found into the mapping for data referenced by access.
   The method declare in the hash map a newversion 
*/
kaapi_hashentries_t* kaapi_threadgroup_newversion( 
    kaapi_threadgroup_t thgrp, 
    kaapi_hashmap_t*    hmap, 
    int                 tid, 
    kaapi_access_t*     access 
)
{
  kaapi_hashentries_t* entry;
  kaapi_version_t* ver;
  entry = kaapi_hashmap_insert( &thgrp->ws_khm, access->data );
   
  /* here a stack allocation attached with the thread group */
  entry->u.value.last_mode    = KAAPI_ACCESS_MODE_VOID;
  entry->u.value.last_version = 0;
  ver = entry->u.version      = kaapi_threadgroup_allocate_version( thgrp );
  kaapi_assert( ver != 0 );
  ver->writer_thread = -1; /* main thread */
  ver->writer.asid = thgrp->tid2asid[ver->writer_thread];
  ver->writer.addr = access->data;
  ver->writer.size = 0;  /* in fact unknown size, should be set by the task param */
  ver->writer.ith  = -1; /* in fact unknown ith, should be set by the task param */
  return entry;
}


/*
*/
void kaapi_threadgroup_deleteversion( kaapi_threadgroup_t thgrp, kaapi_version_t* ver )
{
}
