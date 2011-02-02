/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threadctxts.
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

/**
*/
int kaapi_threadgroup_synchronize(kaapi_threadgroup_t thgrp )
{
  /* update the list of version in the hashmap to suppress reference to previously executed task */
  for (int i=0; i<KAAPI_HASHMAP_SIZE; ++i)
  {
    kaapi_hashentries_t* entry = _get_hashmap_entry(&thgrp->ws_khm, i);
    while (entry != 0)
    {
      kaapi_version_t* ver = entry->u.version;
      if (ver->writer_mode != KAAPI_ACCESS_MODE_VOID)
      {
        kaapi_address_space_id_t asid = kaapi_threadgroup_tid2asid(thgrp, -1);
        /* */
        kaapi_memory_copy( asid, (kaapi_pointer_t)ver->orig.addr, &ver->orig.view, 
                           ver->writer.asid, (kaapi_pointer_t)ver->writer.addr, &ver->writer.view );
      }
      entry = entry->next;
    }
  }
}