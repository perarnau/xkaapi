/*
** kaapi_abort.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
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
#include <stdlib.h>

void* kaapi_malloc_align( unsigned int align_size, size_t size, void** addr_tofree)
{
  /* align_size in bytes */
  if (align_size == 0)
  {
    *addr_tofree = malloc(size);
    return *addr_tofree;
  }

  const kaapi_uintptr_t align_mask = align_size - 1;
  void* retval = (void*)malloc(align_mask + size);
  if (retval != NULL)
  {
    if (addr_tofree !=0)
      *addr_tofree = retval;

    if ((((kaapi_uintptr_t)retval) & align_mask) != 0U)
      retval = (void*)(((kaapi_uintptr_t)retval + align_mask) & ~align_mask);
    kaapi_assert_debug( (((kaapi_uintptr_t)retval) & align_mask) == 0U);
  }

  return retval;
}
