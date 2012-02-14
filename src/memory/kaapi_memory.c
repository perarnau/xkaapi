/*
** xkaapi
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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


static kaapi_atomic_t count_localid = {0};

/** Address space are 64 bits identifier decomposed in (from higher bit to lower):
    - 4 bits: type/architecture
    - 24 bits (16777216 values): for gid
    - 32 bits for local identifier
*/
kaapi_address_space_id_t kaapi_memory_address_space_create(
  kaapi_globalid_t _gid, 
  int _type, 
  size_t size 
)
{
  uint64_t gid;
  uint64_t type;
  uint64_t lid;

  uint64_t asid = 0;

  /* extra bit used by the user: */
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(0 == (_type & ~0x0F));
#else  
  if (_type & ~0x0F) return -1UL;
#endif
  type = ((uint64_t)_type) << 56ULL;
  
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(0 == (_gid & ~0xFFFFFFFFFFUL));
#else  
  if (_gid & ~0xFFFFFFFFFFUL) return -1UL;
#endif
  gid = ((uint64_t)_gid) << 16ULL;

  lid = (uint64_t)KAAPI_ATOMIC_INCR( &count_localid );

  /* store type in the 8 highest bits */
  asid   = type | gid | lid;
  return (kaapi_address_space_id_t)asid;
}


/**
*/
void kaapi_memory_global_barrier(void)
{
  kaapi_mem_barrier();
#if defined(KAAPI_USE_NETWORK)
  kaapi_network_barrier();
#endif  
}


/**
*/
int kaapi_memory_address_space_fprintf( FILE* file, kaapi_address_space_id_t kasid )
{ 
  return fprintf(file, "[%i, %u]", 
    kaapi_memory_address_space_gettype(kasid),
    kaapi_memory_address_space_getgid(kasid)
  );
}


