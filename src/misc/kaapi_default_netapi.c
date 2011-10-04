/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** vincent.danjean@imag.fr
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
#include "kaapi_network.h"
#include "kaapi_compiler.h"


#if 1//ndef KAAPI_USE_NETWORK
/* Weak symbols that will be overloaded by libkanet when
 * linked with it
 */

/**
*/
__KA_COMPILER_WEAK int kaapi_network_init (int* argc, char*** argv) 
{
///  printf("Use default weak symbol...\n");
  return 0;
}

/**
*/
__KA_COMPILER_WEAK int kaapi_network_finalize(void)
{  
  return 0;
}


/**
*/
__KA_COMPILER_WEAK kaapi_globalid_t kaapi_network_get_current_globalid(void)
{
  return 0;
}


/**
*/
__KA_COMPILER_WEAK uint32_t kaapi_network_get_count(void)
{
  return 1;
}


__KA_COMPILER_WEAK void kaapi_network_poll(void)
{
}

__KA_COMPILER_WEAK void kaapi_network_barrier(void)
{
}

__KA_COMPILER_WEAK int kaapi_network_get_seginfo(kaapi_address_space_t* retval,
			      kaapi_globalid_t gid )
{
  retval->segaddr = 0;
  retval->segsize = (size_t)-1;
  return 0;
}

__KA_COMPILER_WEAK int kaapi_network_rdma(
  kaapi_globalid_t gid_dest, 
  kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  const void* src, const kaapi_memory_view_t* view_src 
)
{
  return 0;
}

__KA_COMPILER_WEAK void* kaapi_network_allocate_rdma(size_t size)
{
  return 0;
}

#endif