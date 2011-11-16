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
#ifndef _KAAPI_NETWORK_C_H_
#define _KAAPI_NETWORK_C_H_ 1

#include "kaapi_compiler.h"

#undef __KA_COMPILER_WEAK
#define __KA_COMPILER_WEAK  

// --------------------------------------------------------------------
/** Implementation note.
    - The network interface allows to use multiple networks.
    (but the current network of XKaapi only use GASNET).
    - The rest of Kaapi try to load the network shared library at runtime
    iff the environnement variable KAAPI_NETWORK is defined (and its value
    is used to initialize the correct device). It call kaapinet_init in 
    the loaded shared library libkanet.
    - The current implementation declare only use a network service to steal
    on the local node: once an active message incomes, the network thread 
    post a request to a random selected thread and wait the result.
    Thus the remote requests are serialized due to the service.
*/

/** This file is the C header file to the C++ network implementation
*/
typedef void (*kaapi_service_t)(int errocode, kaapi_globalid_t source, void* buffer, size_t size);
typedef uint8_t kaapi_service_id_t;

/** To be called prior to any other calls to network
*/
extern __KA_COMPILER_WEAK int kaapi_network_init(int* argc, char*** argv);

/**
*/
extern __KA_COMPILER_WEAK int kaapi_network_finalize(void);

/** Return the local global id 
*/
extern __KA_COMPILER_WEAK kaapi_globalid_t kaapi_network_get_current_globalid(void);

/** Return the number of the nodes in the network
*/
extern __KA_COMPILER_WEAK uint32_t kaapi_network_get_count(void);

/** Allocate data that can be used into rdma operation
*/
extern __KA_COMPILER_WEAK void* kaapi_network_allocate_rdma(size_t size);

/** Deallocate a pointer in a memory region which is rdmable
*/
extern __KA_COMPILER_WEAK void kaapi_network_deallocate_rdma(kaapi_pointer_t, size_t size);

/** Get the segment info for gid 
*/
extern __KA_COMPILER_WEAK int kaapi_network_get_seginfo( kaapi_address_space_t* retval, kaapi_globalid_t gid );

/** Make progress of communication
*/
extern __KA_COMPILER_WEAK void kaapi_network_poll(void);

/**
*/
extern __KA_COMPILER_WEAK void kaapi_network_barrier(void);

/** Do blocking remote  write 
*/
extern __KA_COMPILER_WEAK int kaapi_network_rdma(
  kaapi_globalid_t gid_dest, 
  kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  const void* src, const kaapi_memory_view_t* view_src 
);

/* synchronous am */
extern __KA_COMPILER_WEAK int kaapi_network_am(
  kaapi_globalid_t gid_dest, 
  kaapi_service_id_t service, const void* data, size_t size 
);

#undef __KA_COMPILER_WEAK
#  define __KA_COMPILER_WEAK __attribute__((weak))


#endif /* _KAAPI_NETWORK_C_H_ */
