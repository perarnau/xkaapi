/*
** kaapi_mem.h
** xkaapi
** 
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
#ifndef KAAPI_MEM_H_INCLUDED
# define KAAPI_MEM_H_INCLUDED

#include <sys/types.h>
#include <stdint.h>

/* kaapi_mem_addr_t is a type large enough to
   contain all the addresses of all memory spaces.  */
typedef uintptr_t kaapi_mem_addr_t;

/* address space identifier
 */
typedef unsigned int kaapi_mem_asid_t;

#define KAAPI_MEM_ASID_MAX 32
typedef struct kaapi_mem_data_t {
    kaapi_mem_addr_t addr[KAAPI_MEM_ASID_MAX];

    struct kaapi_mem_data_t* parent;
    unsigned int dirty_bits;
    unsigned int addr_bits;
} kaapi_mem_data_t;

typedef struct kaapi_mem_host_map_t {
    kaapi_mem_asid_t asid;
    kaapi_big_hashmap_t hmap; /* TODO remove */
} kaapi_mem_host_map_t;

void kaapi_mem_init( void );

void kaapi_mem_destroy( void );

#if defined(KAAPI_USE_CUDA)
int kaapi_mem_sync_data( kaapi_data_t*, cudaStream_t );
#endif

#endif /* ! KAAPI_MEM_H_INCLUDED */
