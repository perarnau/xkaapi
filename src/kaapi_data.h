/*
** kaapi_data.h
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
#ifndef _KAAPI_DATA_H_
#define _KAAPI_DATA_H_ 1


/** Maximal different memory spaces on multiprocessors:
    - CPU memory: index 0
    - accelerator memory (GPU)
  This constant will be suppress for supporting distributed memory
  over network.
*/
#define KAAPI_MAX_MEMORY_SPACE 8

/** Whatis a pointer into a local memory
*/
typedef struct kaapi_local_data_info_t {
  kaapi_uintptr_t ptr;
} kaapi_local_data_info_t;


/** Global information about a memory @ of the host CPU.
    This data structure stores all the valid and invalid copies of the data.
    lptr[i].ptr is valid iff validbits & (1<<i) is not null.
    If lptr[i].ptr is 0, then no data has been allocated in the i-th local
    memory. 
    The size represents the size in bytes of the array.
*/
typedef struct kaapi_data_info_t {
  unsigned int            validbits;
  size_t                  size;
  kaapi_local_data_info_t lptr[KAAPI_MAX_MEMORY_SPACE];
} kaapi_data_info_t;


/** Should represent the memory of a device (CPU or a GPUs)
*/
typedef struct kaapi_local_memory_t {
  unsigned int       index;                                        /* to be used in kaapi_data_info_t */
  kaapi_uintptr_t   (*allocate)(size_t size);                      /* allocate memory */
  void              (*deallocate)(kaapi_uintptr_t, size_t size);   /* free allocated memory */
} kaapi_local_memory_t;



/** Global functions
*/

/** Allocate in the local memory kml a contiguous array of size size byte.
    Initialize the kdi data structure if not null.
    Return the kdi pointer.
    If the allocation failed, then kdi.lptr[klm->index].ptr is null.
*/
extern kaapi_data_info_t* kaapi_mem_allocate( kaapi_local_memory_t* klm, kaapi_data_info_t* kdi, size_t size );

/** Deallocate in the local memory kml the array referenced by kdi.
    If no data resides in klm, then do nothing.
    Else the data is marked as invalid (free) and could be reused in further allocation.
*/
extern void kaapi_mem_deallocate( kaapi_local_memory_t* klm, kaapi_data_info_t* kdi );

/** Bind an address in the local memory kml as a contiguous array of size size byte into the kdi meta data.
    Return the kdi pointer.
*/
extern kaapi_data_info_t* kaapi_mem_bind( kaapi_local_memory_t* klm, kaapi_data_info_t* kdi, void* ptr, size_t size );

/** Unbind the address in the local memory kml in the kdi meta data.
*/
extern void kaapi_mem_unbind( kaapi_local_memory_t* klm, kaapi_data_info_t* kdi );

/** Copy to klm the memory pointed by one of the valid entries in kdi.
    The memory in klm should have been allocated by a previous call to kaapi_mem_allocate.
    Update kdi to mark the copy of the data into klm has valid.
    The call in no blocking call.... A specifier un peu mieux....
*/
extern void kaapi_mem_copy( kaapi_local_memory_t* klm, kaapi_data_info_t* kdi );

/** Return the meta data associated to a virtual address @ in the host.
    Return 0 if no meta data is attached.
*/
extern kaapi_data_info_t* kaapi_mem_find( void* ptr );

/** Return the meta data associated to a virtual address @ in the host.
    Return a new allocated data_info if no meta data is attached.
    The newly created data info structure has no
*/
extern kaapi_data_info_t* kaapi_mem_findinsert( void* ptr, size_t size );

#endif /*_KAAPI_DATA_H_*/
