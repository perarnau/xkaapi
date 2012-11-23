/*
 ** xkaapi
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
 ** Joao.Lima@imagf.r / joao.lima@inf.ufrgs.br
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

#ifndef _KAAPI_MEMORY_METADATA_H_
#define _KAAPI_MEMORY_METADATA_H_

/** Meta data attached to a pointer and its remote copies on a set of address spaces.
 The data structure store information for at most 64 address spaces.
 This data structure stores all the valid and invalid copies of the data.
 data[i] is valid iff validbits & (1<<i) is not null.
 data[i] is not valid but yet allocated if dirtybits & ( 1<<i) is not null.
 If data[i] is 0, then no data has been allocated in the i-th local
 memory, and both validbits & (1<<i) and dirtybits & (1<<i) are nul.
 The view of each allocated block of data is stored in the kaapi_data_t structure.
 */
typedef struct kaapi_metadata_info_t {
  uint64_t                 validbits;
  kaapi_data_t             data[KAAPI_MAX_ADDRESS_SPACE];
  struct kaapi_version_t*  version[KAAPI_MAX_ADDRESS_SPACE];          /* opaque */
} kaapi_metadata_info_t;


/** Return the meta data associated to a virtual address @ in the host.
 Return 0 if no meta data is attached.
 */
extern kaapi_metadata_info_t* kaapi_memory_find_metadata( void* ptr );

/** Return the meta data associated to a virtual address @ in the host.
 Return a new allocated data_info if no meta data is attached.
 The newly created data info structure has no
 */
extern kaapi_metadata_info_t* kaapi_mem_findinsert_metadata( void* ptr );

/** Get a copy to the local address space...
 TODO
 */
extern kaapi_pointer_t kaapi_memory_synchronize_( kaapi_metadata_info_t* kdmi );

/** Bind an address of the calling virtual space of the process in the address space data structure kasid.
 Once binded the memory will be deallocated if the sticky flag is not set in flag.
 Return the pointer.
 */
extern kaapi_metadata_info_t* kaapi_memory_bind(
                                                kaapi_address_space_id_t kasid,
                                                int                      flag,
                                                void*                    ptr,
                                                size_t                   size
                                                );

/** Bind an address of the calling virtual space of the process in the address space data structure kasid.
 Once binded the memory will be deallocated if the sticky flag is not set in flag.
 Return the pointer.
 */
extern kaapi_metadata_info_t* kaapi_memory_bind_view(
                                                     kaapi_address_space_id_t   kasid,
                                                     int                        flag,
                                                     void*                      ptr,
                                                     const kaapi_memory_view_t* view
                                                     );

/** Unbind the address from the address space kasid.
 If the data exist in kasid, then the caller takes the owner ship of the data.
 If the data does not exist in the address space kasid, then the method do nothing, even if the data
 resides in other address spaces.
 */
extern void* kaapi_memory_unbind( kaapi_metadata_info_t* kmdi );


/**/
extern struct kaapi_version_t** _kaapi_metadata_info_bind_data(
                                                               kaapi_metadata_info_t* kmdi,
                                                               kaapi_address_space_id_t kasid,
                                                               void* ptr, const kaapi_memory_view_t* view
                                                               );


/**/
static inline void _kaapi_metadata_info_unbind_alldata(
                                                       kaapi_metadata_info_t* kmdi
                                                       )
{
  kmdi->validbits = 0UL;
}


/**/
static inline kaapi_data_t* _kaapi_metadata_info_get_data(
                                                          kaapi_metadata_info_t* kmdi,
                                                          kaapi_address_space_id_t kasid
                                                          )
{
  uint16_t lid = kaapi_memory_address_space_getlid( kasid );
  kaapi_assert_debug( lid < KAAPI_MAX_ADDRESS_SPACE );
  return &kmdi->data[lid];
}

/**/
static inline int _kaapi_metadata_info_is_valid(
                                                kaapi_metadata_info_t* kmdi,
                                                kaapi_address_space_id_t kasid
                                                )
{
  uint16_t lid = kaapi_memory_address_space_getlid( kasid );
  kaapi_assert_debug( lid < KAAPI_MAX_ADDRESS_SPACE );
  return (kmdi->validbits & (1UL<<lid)) !=0;
}

/**/
static inline int _kaapi_metadata_info_is_novalid(
                                                  kaapi_metadata_info_t* kmdi
                                                  )
{
  return (kmdi->validbits == 0UL);
}

/**/
static inline void _kaapi_metadata_info_set_writer( 
                                                   kaapi_metadata_info_t* kmdi,
                                                   kaapi_address_space_id_t kasid
                                                   )
{ 
  uint16_t lid = kaapi_memory_address_space_getlid( kasid );
  kaapi_assert_debug( lid < KAAPI_MAX_ADDRESS_SPACE );
  kmdi->validbits = 1UL << lid;
}

#if 0
/**/
extern struct kaapi_version_t* _kaapi_metadata_info_get_version( 
                                                                kaapi_metadata_info_t* kmdi,
                                                                kaapi_address_space_id_t kasid
                                                                );
#endif

/** Return a version that handle a valid copy in order to make a copy to kasid 
 */
extern struct kaapi_version_t* _kaapi_metadata_info_find_onewriter(
                                                                   kaapi_metadata_info_t* kmdi,
                                                                   kaapi_address_space_id_t kasid
                                                                   );

/* Make a copy of the data into kasid
 */
static inline struct kaapi_version_t** _kaapi_metadata_info_copy_data(
                                                                      kaapi_metadata_info_t* kmdi,
                                                                      kaapi_address_space_id_t kasid,
                                                                      const kaapi_memory_view_t* view    
                                                                      )
{
  uint16_t lid = kaapi_memory_address_space_getlid( kasid );
  kmdi->validbits |= 1UL << lid;
  kmdi->data[lid].ptr  = kaapi_make_pointer(kasid, 0);
  kmdi->data[lid].view = *view;
  kaapi_memory_view_reallocated(&kmdi->data[lid].view);
  return &kmdi->version[lid];
}

#endif /* _KAAPI_MEMORY_METADATA_H_ */