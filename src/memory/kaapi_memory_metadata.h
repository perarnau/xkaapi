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

#include "kaapi_impl.h"

#include "machine/mt/kaapi_mt_bitmap.h"

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
  kaapi_data_t             data[KAAPI_MAX_ADDRESS_SPACE];
  struct kaapi_version_t*  version[KAAPI_MAX_ADDRESS_SPACE];          /* opaque */
  
  kaapi_bitmap64_t  valid_bits;
  kaapi_bitmap64_t  addr_bits;
} kaapi_metadata_info_t;


/** Return the meta data associated to a virtual address @ in the host.
 Return 0 if no meta data is attached.
 */
extern kaapi_metadata_info_t* kaapi_memory_find_metadata( void* ptr );

/** Return the meta data associated to a virtual address @ in the host.
 Return a new allocated data_info if no meta data is attached.
 The newly created data info structure has no
 */
extern kaapi_metadata_info_t* kaapi_memory_findinsert_metadata( void* ptr );

/** Get a copy to the local address space...
 TODO
 */
extern kaapi_pointer_t kaapi_memory_synchronize_metadata( kaapi_metadata_info_t* kdmi );

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

/** Same as the previous call, except that it does not allocate a new kmdi.
 */
extern kaapi_metadata_info_t* kaapi_memory_bind_with_metadata(
                                                       kaapi_address_space_id_t kasid,
                                                       kaapi_metadata_info_t* const kmdi,
                                                       int flag,
                                                       void* ptr,
                                                       size_t size
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

/** Same as the previous call, except that it does not allocate a new kmdi.
 */
extern kaapi_metadata_info_t* kaapi_memory_bind_view_with_metadata(
                                                                   kaapi_address_space_id_t kasid,
                                                                   kaapi_metadata_info_t* const kmdi,
                                                                   int flag,
                                                                   void* ptr,
                                                                   const kaapi_memory_view_t* view
                                                                   );

/** Unbind the address from the address space kasid.
 If the data exist in kasid, then the caller takes the owner ship of the data.
 If the data does not exist in the address space kasid, then the method do nothing, even if the data
 resides in other address spaces.
 */
extern void* kaapi_memory_unbind( kaapi_metadata_info_t* kmdi );


/**/
extern struct kaapi_version_t** kaapi_metadata_info_bind_data(
                                                               kaapi_metadata_info_t* kmdi,
                                                               kaapi_address_space_id_t kasid,
                                                               void* ptr, const kaapi_memory_view_t* view
                                                               );

/*********************************************************************************/
/** WARNING: below there are all low level functions for kaapi_metadata_info_t. **/
/*********************************************************************************/

/**/
static inline void kaapi_metadata_info_unbind_alldata(kaapi_metadata_info_t* kmdi)
{
  kaapi_bitmap_clear_64( &kmdi->valid_bits );
  kaapi_bitmap_clear_64( &kmdi->addr_bits );
}


static inline void kaapi_metadata_info_unbind( kaapi_metadata_info_t* kmdi, kaapi_address_space_id_t kasid )
{
  kaapi_bitmap_unset_64( &kmdi->addr_bits, kasid );
}

static inline int kaapi_metadata_info_clear_data(
                                               kaapi_metadata_info_t* kmdi,
                                               kaapi_address_space_id_t kasid
                                               )
{
  return kaapi_bitmap_unset_64( &kmdi->addr_bits, kaapi_memory_address_space_getlid(kasid) );
}

static inline void kaapi_metadata_info_set_data(
                                               kaapi_metadata_info_t* kmdi,
                                               kaapi_address_space_id_t kasid,
                                                void* ptr, const kaapi_memory_view_t* view
                                               )
{
  const uint16_t lid = kaapi_memory_address_space_getlid( kasid );
  kaapi_assert_debug(lid < KAAPI_MAX_ADDRESS_SPACE);
  kmdi->data[lid].ptr  = kaapi_make_pointer(kasid, ptr);
  kmdi->data[lid].view = *view;
  kmdi->data[lid].mdi = kmdi;
  kaapi_bitmap_set_64(&kmdi->addr_bits, kaapi_memory_address_space_getlid(kasid));
}

static inline int kaapi_metadata_info_has_data(
                                               kaapi_metadata_info_t* kmdi,
                                               kaapi_address_space_id_t kasid
                                               )
{
  return kaapi_bitmap_get_64( &kmdi->addr_bits, kaapi_memory_address_space_getlid(kasid) );
}

/**/
static inline kaapi_data_t* kaapi_metadata_info_get_data(
                                                          kaapi_metadata_info_t* kmdi,
                                                          kaapi_address_space_id_t kasid
                                                          )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  kaapi_assert_debug(kaapi_metadata_info_has_data(kmdi, kasid));
  return &kmdi->data[kaapi_memory_address_space_getlid(kasid)];
}

/**/
static inline int kaapi_metadata_info_is_valid(
                                                kaapi_metadata_info_t* kmdi,
                                                kaapi_address_space_id_t kasid
                                                )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  return kaapi_bitmap_get_64( &kmdi->valid_bits, kaapi_memory_address_space_getlid(kasid));
}

static inline void kaapi_metadata_info_clear_dirty(
                                               kaapi_metadata_info_t* kmdi,
                                               kaapi_address_space_id_t kasid
                                               )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  kaapi_bitmap_set_64(&kmdi->valid_bits, kaapi_memory_address_space_getlid(kasid));
}

static inline void kaapi_metadata_info_set_dirty(
                                                   kaapi_metadata_info_t* kmdi,
                                                   kaapi_address_space_id_t kasid
                                                   )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  kaapi_bitmap_unset_64(&kmdi->valid_bits, kaapi_memory_address_space_getlid(kasid));
}

static inline void kaapi_metadata_info_set_all_dirty_except(
                                                 kaapi_metadata_info_t* kmdi,
                                                 kaapi_address_space_id_t kasid
                                                 )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  kaapi_bitmap_clear_64(&kmdi->valid_bits);
  kaapi_bitmap_set_64(&kmdi->valid_bits, kaapi_memory_address_space_getlid(kasid));
}

static inline int kaapi_metadata_info_clear_dirty_and_check(
                                                            kaapi_metadata_info_t* kmdi,
                                                            kaapi_address_space_id_t kasid
                                                            )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  return (kaapi_bitmap_fetch_and_set_64(&kmdi->valid_bits, kaapi_memory_address_space_getlid(kasid)) == 0);
}

/**/
static inline int kaapi_metadata_info_is_novalid(
                                                  kaapi_metadata_info_t* kmdi
                                                  )
{
  return kaapi_bitmap_empty_64(&kmdi->valid_bits);
}

/**/
static inline void kaapi_metadata_info_set_writer( 
                                                   kaapi_metadata_info_t* kmdi,
                                                   kaapi_address_space_id_t kasid
                                                   )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  kaapi_bitmap_unset_64(&kmdi->valid_bits, kaapi_memory_address_space_getlid(kasid));
}

static inline struct kaapi_version_t* kaapi_metadata_info_get_version(
                                                                kaapi_metadata_info_t* kmdi,
                                                                kaapi_address_space_id_t kasid
                                                                )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  return kmdi->version[kaapi_memory_address_space_getlid(kasid)];
}

static inline struct kaapi_version_t** kaapi_metadata_info_get_version_(
                                                                      kaapi_metadata_info_t* kmdi,
                                                                      kaapi_address_space_id_t kasid
                                                                      )
{
  kaapi_assert_debug(kaapi_memory_address_space_getlid(kasid) < KAAPI_MAX_ADDRESS_SPACE);
  return &kmdi->version[kaapi_memory_address_space_getlid(kasid)];
}

static inline struct kaapi_version_t* kaapi_metadata_info_get_version_by_lid(
                                                                               kaapi_metadata_info_t* kmdi,
                                                                               uint16_t lid
                                                                               )
{
  kaapi_assert_debug(lid < KAAPI_MAX_ADDRESS_SPACE);
  return kmdi->version[lid];
}

static inline struct kaapi_version_t** kaapi_metadata_info_get_version_by_lid_(
                                                                      kaapi_metadata_info_t* kmdi,
                                                                      uint16_t lid
                                                                      )
{
  kaapi_assert_debug(lid < KAAPI_MAX_ADDRESS_SPACE);
  return &kmdi->version[lid];
}

static inline uint16_t kaapi_metadata_info_first_valid( kaapi_metadata_info_t* kmdi )
{
  return ( kaapi_bitmap_first1_64(&kmdi->valid_bits) - 1);
}

/** Return a version that handle a valid copy in order to make a copy to kasid 
 */
static inline struct kaapi_version_t* kaapi_metadata_info_find_onewriter( kaapi_metadata_info_t* kmdi )
{
  uint16_t lid = kaapi_metadata_info_first_valid( kmdi );
  kaapi_assert_debug(lid < KAAPI_MAX_ADDRESS_SPACE);
  return kaapi_metadata_info_get_version_by_lid(kmdi, lid);
}

/* Make a copy of the data into kasid
 */
static inline struct kaapi_version_t** _kaapi_metadata_info_copy_data(
                                                                      kaapi_metadata_info_t* kmdi,
                                                                      kaapi_address_space_id_t kasid,
                                                                      const kaapi_memory_view_t* view    
                                                                      )
{
  uint16_t lid = kaapi_memory_address_space_getlid( kasid );
  kmdi->data[lid].ptr  = kaapi_make_pointer(kasid, 0);
  kmdi->data[lid].view = *view;
  kaapi_memory_view_reallocated(&kmdi->data[lid].view);
  kaapi_metadata_info_clear_dirty(kmdi, kasid);
  return kaapi_metadata_info_get_version_(kmdi, kasid);
}

static inline kaapi_metadata_info_t* kaapi_metadata_info_alloc(void)
{
  kaapi_metadata_info_t* kmdi = (kaapi_metadata_info_t*)malloc( sizeof(kaapi_metadata_info_t) );
  if (kmdi == 0) return 0;
  kaapi_bitmap_clear_64( &kmdi->valid_bits );
  kaapi_bitmap_clear_64( &kmdi->addr_bits );
  return kmdi;
}

#endif /* _KAAPI_MEMORY_METADATA_H_ */