/*
** kaapi_mem.h
** xkaapi
** 
** Created on Jul 2010
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


/* memory management api
 */

/* kaapi_mem_addr_t is a type large
   enough to contain all the addresses
   of all memory spaces.
 */

typedef uintptr_t kaapi_mem_addr_t;


/* address space identifier
 */

typedef unsigned int kaapi_mem_asid_t;


/* kaapi_mem_mapping is a bitmap
   containing a list of address
   mappings as seen by a given
   memory space.
 */

typedef struct kaapi_mem_mapping
{
  struct kaapi_mem_mapping* next;
  unsigned int bitmap;
#define KAAPI_MEM_ASID_MAX 2
  kaapi_mem_addr_t addrs[KAAPI_MEM_ASID_MAX];
} kaapi_mem_mapping_t;


/* a map contains all the mappings
   of the owning space.
 */

typedef struct kaapi_mem_map
{
  kaapi_mem_asid_t asid;
  kaapi_mem_mapping_t* head;
} kaapi_mem_map_t;


/* static inline functions
 */

static inline void kaapi_mem_mapping_init
(kaapi_mem_mapping_t* mapping)
{
  mapping->next = NULL;
  mapping->bitmap = 0;
}

static inline void kaapi_mem_mapping_set
(kaapi_mem_mapping_t* mapping, kaapi_mem_addr_t addr, kaapi_mem_asid_t asid)
{
  mapping->bitmap |= 1 << asid;
  mapping->addrs[asid] = addr;
}

static inline kaapi_mem_addr_t kaapi_mem_mapping_get
(kaapi_mem_mapping_t* mapping, kaapi_mem_asid_t asid)
{
  return mapping->addrs[asid];
}

static inline unsigned int kaapi_mem_mapping_isset
(const kaapi_mem_mapping_t* mapping, kaapi_mem_asid_t asid)
{
  return mapping->bitmap & (1 << asid);
}


/* non inlined api
 */

int kaapi_mem_map_initialize(kaapi_mem_map_t*, kaapi_mem_asid_t);
void kaapi_mem_map_cleanup(kaapi_mem_map_t*);
int kaapi_mem_map_find_or_insert
(kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_mapping_t**);
int kaapi_mem_map_find
(kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_mapping_t**);
int kaapi_mem_map_find_inverse
(kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_mapping_t**);



#endif /* ! KAAPI_MEM_H_INCLUDED */
