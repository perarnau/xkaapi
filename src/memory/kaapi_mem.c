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


#include "kaapi_impl.h"
#include "kaapi_mem.h"


/* exported */

int kaapi_mem_map_initialize
(kaapi_mem_map_t* map, kaapi_mem_asid_t asid)
{
  map->asid = asid;
  map->head = NULL;
  return 0;
}

void kaapi_mem_map_cleanup(kaapi_mem_map_t* map)
{
  kaapi_mem_mapping_t* pos = map->head;

  while (pos != NULL)
  {
    kaapi_mem_mapping_t* const tmp = pos;
    pos = pos->next;
    free(tmp);
  }

  map->head = NULL;
}

int kaapi_mem_map_find
(
 kaapi_mem_map_t* map,
 kaapi_mem_addr_t laddr,
 kaapi_mem_mapping_t** mapping
)
{
  /* find a mapping in the map such that
     addrs[map->asid] == laddr.
  */

  kaapi_mem_mapping_t* pos;

  *mapping = NULL;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    /* pos->addrs[map->asid] always set */
    if (pos->addrs[map->asid] == laddr)
    {
      *mapping = pos;
      return 0;
    }
  }

  return -1;
}

int kaapi_mem_map_find_or_insert
(
 kaapi_mem_map_t* map,
 kaapi_mem_addr_t laddr,
 kaapi_mem_mapping_t** mapping
)
{
  /* see comments in the above function.
     if no mapping is found, create one.
   */

  const int res = kaapi_mem_map_find
    (map, laddr, mapping);

  if (res != -1)
    return 0;

  *mapping = malloc(sizeof(kaapi_mem_mapping_t));
  if (*mapping == NULL)
    return -1;

  /* identity mapping */
  kaapi_mem_mapping_init(*mapping);
  kaapi_mem_mapping_set(*mapping, laddr, map->asid);

  /* link against others */
  (*mapping)->next = map->head;
  map->head = *mapping;

  return 0;
}

int kaapi_mem_map_find_inverse
(
 kaapi_mem_map_t* map,
 kaapi_mem_addr_t raddr,
 kaapi_mem_mapping_t** mapping
)
{
  /* given a remote address, find the
     corresponding host address. greedy
     inverted search, exhaust all the
     memory mapping set.
   */

  kaapi_mem_mapping_t* pos;
  kaapi_mem_asid_t asid;

  *mapping = NULL;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    for (asid = 0; asid < KAAPI_MEM_ASID_MAX; ++asid)
    {
      if (kaapi_mem_mapping_isset(pos, asid))
      {
	*mapping = pos;
	return 0;
      }
    }
  }

  return -1;
}
