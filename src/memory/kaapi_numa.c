/*
 ** xkaapi
 ** 
 ** Copyright 2010 INRIA.
 **
 ** Contributors :
 **
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

#if defined(KAAPI_USE_NUMA)

#include <numaif.h>
#include <numa.h>


/*
*/
int kaapi_numa_get_page_node(const void* addr)
{
  const unsigned long flags = MPOL_F_NODE | MPOL_F_ADDR;
  kaapi_bitmap_value_t nodemask;
  kaapi_bitmap_value_clear( &nodemask );
  const int err = get_mempolicy
    (NULL, (unsigned long*)&nodemask, KAAPI_MAX_PROCESSOR, (void*)addr, flags);

  if (err || kaapi_bitmap_value_empty(&nodemask))
  {
    errno = EINVAL;
    return -1;
  }

  return kaapi_bitmap_first1_and_zero(&nodemask)-1;
}

/*
*/
int kaapi_numa_bind_bloc1dcyclic
 (const void* addr, size_t size, size_t blocsize)
{
  const int mode = MPOL_BIND;
  const unsigned int flags = MPOL_MF_STRICT | MPOL_MF_MOVE;
  const char* base = (const char*)addr;
  unsigned long maxnode;
  struct bitmask *nodemask;
  
  if ((blocsize & 0xfff) !=0) /* should be divisible by 4096 */
  {
    errno = EINVAL;
    return -1;
  }
  if (((uintptr_t)addr & 0xfff) !=0) /* should aligned on page boundary */
  {
    errno = EFAULT;
    return -1;
  }
  
  maxnode = numa_num_configured_nodes();
  nodemask = numa_allocate_nodemask();
  int node = 0;
  while (size >0)
  {
    numa_bitmask_setbit(nodemask, node);
    if (mbind((void*)base, blocsize, mode, nodemask->maskp, nodemask->size, flags))
    {
#if 0
      printf("***Cannot mbind address: %p, size:%i on node:%i\n", (void*)base, (int)blocsize, node );
      fflush(stdout);
#endif
      return -1;
    }
#if 0
    else {
      printf("***mbind address: %p, size:%i on node:%i\n", (void*)base, (int)blocsize, node );
      fflush(stdout);
    }
#endif
    numa_bitmask_clearbit(nodemask, node);
    ++node;
    if (node >= maxnode) node = 0;  
    base += blocsize;
    size -= blocsize;
  }
  numa_bitmask_free( nodemask );
  return 0;
}


/*
*/
int kaapi_numa_bind(const void* addr, size_t size, int node)
{
  const int mode = MPOL_BIND;
  const unsigned int flags = MPOL_MF_STRICT | MPOL_MF_MOVE;
  const unsigned long maxnode = KAAPI_MAX_PROCESSOR;
  
  if ((size & 0xfff) !=0) /* should be divisible by 4096 */
  {
    errno = EINVAL;
    return -1;
  }
  if (((uintptr_t)addr & 0xfff) !=0) /* should aligned on page boundary */
  {
    errno = EFAULT;
    return -1;
  }
  
  unsigned long nodemask
    [KAAPI_MAX_PROCESSOR / (8 * sizeof(unsigned long))];

  memset(nodemask, 0, sizeof(nodemask));

  nodemask[id / (8 * sizeof(unsigned long))] |=
    1UL << (id % (8 * sizeof(unsigned long)));

  if (mbind((void*)addr, size, mode, nodemask, maxnode, flags))
    return -1;

  return 0;
}



#if 0
int kaapi_numa_initialize(void)
{
  numa_set_bind_policy(1);
  numa_set_strict(1);
  return 0;
}



int kaapi_numa_alloc(void** addr, size_t size, kaapi_numaid_t id)
{
#define NUMA_FUBAR_FIX 1
#if NUMA_FUBAR_FIX /* bug: non reentrant??? */
  *addr = NULL;
  if (posix_memalign(addr, 0x1000, size) == 0)
  {
    if (kaapi_numa_bind(*addr, size, id))
    { free(addr); *addr = NULL; }
  }
#else
  *addr = numa_alloc_onnode(size, (int)id);
#endif
  return (*addr == NULL) ? -1 : 0;
}

void kaapi_numa_free(void* addr, size_t size)
{
#if NUMA_FUBAR_FIX
  free(addr);
#else
  numa_free(addr, size);
#endif
}

/*
*/
int kaapi_numa_alloc_with_procid
(void** addr, size_t size, kaapi_procid_t procid)
{
  const kaapi_numaid_t numaid = kaapi_numa_procid_to_numaid(procid);
  return kaapi_numa_alloc(addr, size, numaid);
}


kaapi_numaid_t kaapi_numa_procid_to_numaid(kaapi_procid_t procid)
{
  return (kaapi_numaid_t)numa_node_of_cpu((int)procid);
}
#endif

#else //if defined(KAAPI_USE_NUMA)

int kaapi_numa_get_page_node(const void* addr)
{ 
  errno = ENOENT;
  return -1; 
}

int kaapi_numa_bind_bloc1dcyclic
 (const void* addr, size_t size, size_t blocsize)
{
  errno = ENOENT;
  return -1; 
}

int kaapi_numa_bind(const void* addr, size_t size, int node)
{
  errno = ENOENT;
  return -1; 
}
#endif // if defined(KAAPI_USE_NUMA)
