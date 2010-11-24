/*
** 
** Created on Jun 23 2010
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
#include "kaapi_impl.h"

#if !defined(KAAPI_USE_HWLOC)
/** Initialize hw topo.
*/
int kaapi_hw_init()
{
  return 0;
}

#else

#include "hwloc.h"

/*
*/
static void kaapi_hwcpuset2affinity( kaapi_affinity_t* affinity, int nproc, hwloc_cpuset_t cpuset )
{
  kaapi_affinity_clear( affinity );
  for (int i=0; i<nproc; ++i)
  {
    if (hwloc_cpuset_isset(cpuset, i))
      kaapi_affinity_set(affinity, i);
  }
}

/*
*/
static size_t kaapi_hw_countcousin( hwloc_obj_t obj )
{
  size_t count = 0;
  while (obj !=0)
  {
    ++count;
    if (obj->next_sibling !=0) 
       obj = obj->next_sibling;
    else
    {
      obj = obj->next_cousin;
    }
  }
  return count;
}


static const char* kaapi_affinity2string( int nproc, kaapi_affinity_t* affinity )
{
  static char string[128];
  for (int i=0; i<nproc; ++i)
  {
    if (kaapi_affinity_has(affinity, i))
      string[i] = '1';
    else
      string[i] = '0';
  }
  string[nproc] = 0;
  return string;
}

/*
*/
static void get_cpuset_snprintf( char* string, int size, int nproc, hwloc_cpuset_t cpuset )
{
  for (int i=0; i<nproc; ++i)
  {
    if (hwloc_cpuset_isset(cpuset, i))
      string[i] = '1';
    else
      string[i] = '0';
  }
  string[nproc] = 0;
}

static const char* get_human_type( hwloc_obj_t obj )
{
  switch (obj->type) 
  {
    case HWLOC_OBJ_SYSTEM:
      return "HWLOC_OBJ_SYSTEM";
    case HWLOC_OBJ_MACHINE:
      return "HWLOC_OBJ_MACHINE";
    case HWLOC_OBJ_NODE:
      return "HWLOC_OBJ_NODE";
    case HWLOC_OBJ_SOCKET:
      return "HWLOC_OBJ_SOCKET";
    case HWLOC_OBJ_CACHE:
      return "HWLOC_OBJ_CACHE";
    case HWLOC_OBJ_CORE:
      return "HWLOC_OBJ_CORE";
    case HWLOC_OBJ_PU:
      return "HWLOC_OBJ_PU";
    case HWLOC_OBJ_GROUP:
      return "HWLOC_OBJ_GROUP";
    case HWLOC_OBJ_MISC:
      return "HWLOC_OBJ_MISC";
    default:
      return "undefined";
  }
}

/** Initialize hw topo.
    The goal here is to build for each system resources.
    - List[0..N-1], where N is the depth of the topology
    - List[L1] = list of cores that shared L1 cache
    - List[L2] = list of cores that shared L2 cache
    - List[L3] = list of cores that shared L3 cache
    - List[Memory] = list of cores that shared a memory banc cache
    - List[OS] = list of cores that shared a same administration domain
    
    After the selection of the set of CPU (KAAPI_CPUSET) defined by
    the user, we build a restricted topology to used cores.
*/
int kaapi_hw_init()
{
  hwloc_topology_t topology;
  hwloc_obj_t root;
  hwloc_cpuset_t cpuset;
  hwloc_obj_t obj;
  int topodepth, depth;
  int memdepth;
  int countmachine, index, ncousin;
  char string[128];
  int i;

  /* Allocate and initialize topology object. */
  hwloc_topology_init(&topology);

  /* Perform the topology detection. */
  hwloc_topology_load(topology);

  /* 0: system level */
  topodepth = hwloc_topology_get_depth(topology);
  
  /* count the number of PU */
  root = hwloc_get_obj_by_depth(topology, 0, 0);
  kaapi_default_param.syscpucount = hwloc_cpuset_weight( root->cpuset );

  /* count the depth of the memory hierarchy */
  memdepth = 0;
  for (depth = 0; depth < topodepth; depth++) 
  {
    obj = hwloc_get_obj_by_depth(topology, depth, 0);
    if (obj->type == HWLOC_OBJ_MACHINE)
    {
      if (obj->arity >1) { ++memdepth; countmachine = 1; }
      else countmachine = 0;
    }
    else if (obj->type == HWLOC_OBJ_NODE)
    {
      if (!countmachine) ++memdepth;
    } 
    else if (obj->type == HWLOC_OBJ_CACHE)
    {
      ++memdepth;
    }
  }
  printf("Memory depth: %i\n", memdepth );
  kaapi_default_param.memory.depth  = memdepth;
  kaapi_default_param.memory.levels = (kaapi_hierarchy_one_level_t*)calloc( memdepth, sizeof(kaapi_hierarchy_one_level_t) );
  for (depth = 0; depth < topodepth; depth++) 
  {
    obj = hwloc_get_obj_by_depth(topology, depth, 0);
    if (obj->type == HWLOC_OBJ_MACHINE)
    {
      if (obj->arity ==1) {
        --memdepth;
        kaapi_default_param.memory.levels[memdepth].count = 1;
        kaapi_default_param.memory.levels[memdepth].mem = (kaapi_memory_t*)calloc(1, sizeof(kaapi_memory_t) );
        kaapi_default_param.memory.levels[memdepth].mem[0].size = obj->memory.total_memory;
        kaapi_hwcpuset2affinity(&kaapi_default_param.memory.levels[memdepth].mem[0].who, KAAPI_MAX_PROCESSOR, obj->cpuset );
        kaapi_default_param.memory.levels[memdepth].mem[0].type = KAAPI_MEM_NODE;
        countmachine = 1; 
      }
      else countmachine = 0;
    }
    else if ((obj->type == HWLOC_OBJ_NODE) || (obj->type == HWLOC_OBJ_CACHE))
    {
      if ((!countmachine) || (obj->type == HWLOC_OBJ_CACHE))
      {
        --memdepth;
        ncousin = kaapi_hw_countcousin( obj );
        kaapi_default_param.memory.levels[memdepth].count = ncousin;
        kaapi_default_param.memory.levels[memdepth].mem = (kaapi_memory_t*)calloc(ncousin, sizeof(kaapi_memory_t) );

        /* iterator over all cousins */
        index = 0;
        while (obj !=0)
        {
          if (obj->type == HWLOC_OBJ_NODE)
            kaapi_default_param.memory.levels[memdepth].mem[index].size = obj->memory.local_memory;
          else
            kaapi_default_param.memory.levels[memdepth].mem[index].size = obj->attr->cache.size;

          kaapi_hwcpuset2affinity(&kaapi_default_param.memory.levels[memdepth].mem[index].who, KAAPI_MAX_PROCESSOR, obj->cpuset );
          
          kaapi_default_param.memory.levels[memdepth].mem[index].type = (obj->type == HWLOC_OBJ_CACHE ? KAAPI_MEM_CACHE : KAAPI_MEM_NODE);

          ++index;
          if (obj->next_sibling !=0) 
             obj = obj->next_sibling;
          else {
            obj = obj->next_cousin;
          }
        }
      }
    } 
  }
  /* end of detection of memory hierarchy */
  
  
  /* display result... */
  printf("Memory hierarchy levels:%i\n", kaapi_default_param.memory.depth);
  for (depth=0; depth < kaapi_default_param.memory.depth; ++depth)
  {
    printf("level[%i]: #memory:%i \t", depth, kaapi_default_param.memory.levels[depth].count );
    for (i=0; i< kaapi_default_param.memory.levels[depth].count; ++i)
    {
      printf("[size:%u, cpuset:%s, type:%u]   ", 
        (unsigned int)kaapi_default_param.memory.levels[depth].mem[i].size,
        kaapi_affinity2string(16, &kaapi_default_param.memory.levels[depth].mem[i].who),
        (unsigned int)kaapi_default_param.memory.levels[depth].mem[i].type
      );
    }
    printf("\n");
  }
  
  
  for (depth = 0; depth < topodepth; depth++) 
  {
    obj = hwloc_get_obj_by_depth(topology, depth, 0);
    printf("Object depth: %i, index: 0, has type:%s\n", depth, get_human_type(obj) );
    printf("\t->");
    /* iterate over the same sub group of sibling objects */
    while (obj !=0)
    {
      get_cpuset_snprintf( string, 128, 4, obj->cpuset );
      printf("[logical index:%i, type:%s cpuset: %s] ", obj->logical_index, get_human_type(obj), string);
      if (obj->next_sibling !=0) 
         obj = obj->next_sibling;
      else
      {
        obj = obj->next_cousin;
        if (obj !=0) printf(" || ");
      }
    }
    printf("\n");

#if 0
    printf("*** Objects at level %d\n", depth);
    for (i = 0; 
         i < hwloc_get_nbobjs_by_depth(topology, depth); 
         i++) 
    {
      hwloc_obj_snprintf(string, sizeof(string), topology,
                 hwloc_get_obj_by_depth(topology, depth, i),
                 "#", 0);
      printf("Index %u: %s\n", i, string);
    }
#endif
  }

  return 0;
}

#endif
