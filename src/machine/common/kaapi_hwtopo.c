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
#include "kaapi_procinfo.h"
#if defined(KAAPI_USE_HWLOC)
#  include "hwloc.h"
#endif

#if defined(KAAPI_USE_HWLOC)
/*
*/
static void kaapi_hwcpuset2affinity( 
  kaapi_cpuset_t* affinity, 
  int             nproc, 
  hwloc_cpuset_t  cpuset 
)
{
  kaapi_cpuset_clear( affinity );
  for (int i=0; i<nproc; ++i)
  {
    if (hwloc_cpuset_isset(cpuset, i))
      kaapi_cpuset_set(affinity, i);
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


__attribute__((unused))
static const char* kaapi_kids2string(int nkids, kaapi_processor_id_t* kids)
{
  static char buffer[1024];
  int i, err, size;
  size = 0;
  
  for (i =0; i<nkids; ++i)
  {
    err = sprintf(buffer+size,"%i ", kids[i]);
    kaapi_assert(err >0);
    size += err;
    kaapi_assert(size <1023);
  }
  buffer[size+1] = 0;
  return buffer;
}

#endif

/** Common initialization 
*/
static void kaapi_hw_standardinit(void)
{
  bzero( &kaapi_default_param.memory, sizeof(kaapi_default_param.memory) );

  kaapi_cpuset_clear(&kaapi_default_param.usedcpu);

  /* build the procinfo list */
  kaapi_default_param.kproc_list = (kaapi_procinfo_list_t*)malloc(sizeof(kaapi_procinfo_list_t) );
  kaapi_assert_debug(kaapi_default_param.kproc_list);

  kaapi_default_param.kid2cpu=(unsigned int*)malloc(kaapi_default_param.cpucount*sizeof(unsigned int));
  kaapi_assert_debug(kaapi_default_param.kid2cpu);
  kaapi_default_param.cpu2kid=(unsigned int*)malloc(kaapi_default_param.cpucount*sizeof(unsigned int));
  kaapi_assert_debug(kaapi_default_param.cpu2kid);
  for (int i=0; i<kaapi_default_param.cpucount; ++i)
  {
    kaapi_default_param.kid2cpu[i]= -1;
    kaapi_default_param.cpu2kid[i]= -1;
  }

  kaapi_procinfo_list_init(kaapi_default_param.kproc_list);
  kaapi_mt_register_procs(kaapi_default_param.kproc_list);

#if defined(KAAPI_USE_CUDA)
  kaapi_cuda_register_procs(kaapi_default_param.kproc_list);
#endif /* KAAPI_USE_CUDA */
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
#if defined(KAAPI_USE_HWLOC)
  hwloc_topology_t topology;
  hwloc_obj_t root;
  hwloc_obj_t obj;
  int topodepth, depth;
  int memdepth;
  int idx, ncousin;
  int ncpu;
#endif

  kaapi_hw_standardinit();
  
#if defined(KAAPI_USE_HWLOC)
  /* Allocate and initialize topology object. */
  hwloc_topology_init(&topology);

  /* Perform the topology detection. */
  hwloc_topology_load(topology);

  /* 0: system level */
  topodepth = hwloc_topology_get_depth(topology);
  
  /* count the number of PU */
  root = hwloc_get_obj_by_depth(topology, 0, 0);
  kaapi_default_param.syscpucount = hwloc_cpuset_weight( root->cpuset );

  /* count the depth of the memory hierarchy 
     In order to have lower index in kaapi_default_param.memory
     that represents first cache level.
  */
  memdepth = 0;
  for (depth = 0; depth < topodepth; depth++) 
  {
    obj = hwloc_get_obj_by_depth(topology, depth, 0);
    if (obj->type == HWLOC_OBJ_MACHINE)
    {
      if (obj->arity >1) { ++memdepth; }
    }
    else if (obj->type == HWLOC_OBJ_NODE)
    {
      ++memdepth;
    } 
    else if (obj->type == HWLOC_OBJ_CACHE)
    {
      ++memdepth;
    }
  }

  kaapi_default_param.memory.depth  = memdepth;
  kaapi_default_param.memory.levels = (kaapi_hierarchy_one_level_t*)calloc( memdepth, sizeof(kaapi_hierarchy_one_level_t) );
  for (depth = 0; depth < topodepth; depth++) 
  {
    obj = hwloc_get_obj_by_depth(topology, depth, 0);
    if (obj->type == HWLOC_OBJ_MACHINE)
    {
//printf("Find machine level, memory:%lu\n", (unsigned long)obj->memory.total_memory);
      if (obj->arity >1) 
      {
        --memdepth;
        ncpu = hwloc_cpuset_weight( obj->cpuset );
        kaapi_default_param.memory.levels[memdepth].count = 1;
        kaapi_default_param.memory.levels[memdepth].affinity = (kaapi_affinityset_t*)calloc( 1, sizeof(kaapi_affinityset_t) );
        kaapi_default_param.memory.levels[memdepth].affinity[0].mem_size = obj->memory.total_memory;
        kaapi_default_param.memory.levels[memdepth].affinity[0].ncpu = ncpu = hwloc_cpuset_weight( obj->cpuset );
        kaapi_default_param.memory.levels[memdepth].affinity[0].type = KAAPI_MEM_NODE;
        kaapi_hwcpuset2affinity(
            &kaapi_default_param.memory.levels[memdepth].affinity[0].who,
            KAAPI_MAX_PROCESSOR, 
            obj->cpuset 
      );
      }
    }
    else if ((obj->type == HWLOC_OBJ_NODE) || (obj->type == HWLOC_OBJ_CACHE))
    {
      --memdepth;
      ncousin = (int)kaapi_hw_countcousin( obj );
      kaapi_default_param.memory.levels[memdepth].count = ncousin;
      kaapi_default_param.memory.levels[memdepth].affinity = (kaapi_affinityset_t*)calloc(ncousin, sizeof(kaapi_affinityset_t) );

      /* iterator over all cousins */
      idx = 0;
      while (obj !=0)
      {
        if (obj->type == HWLOC_OBJ_NODE)
          kaapi_default_param.memory.levels[memdepth].affinity[idx].mem_size = obj->memory.local_memory;
        else
          kaapi_default_param.memory.levels[memdepth].affinity[idx].mem_size = obj->attr->cache.size;

        kaapi_default_param.memory.levels[memdepth].affinity[idx].ncpu = ncpu = hwloc_cpuset_weight( obj->cpuset );
        kaapi_default_param.memory.levels[memdepth].affinity[idx].type = (obj->type == HWLOC_OBJ_CACHE ? KAAPI_MEM_CACHE : KAAPI_MEM_NODE);
        kaapi_hwcpuset2affinity(
            &kaapi_default_param.memory.levels[memdepth].affinity[idx].who,
            KAAPI_MAX_PROCESSOR, 
            obj->cpuset 
        );

        ++idx;
        if (obj->next_sibling !=0) 
           obj = obj->next_sibling;
        else {
          obj = obj->next_cousin;
        }
      }
    } 
  }
  /* end of detection of memory hierarchy */
  
  
#if 0
{
  unsigned int i;
  /* display result... */
  printf("Memory hierarchy levels:%i\n", kaapi_default_param.memory.depth);
  printf("System cpu:%i\n", kaapi_default_param.syscpucount);
  printf("Used cpu  :%i\n", kaapi_default_param.cpucount);
  printf("Whole CPU SET:'%s'\n",kaapi_cpuset2string(kaapi_default_param.syscpucount, kaapi_default_param.usedcpu));
  for (depth=0; depth < kaapi_default_param.memory.depth; ++depth)
  {
    printf("level[%i]: #memory:%i \t", depth, kaapi_default_param.memory.levels[depth].count );
    for (i=0; i< kaapi_default_param.memory.levels[depth].count; ++i)
    {
      if (kaapi_cpuset_intersect(kaapi_default_param.memory.levels[depth].affinity[i].who, kaapi_default_param.usedcpu))
      {
        const char* str = kaapi_cpuset2string(kaapi_default_param.syscpucount, kaapi_default_param.memory.levels[depth].affinity[i].who);
        printf("[size:%lu, cpuset:'%s', type:%u]   ", 
          (unsigned long)kaapi_default_param.memory.levels[depth].affinity[i].mem_size,
          str, 
          (unsigned int)kaapi_default_param.memory.levels[depth].affinity[i].type
        );
      }
    }
    printf("\n");
  }
  
  printf("KID2CPU: ");
  for (i=0; i<kaapi_default_param.cpucount; ++i)
    printf("%i ", kaapi_default_param.kid2cpu[i] );
  printf("\n");

  printf("CPU2KID: ");
  for (i=0; i<kaapi_default_param.cpucount; ++i)
    printf("%i ", kaapi_default_param.cpu2kid[i] );
  printf("\n");
}
#endif
  
#endif

  return 0;
}

