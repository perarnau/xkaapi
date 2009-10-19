/*
** kaapi_init.c
** ckaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#include "kaapi_stealapi.h"
#include <stdlib.h>

/*
*/
kaapi_paramrt_t default_param;

/*
*/
kaapi_attr_t kaapi_default_attr;

/* current processor */
pthread_key_t kaapi_current_processor_key;

/*
*/
pthread_key_t kaapi_current_thread_key;

/**
*/
int kaapi_min_index_cpu_set[256]; 

#if defined(KAAPI_USE_SCHED_AFFINITY)
/*
*/
int  kaapi_countcpu = 0;

/*
*/
int* kaapi_kproc2cpu = 0;

/*
*/
int* kaapi_cpu2kproc = 0;
#endif

#if defined(HAVE_NUMA_H) 
/*
*/
kaapi_kprocneighbor_t* kaapi_kprocneighbor = 0;


static kaapi_neighbor_t idkoiff_level2[] = {
  {2, {1, 0} }, /* 0 */
  {2, {0, 1} }, /* 1 */
  {2, {3, 2} }, /* 2 */
  {2, {2, 3} }, /* 3 */
  {2, {5, 4} }, /* 4 */
  {2, {4, 5} }, /* 5 */
  {2, {7, 6} }, /* 6 */
  {2, {6, 7} }, /* 7 */
  {2, {9, 8} }, /* 8 */
  {2, {8, 9} }, /* 9 */
  {2, {11,10}}, /* 10 */
  {2, {10,11}}, /* 11 */
  {2, {13,12}}, /* 12 */
  {2, {12,13}}, /* 13 */
  {2, {15,14}}, /* 14 */
  {2, {14,15}}  /* 15 */
};
#endif

/*
 */
kaapi_global_key kaapi_global_keys[KAAPI_KEYS_MAX];

/**
 */
void _kaapi_dummy(void* foo)
{
}

/**
*/
void __attribute__ ((constructor)) my_init(void)
{
  int i;
  default_param.stacksize = getpagesize()*4;

  ckaapi_assert( 0 == pthread_key_create( &kaapi_current_thread_key, 0 ) );
  kaapi_thread_descr_t* td_main_thread;  
  td_main_thread = allocate_thread_descriptor(KAAPI_SYSTEM_SCOPE, 1);
  td_main_thread->_scope = KAAPI_SYSTEM_SCOPE;
  pthread_setspecific( kaapi_current_thread_key, td_main_thread );
  
  /* init dataspecific table */
  for (i = 0; i < KAAPI_KEYS_MAX; i++)
  {
    kaapi_global_keys[i].dest = &_kaapi_dummy;
    kaapi_global_keys[i].next = i+1;
  }
  kaapi_global_keys[KAAPI_KEYS_MAX - 1].next = -1;

  /* should be the first kaapi key with value == 0 */
  ckaapi_assert( 0 == pthread_key_create( &kaapi_current_processor_key, 0 ) );

  /* compute the number of cpu of the system */
#if defined(KAAPI_USE_LINUX)
  default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(KAAPI_USE_APPLE)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(default_param.syscpucount);
    sysctl(mib, 2, &default_param.syscpucount, &len, 0, 0);
  }
#else
  #warning "Could not compute number of physical cpu of the system."
  default_param.syscpucount = 1;
#endif
  
  default_param.cpucount  = default_param.syscpucount;
  default_param.usecpuset = 0;
#if defined(KAAPI_USE_SCHED_AFFINITY)

  kaapi_kproc2cpu = (int*)calloc( default_param.syscpucount, sizeof(int) );
  kaapi_cpu2kproc = (int*)calloc( default_param.syscpucount, sizeof(int) );

  {
    /* initialize the cpu set of virtual kaapi processor to physical cpu */
    unsigned long cpu_mask;
    unsigned long mask;
    int i;
    char* str_mask = getenv("KAAPI_CPUSET");
    if (str_mask==0) {
      cpu_mask = ~0U;
      default_param.usecpuset = 0;
    }
    else {
      cpu_mask = atoi(str_mask);
      default_param.usecpuset = 1;
    }
    CPU_ZERO(&default_param.cpuset);
    
printf("[KAAPI::INIT] use cpu set: \n\t");
for (i=default_param.syscpucount-1; i>=0; --i)
  printf("% 3i",i);
printf("\n\t");
for (i=default_param.syscpucount-1; i>=0; --i)
  if ( ((1UL<<i) & cpu_mask) !=0) printf("  1");
  else printf("  0");
printf("\n");

    kaapi_countcpu = 0;
    for ( i=0; i<default_param.syscpucount; ++i)
    {
      kaapi_cpu2kproc[i] = -1;
      kaapi_kproc2cpu[i] = -1;
      mask = 1UL <<i;
      if ((mask & cpu_mask) !=0)
      {
        kaapi_kproc2cpu[kaapi_countcpu] = i;
        kaapi_cpu2kproc[i] = kaapi_countcpu++;
        CPU_SET( i, &default_param.cpuset );
      }
    }
    default_param.cpucount = kaapi_countcpu;
  }  
printf("[KAAPI::INIT] use #physical cpu:%u\n", default_param.cpucount);
#endif  

  /* initialize the default attr */
  kaapi_attr_init( &kaapi_default_attr );

#if defined(HAVE_NUMA_H)
{
  int k, cnt;
  kaapi_kprocneighbor = malloc(kaapi_countcpu*sizeof(kaapi_kprocneighbor_t));
  for (i=0; i<kaapi_countcpu; ++i)
  {
    kaapi_kprocneighbor[i][KAAPI_L1].count = 1;
    kaapi_kprocneighbor[i][KAAPI_L1].kprocs[0] = i;

    /* recopy physical cpu into set of virtual kproc */
    kaapi_kprocneighbor[i][KAAPI_L2] = idkoiff_level2[ kaapi_kproc2cpu[i] ];
    cnt = kaapi_kprocneighbor[i][KAAPI_L2].count;
    kaapi_kprocneighbor[i][KAAPI_L2].count = 0;
    for (k = 0; k<cnt; ++k)
    {
      if (kaapi_cpu2kproc[kaapi_kprocneighbor[i][KAAPI_L2].kprocs[k]] !=-1)
      { /* physical cpu kprocs[k] is in the set */
        kaapi_kprocneighbor[i][KAAPI_L2].kprocs[kaapi_kprocneighbor[i][KAAPI_L2].count++] 
          = kaapi_cpu2kproc[kaapi_kprocneighbor[i][KAAPI_L2].kprocs[k]];
      }
    }
  }
  /* print topology in term of mapping */
  printf("[KAAPI::INIT] virtual processor : ");
  for (i=0; i<kaapi_countcpu; ++i)
  {
    printf("% 3i",i);
  }
  printf("\n");
  printf("[KAAPI::INIT] physical processor: ");
  for (i=0; i<kaapi_countcpu; ++i)
  {
    printf("% 3i",kaapi_kproc2cpu[i]);
  }
  printf("\n");
  printf("[KAAPI::INIT] numa hierarchy:\n");
  for (i=0; i<kaapi_countcpu; ++i)
  {
    printf("% 3i [% 3i] -> ", i, kaapi_kproc2cpu[i]);
    /* L1 */
    printf("L1: % 3i [% 3i]", i, kaapi_kproc2cpu[i]);
    /* L2 */
    printf("    L2: ");
    for (k=0; k<kaapi_kprocneighbor[i][KAAPI_L2].count; ++k)
      printf("% 3i [% 3i]   ", 
           kaapi_kprocneighbor[i][KAAPI_L2].kprocs[k],
           kaapi_kproc2cpu[kaapi_kprocneighbor[i][KAAPI_L2].kprocs[k]]
      );
    printf("\n");
  }
  printf("[KAAPI::INIT] physical processor: ");
}
#endif

  /* initialize kaapi_min_index_cpu_set */
  kaapi_min_index_cpu_set[0] = -1;
  for (i=1; i<256; ++i)
  {
    if ((i & (1<<0)) !=0) kaapi_min_index_cpu_set[ i ] = 0;
    else if ((i & (1<<1)) !=0) kaapi_min_index_cpu_set[ i ] = 1;
    else if ((i & (1<<2)) !=0) kaapi_min_index_cpu_set[ i ] = 2;
    else if ((i & (1<<3)) !=0) kaapi_min_index_cpu_set[ i ] = 3;
    else if ((i & (1<<4)) !=0) kaapi_min_index_cpu_set[ i ] = 4;
    else if ((i & (1<<5)) !=0) kaapi_min_index_cpu_set[ i ] = 5;
    else if ((i & (1<<6)) !=0) kaapi_min_index_cpu_set[ i ] = 6;
    else if ((i & (1<<7)) !=0) kaapi_min_index_cpu_set[ i ] = 7;
  }
  
  /* steal api part */
  kaapi_stealapi_initialize();
  
  /* initialize kaapi_main_processor */
  kaapi_processor_t* kaapi_main_processor = kaapi_allocate_processor();
  ckaapi_assert( kaapi_main_processor !=0 );
  ckaapi_assert( 0 == pthread_setspecific(kaapi_current_processor_key, kaapi_main_processor ) );

  /* TODO : set to 1  default_param.cpuset; */
  printf("[KAAPI::INIT] Current thread is: %lu\n", (unsigned long)pthread_self() );
  fflush( stdout );
}

/**
*/
void __attribute__ ((destructor)) my_fini(void)
{
  printf("[KAAPI::TERM] Current thread is: %lu\n", (unsigned long)pthread_self() );
  fflush( stdout );
  
  /* steal api part */
  kaapi_stealapi_terminate();

#if defined(KAAPI_USE_SCHED_AFFINITY)
  free(kaapi_kproc2cpu);
  free(kaapi_cpu2kproc);
#endif
}
