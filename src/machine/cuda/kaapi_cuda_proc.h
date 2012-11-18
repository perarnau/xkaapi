/*
** kaapi_cuda_proc.h
** xkaapi
** 
** Created on Jul 2010
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
** Joao.Lima@imag.fr
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
#ifndef KAAPI_CUDA_PROC_H_INCLUDED
#define KAAPI_CUDA_PROC_H_INCLUDED


#include <pthread.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#define	KAAPI_CUDA_MAX_DEV	16

#define KAAPI_CUDA_ASYNC	1
#define KAAPI_CUDA_MEM_FREE_FACTOR	1

/* Write-through memory cache in the GPU */
//#define KAAPI_CUDA_DATA_CACHE_WT 1

/* all CUDA kprocs indexed by device Id */
extern struct kaapi_processor_t
*kaapi_cuda_all_kprocessors[KAAPI_CUDA_MAX_DEV];
extern uint32_t kaapi_cuda_count_kprocessors;

/* barrier to synchronize all CUDA proc devices */
extern kaapi_atomic_t kaapi_cuda_synchronize_barrier;

struct kaapi_cuda_stream_t;

typedef struct kaapi_cuda_ctx {
  cublasHandle_t handle;
} kaapi_cuda_ctx_t;

struct kaapi_cuda_proc_t;

/*
 Basic interface structure to GPU software cache.
 */
typedef struct kaapi_cuda_mem_cache {
  int (*init)( void** data );
  
  int (*insert)(void *, uintptr_t, size_t, kaapi_access_mode_t);
  
  void* (*remove)(void *, const size_t);
  
  int (*is_full)(void *, const size_t);
  
  int (*inc_use)(void *, uintptr_t, kaapi_memory_view_t* const, const kaapi_access_mode_t);
  
  int (*dec_use)(void *, uintptr_t, kaapi_memory_view_t* const, const kaapi_access_mode_t );
  
  void (*destroy)(void *);
  
  void* data;
} kaapi_cuda_mem_cache_t;

typedef struct kaapi_cuda_proc_t {
  unsigned int index;
  struct cudaDeviceProp deviceProp;
  struct kaapi_cuda_stream_t *kstream;
  kaapi_cuda_ctx_t ctx;
  kaapi_cuda_mem_cache_t cache;

  kaapi_atomic_t synchronize_flag;	/* synchronization flag */

  int is_initialized;

  /* cached attribtues */
  unsigned int kasid_user;
  kaapi_address_space_id_t asid;

  unsigned int peers[KAAPI_CUDA_MAX_DEV];	/* enabled access peer */

} kaapi_cuda_proc_t;

void kaapi_cuda_init(void);

int kaapi_cuda_proc_initialize(kaapi_cuda_proc_t *, unsigned int);

int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t *);

size_t kaapi_cuda_get_proc_count(void);

cudaStream_t kaapi_cuda_kernel_stream(void);

cudaStream_t kaapi_cuda_HtoD_stream(void);

cudaStream_t kaapi_cuda_DtoH_stream(void);

cudaStream_t kaapi_cuda_DtoD_stream(void);

static inline int kaapi_cuda_device_sync(void)
{
  const cudaError_t res = cudaDeviceSynchronize();
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: cudaDeviceSynchronize ERROR %d\n", __FUNCTION__,
	    res);
    fflush(stdout);
  }
  return (int) res;
}

static inline struct kaapi_processor_t *kaapi_cuda_get_proc_by_dev(unsigned
								   int id)
{
  kaapi_assert_debug((id >= 0) && (id < KAAPI_CUDA_MAX_DEV));
  return kaapi_cuda_all_kprocessors[id];
}

extern void kaapi_cuda_stream_poll(struct kaapi_processor_t *const);

/* Polls current operations on GPU cards using kstream from kproc */
extern void kaapi_cuda_proc_poll(struct kaapi_processor_t *const);

/* Test if this kproc CUDA had finished its operations (kstream, etc) */
extern int
kaapi_cuda_proc_end_isvalid(struct kaapi_processor_t *const kproc);

/* Synchronizes all CUDA kprocs */
extern int kaapi_cuda_proc_sync_all(void);

/* Synchronize CUDA kproc operations and memory */
extern int kaapi_cuda_sync(struct kaapi_processor_t *const);

/* Test if all kproc CUDA had finished their operations (kstream, etc) */
extern int kaapi_cuda_proc_all_isvalid(void);

extern void kaapi_cuda_proc_destroy(struct kaapi_processor_t *const);

#endif				/* ! KAAPI_CUDA_PROC_H_INCLUDED */
