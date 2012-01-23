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
# define KAAPI_CUDA_PROC_H_INCLUDED


#include <pthread.h>
#include <sys/types.h>
#include "cuda.h"
#include "cublas_v2.h"

//#define KAAPI_CUDA_USE_POOL	1

#define KAAPI_CUDA_MAX_STREAMS		4
#define KAAPI_CUDA_HTOD_STREAM		0
#define KAAPI_CUDA_KERNEL_STREAM        1
#define KAAPI_CUDA_DTOH_STREAM		2
#define KAAPI_CUDA_DTOD_STREAM		3

typedef struct kaapi_cuda_ctx
{
	CUcontext ctx;
	cublasHandle_t handle;
	pthread_mutex_t mutex;
	pthread_mutexattr_t mta;
} kaapi_cuda_ctx_t;

struct kaapi_cuda_mem_blk_t;

typedef struct kaapi_cuda_mem
{
	size_t total;
	size_t used;
	struct kaapi_cuda_mem_blk_t* beg;
	struct kaapi_cuda_mem_blk_t* end;
} kaapi_cuda_mem_t;

#ifdef KAAPI_CUDA_USE_POOL
struct kaapi_cuda_pool;
#endif

typedef struct kaapi_cuda_proc
{
  CUdevice dev;
  CUstream stream[KAAPI_CUDA_MAX_STREAMS];
  kaapi_cuda_ctx_t ctx;
  kaapi_cuda_mem_t memory;
#ifdef KAAPI_CUDA_USE_POOL
  struct kaapi_cuda_pool* pool;
#endif
  int is_initialized;

  /* cached attribtues */
  unsigned int kasid_user;
  kaapi_address_space_id_t asid;
} kaapi_cuda_proc_t;


int kaapi_cuda_proc_initialize(kaapi_cuda_proc_t*, unsigned int);

int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t*);

size_t kaapi_cuda_get_proc_count(void);

unsigned int kaapi_cuda_get_first_kid(void);

CUstream kaapi_cuda_kernel_stream(void);

CUstream kaapi_cuda_HtoD_stream(void);

CUstream kaapi_cuda_DtoH_stream(void);

CUstream kaapi_cuda_DtoD_stream(void);

#endif /* ! KAAPI_CUDA_PROC_H_INCLUDED */
