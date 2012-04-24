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
# define KAAPI_CUDA_PROC_H_INCLUDED


#include <pthread.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#define KAAPI_CUDA_ASYNC	1
#define KAAPI_CUDA_MEM_ALLOC_MANAGER	1
#define KAAPI_CUDA_MEM_FREE_FACTOR	1
//#define	KAAPI_CUDA_MODE_BASIC	1

//#define KAAPI_CUDA_USE_POOL	1

#define KAAPI_CUDA_MAX_STREAMS	4

#define KAAPI_CUDA_SLIDING_WINDOW	4

#define KAAPI_CUDA_KSTREAM	1

struct kaapi_cuda_stream_t;

enum {
    KAAPI_CUDA_STREAM_HTOD,
    KAAPI_CUDA_STREAM_KERNEL,
    KAAPI_CUDA_STREAM_DTOH,
    KAAPI_CUDA_STREAM_DTOD
};

typedef struct kaapi_cuda_ctx
{
	cublasHandle_t handle;
} kaapi_cuda_ctx_t;

struct kaapi_cuda_mem_blk_t;

typedef struct kaapi_cuda_mem
{
	size_t total;
	size_t used;
	struct {
	    struct kaapi_cuda_mem_blk_t* beg;
	    struct kaapi_cuda_mem_blk_t* end;
	} ro;
	struct {
	    struct kaapi_cuda_mem_blk_t* beg;
	    struct kaapi_cuda_mem_blk_t* end;
	} rw;

	/* all GPU allocated pointers */
	kaapi_big_hashmap_t kmem;
} kaapi_cuda_mem_t;

typedef struct kaapi_cuda_proc
{
    unsigned int index;
    struct cudaDeviceProp  deviceProp;

#if 0
    /* WARNING: some old devices get errors on multiple stream */
    cudaStream_t stream[KAAPI_CUDA_MAX_STREAMS];
    cudaEvent_t event;
#endif

    struct kaapi_cuda_stream_t* kstream;

    kaapi_cuda_ctx_t ctx;

    kaapi_cuda_mem_t memory;

    int is_initialized;

    /* cached attribtues */
    unsigned int kasid_user;
    kaapi_address_space_id_t asid;

} kaapi_cuda_proc_t;


int kaapi_cuda_proc_initialize(kaapi_cuda_proc_t*, unsigned int);

int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t*);

size_t kaapi_cuda_get_proc_count(void);

struct kaapi_processor_t;
struct kaapi_processor_t* kaapi_cuda_get_proc_by_dev( unsigned int id );

cudaStream_t kaapi_cuda_kernel_stream(void);

cudaStream_t kaapi_cuda_HtoD_stream(void);

cudaStream_t kaapi_cuda_DtoH_stream(void);

cudaStream_t kaapi_cuda_DtoD_stream(void);

static inline int
kaapi_cuda_sync( void )
{
    const cudaError_t res = cudaDeviceSynchronize( );
    if( res != cudaSuccess ) {
	    fprintf( stdout, "%s: cudaDeviceSynchronize ERROR %d\n", __FUNCTION__, res);
	    fflush(stdout);
    }
    return (int)res;
}

#endif /* ! KAAPI_CUDA_PROC_H_INCLUDED */
