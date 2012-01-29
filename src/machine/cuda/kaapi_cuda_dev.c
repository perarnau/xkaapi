

#include <stdio.h>
#include <pthread.h>
#include <cuda_runtime_api.h>

#include "../../kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_dev.h"

int
kaapi_cuda_dev_open( kaapi_cuda_proc_t* proc, unsigned int index )
{
  cudaError_t res;
  size_t mfree;
  struct cudaDeviceProp prop;

  proc->index = index;
#if 0
  res = cuDeviceGet( &proc->dev, index );
  if (res != CUDA_SUCCESS)
  {
	if( res != CUDA_SUCCESS ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
    return -1;
  }

  /* use sched_yield while waiting for sync.
     context is made current for the thread.
   */
  res = cuCtxCreate( &proc->ctx.ctx, CU_CTX_SCHED_YIELD, proc->dev );
  if (res != CUDA_SUCCESS) {
	if( res != CUDA_SUCCESS ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
    return -1;
  }
#endif
  pthread_mutexattr_init( &proc->ctx.mta );
  pthread_mutexattr_settype( &proc->ctx.mta, PTHREAD_MUTEX_RECURSIVE );
  pthread_mutex_init( &proc->ctx.mutex, &proc->ctx.mta );


  //res= cuMemGetInfo( &proc->memory.total, &mfree );
#if 0
  res= cuDeviceTotalMem( &proc->memory.total, proc->dev );
#endif
  res = cudaGetDeviceProperties( &prop, index );
  if (res != CUDA_SUCCESS) {
	if( res != CUDA_SUCCESS ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
    return -1;
  }
  /* 80% of total memory */
  proc->memory.total = 0.8*prop.totalGlobalMem;
  proc->memory.used= 0;
  proc->memory.beg = proc->memory.end = NULL;

  return 0;
}

void
kaapi_cuda_dev_close( kaapi_cuda_proc_t* proc )
{
	pthread_mutex_destroy( &proc->ctx.mutex );
#if (CUDART_VERSION >= 4010)
	cudaDeviceReset();
#endif
//	cuCtxDestroy( proc->ctx.ctx );
}

kaapi_processor_t*
kaapi_cuda_mem_get_proc( void )
{
	unsigned int i;

	for (i=0; i<kaapi_count_kprocessors; ++i) {
		if( kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA ) {
			return kaapi_all_kprocessors[i];
		}
	}

	return NULL;
}

kaapi_processor_t*
kaapi_cuda_get_proc_by_asid( kaapi_address_space_id_t asid )
{
	unsigned int i;

	for (i=0; i < kaapi_count_kprocessors; ++i) {
		if( (kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA) && 
			 	(kaapi_all_kprocessors[i]->cuda_proc.asid == asid) ) {
			return kaapi_all_kprocessors[i];
		}
	}

	return NULL;
}
