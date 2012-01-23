

#include <stdio.h>
#include <pthread.h>

#include "../../kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_utils.h"
//#include "kaapi_cuda_error.h"


void kaapi_cuda_ctx_push( kaapi_cuda_proc_t *proc )
{
#if KAAPI_VERBOSE
	fprintf( stdout, "%s kid=%d\n", __FUNCTION__, kaapi_get_self_kid() ); fflush(stdout);
#endif
	pthread_mutex_lock( &proc->ctx.mutex );
	const CUresult res = cuCtxSetCurrent( proc->ctx.ctx );
	if( res != CUDA_SUCCESS ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
}

void kaapi_cuda_ctx_pop( kaapi_cuda_proc_t *proc )
{
#if KAAPI_VERBOSE
	fprintf( stdout, "%s kid=%d\n", __FUNCTION__, kaapi_get_self_kid() ); fflush(stdout);
#endif
	const CUresult res = cuCtxSetCurrent( NULL );
	if( res != CUDA_SUCCESS ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
	pthread_mutex_unlock( &proc->ctx.mutex );
}

int kaapi_cuda_dev_open( kaapi_cuda_proc_t* proc, unsigned int index )
{
  CUresult res;
  size_t mfree;

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
  pthread_mutexattr_init( &proc->ctx.mta );
  pthread_mutexattr_settype( &proc->ctx.mta, PTHREAD_MUTEX_RECURSIVE );
  pthread_mutex_init( &proc->ctx.mutex, &proc->ctx.mta );


  //res= cuMemGetInfo( &proc->memory.total, &mfree );
  res= cuDeviceTotalMem( &proc->memory.total, proc->dev );
  if (res != CUDA_SUCCESS) {
	if( res != CUDA_SUCCESS ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
    return -1;
  }
  /* 80% of total memory */
  proc->memory.total = 0.8*proc->memory.total;
  proc->memory.used= 0;
  proc->memory.beg = proc->memory.end = NULL;

  return 0;
}

void kaapi_cuda_dev_close( kaapi_cuda_proc_t* proc )
{
	pthread_mutex_destroy( &proc->ctx.mutex );
	cuCtxDestroy( proc->ctx.ctx );
}

kaapi_cuda_proc_t* kaapi_cuda_mem_get_proc( void )
{
	unsigned int i;

	for (i=0; i<kaapi_count_kprocessors; ++i) {
		if( kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA ) {
			return &kaapi_all_kprocessors[i]->cuda_proc;
		}
	}

	return NULL;
}

kaapi_cuda_proc_t* kaapi_cuda_get_proc_by_asid( kaapi_address_space_id_t asid )
{
	unsigned int i;

	for (i=0; i < kaapi_count_kprocessors; ++i) {
		if( (kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA) && 
			 	(kaapi_all_kprocessors[i]->cuda_proc.asid == asid) ) {
			return &kaapi_all_kprocessors[i]->cuda_proc;
		}
	}

	return NULL;
}
