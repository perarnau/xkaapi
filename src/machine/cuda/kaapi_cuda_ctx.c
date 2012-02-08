

#include <stdio.h>
#include <pthread.h>
#include <cuda_runtime_api.h>

#include "../../kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_ctx.h"


void
_kaapi_cuda_ctx_push( kaapi_processor_t* proc )
{
#if KAAPI_VERBOSE
	fprintf( stdout, "[%s] kid=%d\n", __FUNCTION__, kaapi_get_self_kid() );
	fflush(stdout);
#endif
	pthread_mutex_lock( &proc->cuda_proc.ctx.mutex );
	const cudaError_t res = cudaSetDevice( proc->cuda_proc.index );
	if( res != cudaSuccess ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
}

void
_kaapi_cuda_ctx_pop( kaapi_processor_t* proc )
{
#if KAAPI_VERBOSE
	fprintf( stdout, "[%s] kid=%d\n", __FUNCTION__, kaapi_get_self_kid() );
	fflush(stdout);
#endif
#if 0
	const cudaError_t res = cuCtxSetCurrent( NULL );
	if( res != cudaSuccess ) {
		fprintf( stderr, "%s ERROR: %d\n", __FUNCTION__, res );
		fflush( stderr );
	}
#endif
	pthread_mutex_unlock( &proc->cuda_proc.ctx.mutex );
}

