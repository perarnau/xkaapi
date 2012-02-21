
#ifndef KAAPI_CUDA_CTX_H_INCLUDED
#define KAAPI_CUDA_CTX_H_INCLUDED

#include <cuda_runtime_api.h>

#include "kaapi_cuda_proc.h"

static inline void
kaapi_cuda_ctx_push( void )
{
    /* TODO future usage. */
}

static inline void
kaapi_cuda_ctx_pop( void )
{
    /* TODO future usage */
}

static inline void
kaapi_cuda_ctx_set( const int dev )
{
    const cudaError_t res= cudaSetDevice( dev );
    if( res != cudaSuccess ) {
	fprintf( stderr, "%s: ERROR %d\n", __FUNCTION__, res );
	fflush( stderr );
    }
}

#endif /* ! KAAPI_CUDA_CTX_H_INCLUDED */
