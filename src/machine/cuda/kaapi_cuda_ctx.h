
#ifndef KAAPI_CUDA_CTX_H_INCLUDED
#define KAAPI_CUDA_CTX_H_INCLUDED

#include <cuda_runtime_api.h>

#include "kaapi_cuda_proc.h"

void
_kaapi_cuda_ctx_push( kaapi_processor_t* proc );

void
_kaapi_cuda_ctx_pop( kaapi_processor_t* proc );

static inline void
kaapi_cuda_ctx_push( void )
{
    _kaapi_cuda_ctx_push( kaapi_get_current_processor() );
}

static inline void
kaapi_cuda_ctx_pop( void )
{
    _kaapi_cuda_ctx_pop( kaapi_get_current_processor() );
}

static inline void
kaapi_cuda_ctx_set( const int dev )
{
    const cudaError_t res= cudaSetDevice( dev );
    if( res != cudaSuccess ) {
	fprintf( stderr, "[%s] ERROR: %d\n", __FUNCTION__, res );
	fflush( stderr );
    }
}

#endif /* ! KAAPI_CUDA_CTX_H_INCLUDED */
